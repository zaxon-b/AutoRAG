"""Migrate an existing icon index to one Elasticsearch document per icon group.

Older builds indexed every persisted prototype as its own ES document keyed by
``prototype_id``. The current schema stores one icon chunk per group, so the
migration builds one document per persisted group and keys it by ``chunk_id``.
It copies the vectors already present on the old prototype document, so no
embedding API call is required.

The command is inspectable (``--dry-run``), writes the Group -> Chunk mapping
used to convert visual gold, and does not delete the whole index or touch other
manuals.  Run it after rebuilding the manual with the current ingestion code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from elasticsearch.helpers import bulk

from models.manual import ManualChunk, ManualIconGroup, ManualIconPrototype
from service.es_index_service import INDEX_NAME, _es_client, ensure_index
from service.image_embedding_service import (
    IMAGE_EMBEDDING_FIELD,
    IMAGE_EMBEDDING_V2_FIELD,
)
from service.multimodal_embedding_service import MULTIMODAL_EMBEDDING_FIELD
from utils.database import SessionLocal

VECTOR_FIELDS = (
    IMAGE_EMBEDDING_FIELD,
    IMAGE_EMBEDDING_V2_FIELD,
    MULTIMODAL_EMBEDDING_FIELD,
)
DOCUMENT_FIELDS = (
    "chunk_id",
    "manual_id",
    "modality",
    "page",
    "section_title",
    "content_text",
    "object_name",
    "icon_group_id",
    "asset_id",
    "risk_level",
    "bbox",
    "image_path",
    "thumbnail_path",
    *VECTOR_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild icon ES documents as one document per group.")
    parser.add_argument("--manual-id", default=os.getenv("DEMO_MANUAL_ID", "demo_manual"))
    parser.add_argument("--index", default=INDEX_NAME, help="Target index; defaults to the configured index.")
    parser.add_argument("--dry-run", action="store_true", help="Print the migration plan without writing.")
    parser.add_argument("--mapping-out", type=Path, help="Write the Group -> Chunk mapping JSON here.")
    return parser.parse_args()


def _current_icon_documents(es: Any, index: str, manual_id: str) -> list[dict[str, Any]]:
    response = es.search(
        index=index,
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"manual_id": manual_id}},
                        {"term": {"modality": "icon"}},
                    ]
                }
            },
            "size": 10_000,
            "_source": True,
        },
    )
    return [
        {"_id": hit.get("_id"), **dict(hit.get("_source") or {})}
        for hit in response.get("hits", {}).get("hits", [])
        if hit.get("_id")
    ]


def _target_documents(db, manual_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    groups = (
        db.query(ManualIconGroup)
        .filter(ManualIconGroup.manual_id == manual_id)
        .order_by(ManualIconGroup.icon_group_id)
        .all()
    )
    chunks = {
        str(row.icon_group_id): row
        for row in db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality == "icon",
            ManualChunk.icon_group_id.isnot(None),
        )
        .all()
    }
    prototypes = {
        str(row.icon_group_id): row
        for row in db.query(ManualIconPrototype)
        .filter(
            ManualIconPrototype.manual_id == manual_id,
            ManualIconPrototype.prototype_rank == 1,
        )
        .all()
    }

    documents: list[dict[str, Any]] = []
    problems: list[str] = []
    for group in groups:
        group_id = str(group.icon_group_id)
        chunk = chunks.get(group_id)
        if chunk is None:
            problems.append(f"{group_id}: no icon chunk")
            continue
        prototype = prototypes.get(group_id)
        documents.append(
            {
                "chunk_id": chunk.chunk_id,
                "manual_id": manual_id,
                "modality": "icon",
                "page": chunk.page_number,
                "source_pages": chunk.source_pages,
                "section_title": chunk.section_title,
                "content_text": chunk.content_text,
                "object_name": chunk.object_name or group.canonical_icon_name,
                "icon_group_id": group_id,
                "asset_id": prototype.asset_id if prototype else None,
                "risk_level": chunk.risk_level,
                "bbox": chunk.bbox,
                "image_path": (prototype.image_path if prototype else chunk.image_path),
                "thumbnail_path": (prototype.image_path if prototype else chunk.thumbnail_path),
            }
        )
    return documents, problems


def _copy_vectors(
    document: dict[str, Any],
    old_documents: list[dict[str, Any]],
    prototype_id: str | None,
) -> dict[str, Any]:
    group_id = str(document.get("icon_group_id") or "")
    candidates = [
        old
        for old in old_documents
        if str(old.get("icon_group_id") or "") == group_id
    ]
    source = next(
        (old for old in candidates if prototype_id and old.get("_id") == prototype_id),
        candidates[0] if candidates else None,
    )
    if source is None:
        return document
    for field in VECTOR_FIELDS:
        if source.get(field) is not None:
            document[field] = source[field]
    return document


def main() -> None:
    args = parse_args()
    ensure_index()
    es = _es_client()

    db = SessionLocal()
    try:
        documents, problems = _target_documents(db, args.manual_id)
        prototypes = {
            str(row.icon_group_id): row.prototype_id
            for row in db.query(ManualIconPrototype)
            .filter(
                ManualIconPrototype.manual_id == args.manual_id,
                ManualIconPrototype.prototype_rank == 1,
            )
            .all()
        }
    finally:
        db.close()

    old_documents = _current_icon_documents(es, args.index, args.manual_id)
    documents = [
        _copy_vectors(document, old_documents, prototypes.get(str(document["icon_group_id"])))
        for document in documents
    ]
    new_ids = {str(document["chunk_id"]) for document in documents}
    stale_ids = [
        str(document["_id"])
        for document in old_documents
        if str(document["_id"]) not in new_ids
    ]
    mapping = {
        str(document["icon_group_id"]): str(document["chunk_id"])
        for document in documents
    }

    print(
        f"Icon index migration plan: source={INDEX_NAME} target={args.index} "
        f"manual_id={args.manual_id} old_icon_docs={len(old_documents)} "
        f"new_icon_docs={len(documents)} stale_docs={len(stale_ids)} "
        f"missing_chunks={len(problems)} dry_run={args.dry_run}",
        flush=True,
    )
    for problem in problems:
        print(f"  unresolved: {problem}", flush=True)
    if args.mapping_out:
        if args.dry_run:
            print(f"  mapping would be written to {args.mapping_out}", flush=True)
        else:
            args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
            args.mapping_out.write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"  mapping written to {args.mapping_out}", flush=True)
    if args.dry_run:
        return

    if stale_ids:
        es.delete_by_query(
            index=args.index,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"manual_id": args.manual_id}},
                            {"term": {"modality": "icon"}},
                            {"ids": {"values": stale_ids}},
                        ]
                    }
                }
            },
            conflicts="proceed",
            refresh=True,
        )
    actions = [
        {
            "_op_type": "index",
            "_index": args.index,
            "_id": document["chunk_id"],
            "_source": {
                key: value for key, value in document.items() if key in DOCUMENT_FIELDS and value is not None
            },
        }
        for document in documents
    ]
    if actions:
        bulk(es, actions, refresh=True, raise_on_error=True)
    print(
        f"Icon index migration complete: indexed={len(actions)} stale_deleted={len(stale_ids)}",
        flush=True,
    )
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
