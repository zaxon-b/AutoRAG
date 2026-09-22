"""Backfill cross-modal vectors for chunks that already exist in Elasticsearch.

The script reads chunk metadata from PostgreSQL, generates vectors with the
same batching rules used during normal PDF ingestion, and incrementally updates
only the ``multimodal_vector`` field in Elasticsearch. Existing vectors are
skipped by default, so an interrupted run can safely be resumed.

Run inside the application container::

    python -m backfill_multimodal_vectors --manual-id demo_manual
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Iterable

from elasticsearch.helpers import bulk, scan

from models.manual import ManualChunk
from service.es_index_service import INDEX_NAME, _es_client, ensure_index
from service.multimodal_embedding_service import (
    MAX_MULTIMODAL_TEXT_BATCH_SIZE,
    MULTIMODAL_EMBEDDING_FIELD,
    embed_multimodal_chunks,
)
from utils.database import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill multimodal vectors for an indexed owner manual.")
    parser.add_argument("--manual-id", default=os.getenv("DEMO_MANUAL_ID", "demo_manual"))
    parser.add_argument("--batch-size", type=int, default=MAX_MULTIMODAL_TEXT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N missing chunks; 0 means all.")
    parser.add_argument("--force", action="store_true", help="Regenerate vectors that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned work without calling the API.")
    return parser.parse_args()


def _existing_vector_ids(es: Any, manual_id: str) -> set[str]:
    query = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"manual_id": manual_id}},
                    {"exists": {"field": MULTIMODAL_EMBEDDING_FIELD}},
                ]
            }
        },
        "_source": False,
    }
    return {
        str(hit["_id"])
        for hit in scan(es, index=INDEX_NAME, query=query, preserve_order=False)
    }


def _payload(chunk: ManualChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "manual_id": chunk.manual_id,
        "page": chunk.page_number,
        "modality": chunk.modality,
        "section_title": chunk.section_title,
        "content_text": chunk.content_text,
        "object_name": chunk.object_name,
        "image_path": chunk.image_path,
    }


def _batches(items: list[ManualChunk], size: int) -> Iterable[list[ManualChunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    if not 1 <= args.batch_size <= MAX_MULTIMODAL_TEXT_BATCH_SIZE:
        raise ValueError(f"--batch-size must be between 1 and {MAX_MULTIMODAL_TEXT_BATCH_SIZE}")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    ensure_index()
    es = _es_client()
    existing_ids = set() if args.force else _existing_vector_ids(es, args.manual_id)

    db = SessionLocal()
    try:
        chunks = (
            db.query(ManualChunk)
            .filter(
                ManualChunk.manual_id == args.manual_id,
                ManualChunk.modality.in_(("icon", "operation_figure", "page_snapshot")),
            )
            .order_by(ManualChunk.id)
            .all()
        )
    finally:
        db.close()

    targets = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    if args.limit:
        targets = targets[: args.limit]

    print(
        f"Multimodal backfill plan: manual_id={args.manual_id} total_chunks={len(chunks)} "
        f"already_indexed={len(existing_ids)} targets={len(targets)} force={args.force}",
        flush=True,
    )
    if args.dry_run or not targets:
        return

    started = time.monotonic()
    processed = 0
    indexed = 0
    failed_ids: list[str] = []
    for batch_number, chunk_batch in enumerate(_batches(targets, args.batch_size), start=1):
        payloads = [_payload(chunk) for chunk in chunk_batch]
        vectors = embed_multimodal_chunks(payloads)
        actions = []
        for chunk, vector in zip(chunk_batch, vectors):
            if vector is None:
                failed_ids.append(chunk.chunk_id)
                continue
            actions.append(
                {
                    "_op_type": "update",
                    "_index": INDEX_NAME,
                    "_id": chunk.chunk_id,
                    "doc": {MULTIMODAL_EMBEDDING_FIELD: vector},
                }
            )

        if actions:
            success_count, errors = bulk(
                es,
                actions,
                refresh=False,
                raise_on_error=False,
                raise_on_exception=False,
            )
            indexed += int(success_count)
            for error in errors:
                update_error = error.get("update") or error
                chunk_id = str(update_error.get("_id") or "unknown")
                failed_ids.append(chunk_id)
                print(f"Elasticsearch vector update failed: chunk_id={chunk_id} error={update_error}", flush=True)

        processed += len(chunk_batch)
        elapsed = time.monotonic() - started
        print(
            f"[batch {batch_number}] processed={processed}/{len(targets)} indexed={indexed} "
            f"failed={len(failed_ids)} elapsed={elapsed:.1f}s",
            flush=True,
        )

    es.indices.refresh(index=INDEX_NAME)
    final_count = es.count(
        index=INDEX_NAME,
        query={
            "bool": {
                "filter": [
                    {"term": {"manual_id": args.manual_id}},
                    {"exists": {"field": MULTIMODAL_EMBEDDING_FIELD}},
                ]
            }
        },
    )["count"]
    elapsed = time.monotonic() - started
    print(
        f"Multimodal backfill complete: indexed_this_run={indexed} total_indexed={final_count}/{len(chunks)} "
        f"failed={len(failed_ids)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    if failed_ids:
        print("Failed chunk IDs: " + ",".join(failed_ids), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
