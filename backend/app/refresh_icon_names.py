"""Refresh strict icon-group names from the persisted native visual catalog.

The script reads icon groups and their occurrences from PostgreSQL, re-runs the
existing literal-name selection for every occurrence, and writes the chosen
name to the group, its single icon chunk and the matching Elasticsearch
document. It never changes group/chunk/ES identities, vectors or grouping.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

from elasticsearch.helpers import bulk

from models.manual import (
    Manual,
    ManualChunk,
    ManualIconGroup,
    ManualImageAsset,
    ManualImageOccurrence,
)
from service.es_index_service import INDEX_NAME, _es_client, ensure_index
from service.icon_catalog_service import clear_icon_group_cache
from service.visual_object_naming_service import assign_visual_object_names
from utils.database import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh strict manual icon group names.")
    parser.add_argument("--manual-id", default=os.getenv("DEMO_MANUAL_ID", "demo_manual"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned names without calling the VLM or writing PG/ES.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = SessionLocal()
    plan: list[dict[str, Any]] = []
    try:
        db.query(Manual).filter(Manual.manual_id == args.manual_id).one()

        groups = (
            db.query(ManualIconGroup)
            .filter(ManualIconGroup.manual_id == args.manual_id)
            .order_by(ManualIconGroup.icon_group_id)
            .all()
        )
        icon_chunks = {
            str(row.icon_group_id): row
            for row in db.query(ManualChunk)
            .filter(
                ManualChunk.manual_id == args.manual_id,
                ManualChunk.modality == "icon",
                ManualChunk.icon_group_id.isnot(None),
            )
            .all()
        }
        asset_paths = {
            str(asset.asset_id): asset.native_image_path
            for asset in db.query(ManualImageAsset)
            .filter(ManualImageAsset.manual_id == args.manual_id)
            .all()
        }
        occurrences = (
            db.query(ManualImageOccurrence)
            .filter(
                ManualImageOccurrence.manual_id == args.manual_id,
                ManualImageOccurrence.icon_group_id.isnot(None),
            )
            .order_by(ManualImageOccurrence.page_number)
            .all()
        )
        occurrences_by_group: dict[str, list[ManualImageOccurrence]] = {}
        for occurrence in occurrences:
            occurrences_by_group.setdefault(str(occurrence.icon_group_id), []).append(occurrence)

        for group in groups:
            group_id = str(group.icon_group_id)
            group_occurrences = occurrences_by_group.get(group_id, [])
            payloads = [
                {
                    "chunk_id": occurrence.occurrence_id,
                    "modality": "icon",
                    "image_path": asset_paths.get(str(occurrence.asset_id)),
                    "section_title": occurrence.section_title,
                    "content_text": occurrence.content_text,
                    "_name_candidates": list(occurrence.name_candidates or []),
                }
                for occurrence in group_occurrences
            ]
            if not args.dry_run:
                assign_visual_object_names(payloads)
            names = [
                str(payload["object_name"])
                for payload in payloads
                if payload.get("object_name")
            ]
            canonical = Counter(names).most_common(1)[0][0] if names else None
            plan.append(
                {
                    "group_id": group_id,
                    "chunk_id": icon_chunks.get(group_id).chunk_id if group_id in icon_chunks else None,
                    "current_name": group.canonical_icon_name,
                    "new_name": canonical or group.canonical_icon_name,
                    "occurrences": len(group_occurrences),
                }
            )

        if args.dry_run:
            for item in plan:
                print(
                    f"{item['group_id']} chunk={item['chunk_id']} "
                    f"occurrences={item['occurrences']} "
                    f"name={item['current_name']!r} -> {item['new_name']!r}",
                    flush=True,
                )
            return

        for item in plan:
            group = next(group for group in groups if str(group.icon_group_id) == item["group_id"])
            chunk = icon_chunks.get(item["group_id"])
            if item["new_name"]:
                group.canonical_icon_name = item["new_name"]
                if chunk is not None:
                    chunk.object_name = item["new_name"]
            # A group's member count is its occurrence count, not the number of
            # icon chunks (which is exactly one after the group/Chunk merge).
            group.member_count = item["occurrences"]
        db.commit()
    finally:
        db.close()

    if args.dry_run:
        return

    updates = [
        (item["chunk_id"], item["new_name"])
        for item in plan
        if item["chunk_id"] and item["new_name"]
    ]
    ensure_index()
    actions = [
        {
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": chunk_id,
            "doc": {"object_name": object_name},
        }
        for chunk_id, object_name in updates
    ]
    success, errors = bulk(
        _es_client(),
        actions,
        refresh=True,
        raise_on_error=False,
        raise_on_exception=False,
    )
    clear_icon_group_cache()
    named = sum(bool(item["new_name"]) for item in plan)
    print(
        f"Icon naming complete: groups={len(plan)} named={named} "
        f"occurrences={sum(item['occurrences'] for item in plan)} "
        f"es_updated={success} es_errors={len(errors)}",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
