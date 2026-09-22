"""Re-run grounded VLM naming for persisted operation figures.

This is useful after changing the bounded naming prompt without rebuilding PDF
layout, crops, text chunks or vectors.
"""

from __future__ import annotations

import argparse
import os

from elasticsearch.helpers import bulk

from models.manual import ManualChunk
from service.es_index_service import INDEX_NAME, _es_client, ensure_index
from service.visual_object_naming_service import assign_visual_object_names
from utils.database import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh operation_figure object_name values.")
    parser.add_argument("--manual-id", default=os.getenv("DEMO_MANUAL_ID", "demo_manual"))
    args = parser.parse_args()

    db = SessionLocal()
    try:
        rows = (
            db.query(ManualChunk)
            .filter(
                ManualChunk.manual_id == args.manual_id,
                ManualChunk.modality == "operation_figure",
            )
            .order_by(ManualChunk.page_number, ManualChunk.chunk_id)
            .all()
        )
        payloads = [
            {
                "chunk_id": row.chunk_id,
                "modality": row.modality,
                "section_title": row.section_title,
                "content_text": row.content_text,
                "image_path": row.image_path,
                "_name_candidates": [],
            }
            for row in rows
        ]
        assign_visual_object_names(payloads)
        by_id = {payload["chunk_id"]: payload.get("object_name") for payload in payloads}
        for row in rows:
            row.object_name = by_id.get(row.chunk_id)
        db.commit()
    finally:
        db.close()

    ensure_index()
    actions = [
        {
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": payload["chunk_id"],
            "doc": {"object_name": payload.get("object_name")},
        }
        for payload in payloads
    ]
    success, errors = bulk(
        _es_client(),
        actions,
        refresh=True,
        raise_on_error=False,
        raise_on_exception=False,
    )
    named = sum(bool(payload.get("object_name")) for payload in payloads)
    print(
        f"Operation naming complete: total={len(payloads)} named={named} "
        f"unnamed={len(payloads) - named} es_updated={success} es_errors={len(errors)}",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
