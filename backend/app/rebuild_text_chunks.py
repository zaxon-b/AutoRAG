"""Rebuild only bookmark text chunks and their Elasticsearch documents."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import delete

from models.manual import Manual, ManualChunk
from service.bookmark_outline_service import extract_bookmark_outline
from service.embedding_service import embed_texts
from service.es_index_service import INDEX_NAME, _es_client, ensure_index, index_chunks
from service.hierarchical_chunk_service import build_hierarchical_text_chunks
from service.layout_analysis_service import analyze_pdf_layout
from service.manual_ingestion_service import _payload_to_row
from utils.database import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild text chunks without reprocessing visual chunks.")
    parser.add_argument("--manual-id", default=os.getenv("DEMO_MANUAL_ID", "demo_manual"))
    args = parser.parse_args()

    db = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.manual_id == args.manual_id).one()
        pdf_path = Path(manual.file_path)
        layout_by_page = analyze_pdf_layout(pdf_path)
        bookmark_nodes, warnings = extract_bookmark_outline(pdf_path, layout_by_page)
        payloads = build_hierarchical_text_chunks(args.manual_id, bookmark_nodes, layout_by_page)

        db.execute(
            delete(ManualChunk).where(
                ManualChunk.manual_id == args.manual_id,
                ManualChunk.modality == "text",
            )
        )
        db.add_all([_payload_to_row(payload) for payload in payloads])
        db.commit()
    finally:
        db.close()

    ensure_index()
    es = _es_client()
    es.delete_by_query(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"manual_id": args.manual_id}},
                        {"term": {"modality": "text"}},
                    ]
                }
            }
        },
        conflicts="proceed",
        refresh=True,
    )
    leaves = [payload for payload in payloads if payload.get("is_leaf") is True]
    search_texts = [
        f"{payload.get('section_title') or ''}\n{payload.get('content_text') or ''}"
        for payload in leaves
    ]
    for payload, vector in zip(leaves, embed_texts(search_texts)):
        payload["text_vector"] = vector
    index_chunks(leaves)

    parents = sum(payload.get("is_leaf") is False for payload in payloads)
    print(
        f"Text rebuild complete: leaves={len(leaves)} parents={parents} "
        f"ignored_outline_warnings={len(warnings)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
