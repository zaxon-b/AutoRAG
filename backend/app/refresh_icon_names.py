"""Re-extract local manual candidates and refresh strict icon names/groups."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

from elasticsearch.helpers import bulk

from models.manual import Manual, ManualChunk, ManualIconGroup, ManualPage
from service.bookmark_outline_service import extract_bookmark_outline
from service.es_index_service import INDEX_NAME, _es_client, ensure_index
from service.hierarchical_chunk_service import stable_chunk_id
from service.layout_analysis_service import analyze_pdf_layout
from service.manual_ingestion_service import _extract_pdf_pages
from service.visual_content_association_service import associate_icon_content
from service.visual_object_naming_service import assign_visual_object_names
from service.visual_region_extractor import extract_visual_regions
from utils.database import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh strict manual icon object names.")
    parser.add_argument("--manual-id", default=os.getenv("DEMO_MANUAL_ID", "demo_manual"))
    args = parser.parse_args()

    db = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.manual_id == args.manual_id).one()
        pdf_path = Path(manual.file_path)
        _, pages = _extract_pdf_pages(pdf_path)
        page_by_number = {int(page["page_number"]): page for page in pages}
        rendered_pages = {
            int(page.page_number): {"image_path": page.image_path}
            for page in db.query(ManualPage).filter(ManualPage.manual_id == args.manual_id).all()
        }
        layout_by_page = analyze_pdf_layout(pdf_path)
        bookmark_nodes, _ = extract_bookmark_outline(pdf_path, layout_by_page)

        payloads: list[dict[str, Any]] = []
        for page_number, rendered in sorted(rendered_pages.items()):
            page_info = page_by_number.get(page_number)
            if not page_info or not rendered.get("image_path"):
                continue
            regions = extract_visual_regions(
                page_image_path=Path(str(rendered["image_path"])),
                output_dir=pdf_path.parent / "crops",
                page_number=page_number,
                page_text=str(page_info.get("text") or ""),
                words=page_info.get("words") or [],
                pdf_width=page_info.get("width"),
                pdf_height=page_info.get("height"),
                image_blocks=layout_by_page.get(page_number, []),
            )
            for region in regions:
                metadata = associate_icon_content(region, bookmark_nodes, layout_by_page)
                payloads.append(
                    {
                        "chunk_id": stable_chunk_id(args.manual_id, "icon", region["region_id"]),
                        "modality": "icon",
                        "section_title": metadata["section_title"],
                        "content_text": metadata["content_text"],
                        "image_path": region["image_path"],
                        "_name_candidates": metadata["name_candidates"],
                    }
                )

        rows = (
            db.query(ManualChunk)
            .filter(ManualChunk.manual_id == args.manual_id, ManualChunk.modality == "icon")
            .all()
        )
        row_by_id = {row.chunk_id: row for row in rows}
        payloads = [payload for payload in payloads if payload["chunk_id"] in row_by_id]
        if len(payloads) != len(rows):
            raise RuntimeError(f"Icon extraction mismatch: database={len(rows)} extracted={len(payloads)}")

        assign_visual_object_names(payloads)
        for payload in payloads:
            row_by_id[payload["chunk_id"]].object_name = payload.get("object_name")

        groups = {
            group.icon_group_id: group
            for group in db.query(ManualIconGroup).filter(ManualIconGroup.manual_id == args.manual_id).all()
        }
        members_by_group: dict[str, list[ManualChunk]] = {}
        for row in rows:
            if row.icon_group_id:
                members_by_group.setdefault(row.icon_group_id, []).append(row)
        for group_id, members in members_by_group.items():
            values = [member.object_name for member in members if member.object_name]
            canonical = Counter(values).most_common(1)[0][0] if values else None
            if group_id in groups:
                groups[group_id].canonical_icon_name = canonical
                groups[group_id].member_count = len(members)
            if canonical:
                for member in members:
                    if not member.object_name:
                        member.object_name = canonical
        db.commit()
        final_names = {row.chunk_id: row.object_name for row in rows}
    finally:
        db.close()

    ensure_index()
    actions = [
        {
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": chunk_id,
            "doc": {"object_name": object_name},
        }
        for chunk_id, object_name in final_names.items()
    ]
    success, errors = bulk(
        _es_client(),
        actions,
        refresh=True,
        raise_on_error=False,
        raise_on_exception=False,
    )
    named = sum(bool(value) for value in final_names.values())
    print(
        f"Icon naming complete: total={len(final_names)} named={named} "
        f"unnamed={len(final_names) - named} es_updated={success} es_errors={len(errors)}",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
