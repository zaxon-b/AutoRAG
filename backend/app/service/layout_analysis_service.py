"""Ordered PDF layout extraction without inventing heading or warning chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from service.bookmark_outline_service import extract_ordered_layout


def analyze_pdf_layout(pdf_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Return text and image blocks in reading order.

    The function deliberately does not classify large text as a heading and does
    not emit warning chunks. PDF bookmarks are the only chapter hierarchy.
    """

    _, blocks_by_page = extract_ordered_layout(pdf_path)
    return blocks_by_page


def operation_figure_candidates(
    blocks_by_page: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Filter embedded image blocks down to plausible operation/structure figures."""

    result: dict[int, list[dict[str, Any]]] = {}
    for page, blocks in blocks_by_page.items():
        candidates: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("kind") != "image":
                continue
            x0, y0, x1, y1 = [float(value) for value in block["bbox"]]
            width = max(0.0, x1 - x0)
            height = max(0.0, y1 - y0)
            page_width = float(block.get("page_width") or 1.0)
            page_height = float(block.get("page_height") or 1.0)
            area_ratio = (width * height) / max(1.0, page_width * page_height)

            # Remove logos, header/footer decoration, tiny raster fragments and
            # page-sized backgrounds. Icon-sized regions are handled separately.
            if width < 55 or height < 45 or area_ratio < 0.008 or area_ratio > 0.72:
                continue
            if y1 < page_height * 0.08 or y0 > page_height * 0.92:
                continue
            if height < page_height * 0.12 and width > height * 3.5:
                continue
            candidates.append(
                {
                    "layout_id": str(block["block_id"]),
                    "page": page,
                    "bbox": list(block["bbox"]),
                    "page_width": page_width,
                    "page_height": page_height,
                }
            )
        if candidates:
            result[page] = candidates
    return result
