"""Read the PDF's real bookmark tree and ordered page blocks.

Bookmark levels create the chapter hierarchy. Font size and colour are used only
to locate a bookmark label on its target page when the PDF destination has no
usable y coordinate; they never create a new heading level.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def extract_ordered_layout(pdf_path: Path) -> tuple[int, dict[int, list[dict[str, Any]]]]:
    """Return ordered text/image blocks in PDF coordinates, keyed by 1-based page."""

    import fitz

    document = fitz.open(str(pdf_path))
    pages: dict[int, list[dict[str, Any]]] = {}
    try:
        for page_number, page in enumerate(document, start=1):
            blocks: list[dict[str, Any]] = []
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            for source_index, block in enumerate(page.get_text("dict").get("blocks", []), start=1):
                bbox = [float(value) for value in block.get("bbox", [])]
                if len(bbox) != 4:
                    continue
                if block.get("type") == 0:
                    lines: list[str] = []
                    max_font_size = 0.0
                    for line in block.get("lines", []):
                        line_text = "".join(str(span.get("text") or "") for span in line.get("spans", []))
                        line_text = _clean_text(line_text)
                        if line_text:
                            lines.append(line_text)
                        for span in line.get("spans", []):
                            max_font_size = max(max_font_size, float(span.get("size") or 0.0))
                    text = "\n".join(lines).strip()
                    if not text:
                        continue
                    blocks.append(
                        {
                            "block_id": f"p{page_number:04d}_text_{source_index:04d}",
                            "page": page_number,
                            "kind": "text",
                            "bbox": bbox,
                            "text": text,
                            "max_font_size": max_font_size,
                            "page_width": page_width,
                            "page_height": page_height,
                        }
                    )
                elif block.get("type") == 1:
                    blocks.append(
                        {
                            "block_id": f"p{page_number:04d}_image_{source_index:04d}",
                            "page": page_number,
                            "kind": "image",
                            "bbox": bbox,
                            "text": "",
                            "width": int(block.get("width") or 0),
                            "height": int(block.get("height") or 0),
                            "page_width": page_width,
                            "page_height": page_height,
                        }
                    )
            pages[page_number] = sorted(
                blocks,
                key=lambda item: (
                    round(float(item["bbox"][1]), 1),
                    round(float(item["bbox"][0]), 1),
                    item["block_id"],
                ),
            )
    finally:
        document.close()
    return len(pages), pages


def extract_bookmark_outline(
    pdf_path: Path,
    blocks_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a normalized bookmark tree with stable paths and chapter ranges."""

    import fitz

    document = fitz.open(str(pdf_path))
    warnings: list[str] = []
    try:
        toc = document.get_toc(simple=False) or []
        if not toc:
            return [], ["PDF 没有书签，正文将按页降级切分。"]

        page_heights = {index + 1: float(page.rect.height) for index, page in enumerate(document)}
        nodes: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []
        last_y_by_page: dict[int, float] = {}
        used_title_blocks: set[str] = set()

        seen_destinations: set[tuple[str, int, int]] = set()
        for ordinal, entry in enumerate(toc, start=1):
            if not _has_valid_destination(entry):
                warnings.append(f"忽略无有效页码的书签“{_clean_text(str(entry[1] or ''))}”。")
                continue
            level = max(1, int(entry[0] or 1))
            title = _clean_text(str(entry[1] or ""))
            page = _bookmark_page(entry, len(document))
            page_height = page_heights.get(page, 0.0)
            destination_y = _locate_title_y(
                title,
                page,
                blocks_by_page or {},
                minimum_y=last_y_by_page.get(page, -1.0),
                used_block_ids=used_title_blocks,
            )
            if destination_y is None:
                destination_y = _bookmark_y(entry, page_height)
            if destination_y is None:
                destination_y = 0.0
                warnings.append(f"书签“{title}”第 {page} 页无法定位，按页顶处理。")
            destination_y = max(0.0, min(float(destination_y), page_heights.get(page, float(destination_y))))
            destination_key = (_normalize_for_match(title), page, round(destination_y))
            if destination_key in seen_destinations:
                continue
            seen_destinations.add(destination_key)
            last_y_by_page[page] = max(last_y_by_page.get(page, -1.0), destination_y)

            while stack and int(stack[-1]["level"]) >= level:
                stack.pop()
            parent_id = stack[-1]["node_id"] if stack else None
            path_titles: list[str] = []
            for value in [*(str(item["title"]) for item in stack), title]:
                if not path_titles or _normalize_for_match(path_titles[-1]) != _normalize_for_match(value):
                    path_titles.append(value)
            node = {
                "node_id": f"bookmark_{ordinal:04d}",
                "ordinal": ordinal,
                "level": level,
                "title": title,
                "path": " > ".join(part for part in path_titles if part) or None,
                "parent_id": parent_id,
                "children_ids": [],
                "page": page,
                "y": destination_y,
                "end_page": len(document),
                "end_y": page_heights.get(len(document), 0.0),
            }
            if stack:
                stack[-1]["children_ids"].append(node["node_id"])
            nodes.append(node)
            stack.append(node)

        for index, node in enumerate(nodes):
            for following in nodes[index + 1 :]:
                if int(following["level"]) <= int(node["level"]):
                    node["end_page"] = int(following["page"])
                    node["end_y"] = float(following["y"])
                    break
        return nodes, warnings
    finally:
        document.close()


def section_at_position(nodes: list[dict[str, Any]], page: int, y: float = 0.0) -> dict[str, Any] | None:
    """Return the deepest bookmark active at a page/y position."""

    position = (int(page), float(y))
    active: list[dict[str, Any]] = []
    for node in nodes:
        start = (int(node["page"]), float(node["y"]))
        end = (int(node["end_page"]), float(node["end_y"]))
        if start <= position < end:
            active.append(node)
    if not active:
        previous = [node for node in nodes if (int(node["page"]), float(node["y"])) <= position]
        return previous[-1] if previous else None
    return max(active, key=lambda item: (int(item["level"]), int(item["ordinal"])))


def _bookmark_page(entry: list[Any], page_count: int) -> int:
    destination = entry[3] if len(entry) > 3 and isinstance(entry[3], dict) else {}
    zero_based = destination.get("page")
    if isinstance(zero_based, int) and zero_based >= 0:
        return min(page_count, zero_based + 1)
    try:
        return max(1, min(page_count, int(entry[2] or 1)))
    except (TypeError, ValueError):
        return 1


def _has_valid_destination(entry: list[Any]) -> bool:
    destination = entry[3] if len(entry) > 3 and isinstance(entry[3], dict) else {}
    if isinstance(destination.get("page"), int) and int(destination["page"]) >= 0:
        return True
    try:
        return int(entry[2]) > 0
    except (TypeError, ValueError, IndexError):
        return False


def _bookmark_y(entry: list[Any], page_height: float) -> float | None:
    destination = entry[3] if len(entry) > 3 and isinstance(entry[3], dict) else {}
    point = destination.get("to")
    value = getattr(point, "y", None)
    if value is None and isinstance(point, (tuple, list)) and len(point) >= 2:
        value = point[1]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    # This manual stores outline destinations in bottom-origin PDF coordinates,
    # while PyMuPDF text blocks use top-origin coordinates.
    return max(0.0, page_height - number) if page_height else number


def _locate_title_y(
    title: str,
    page: int,
    blocks_by_page: dict[int, list[dict[str, Any]]],
    *,
    minimum_y: float,
    used_block_ids: set[str],
) -> float | None:
    normalized_title = _normalize_for_match(title)
    if not normalized_title:
        return None
    candidates: list[tuple[float, str]] = []
    for block in blocks_by_page.get(page, []):
        if block.get("kind") != "text" or block.get("block_id") in used_block_ids:
            continue
        normalized_block = _normalize_for_match(str(block.get("text") or ""))
        if not normalized_block:
            continue
        if normalized_title in normalized_block or normalized_block in normalized_title:
            y = float(block["bbox"][1])
            if y + 2.0 >= minimum_y:
                candidates.append((y, str(block["block_id"])))
    if not candidates:
        return None
    y, block_id = min(candidates, key=lambda item: item[0])
    used_block_ids.add(block_id)
    return y


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())
