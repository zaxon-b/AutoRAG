"""Associate icon/figure crops with real bookmark paths and nearby manual text."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from service.bookmark_outline_service import section_at_position
from service.hierarchical_chunk_service import count_tokens


VISUAL_CONTENT_TOKEN_LIMIT = int(os.getenv("MANUAL_VISUAL_CONTENT_TOKEN_LIMIT", "1400"))


def associate_icon_content(
    region: dict[str, Any],
    bookmark_nodes: list[dict[str, Any]],
    blocks_by_page: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Return strict manual name and text that directly explains this icon occurrence."""

    page = int(region["page"])
    bbox = [float(value) for value in region["bbox"]]
    section = section_at_position(bookmark_nodes, page, (bbox[1] + bbox[3]) / 2)
    section_id = str(section["node_id"]) if section else None
    candidates = _manual_name_candidates(region.get("label_candidates") or region.get("nearby_texts") or [])
    object_name = None

    related = _related_icon_blocks(
        page=page,
        bbox=bbox,
        section_id=section_id,
        object_name=object_name,
        bookmark_nodes=bookmark_nodes,
        blocks_by_page=blocks_by_page,
    )
    nearby = [str(value) for value in region.get("nearby_texts") or [] if str(value).strip()]
    content_text = _bounded_join([*related, *nearby], VISUAL_CONTENT_TOKEN_LIMIT)
    return {
        "section_title": section.get("path") if section else None,
        "object_name": object_name,
        "content_text": content_text or None,
        "name_candidates": candidates,
    }


def associate_operation_content(
    figure: dict[str, Any],
    all_figures: list[dict[str, Any]],
    bookmark_nodes: list[dict[str, Any]],
    blocks_by_page: dict[int, list[dict[str, Any]]],
    image_path: Path,
) -> dict[str, Any]:
    """Return a local figure name plus corresponding body and applicable callouts."""

    page = int(figure["page"])
    bbox = [float(value) for value in figure["bbox"]]
    center_y = (bbox[1] + bbox[3]) / 2
    section = section_at_position(bookmark_nodes, page, center_y)
    section_id = str(section["node_id"]) if section else None
    peers = [
        item
        for item in all_figures
        if _section_id_for_figure(item, bookmark_nodes) == section_id
    ]
    peers.sort(key=lambda item: (int(item["page"]), float(item["bbox"][1])))
    local_blocks = _operation_text_window(figure, peers, section_id, bookmark_nodes, blocks_by_page)
    caption_candidates = _caption_candidates(figure, section_id, bookmark_nodes, blocks_by_page)
    object_name = None
    return {
        "section_title": section.get("path") if section else None,
        "object_name": object_name[:255] if object_name else None,
        "content_text": _bounded_join(local_blocks, VISUAL_CONTENT_TOKEN_LIMIT) or None,
        "name_candidates": caption_candidates,
    }


def _related_icon_blocks(
    *,
    page: int,
    bbox: list[float],
    section_id: str | None,
    object_name: str | None,
    bookmark_nodes: list[dict[str, Any]],
    blocks_by_page: dict[int, list[dict[str, Any]]],
) -> list[str]:
    x0, y0, x1, y1 = bbox
    candidates: list[tuple[float, int, float, str]] = []
    for block in blocks_by_page.get(page, []):
        if block.get("kind") != "text":
            continue
        if _is_page_decoration(block):
            continue
        block_section = section_at_position(bookmark_nodes, page, float(block["bbox"][1]))
        if section_id and (not block_section or str(block_section["node_id"]) != section_id):
            continue
        text = _clean_text(str(block.get("text") or ""))
        if not text or _is_section_label(text, block_section):
            continue
        bx0, by0, bx1, by1 = [float(value) for value in block["bbox"]]
        vertical_overlap = max(0.0, min(y1, by1) - max(y0, by0))
        same_row = vertical_overlap > 0 or abs(((by0 + by1) / 2) - ((y0 + y1) / 2)) <= 28
        gap = _rectangle_gap((x0, y0, x1, y1), (bx0, by0, bx1, by1))
        contains_name = bool(object_name and _normalize(object_name) in _normalize(text))
        if contains_name or same_row or gap <= 95:
            priority = 0.0 if contains_name else 1.0 if same_row else 2.0
            candidates.append((priority, int(block["page"]), by0, text))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen = candidates[:5]
    return _document_order([item[3] for item in chosen], [item[1:3] for item in chosen])


def _operation_text_window(
    figure: dict[str, Any],
    peers: list[dict[str, Any]],
    section_id: str | None,
    bookmark_nodes: list[dict[str, Any]],
    blocks_by_page: dict[int, list[dict[str, Any]]],
) -> list[str]:
    current_position = _figure_position(figure)
    index = next((idx for idx, item in enumerate(peers) if item is figure or item.get("layout_id") == figure.get("layout_id")), 0)
    previous_position = _figure_position(peers[index - 1]) if index > 0 else None
    next_position = _figure_position(peers[index + 1]) if index + 1 < len(peers) else None

    selected: list[tuple[int, float, str]] = []
    for page, blocks in sorted(blocks_by_page.items()):
        for block in blocks:
            if block.get("kind") != "text":
                continue
            if _is_page_decoration(block):
                continue
            y = float(block["bbox"][1])
            block_section = section_at_position(bookmark_nodes, page, y)
            if section_id and (not block_section or str(block_section["node_id"]) != section_id):
                continue
            text = _clean_text(str(block.get("text") or ""))
            if not text or _is_section_label(text, block_section):
                continue
            position = (page, y)
            if previous_position and position < _mid_position(previous_position, current_position):
                continue
            if next_position and position >= _mid_position(current_position, next_position):
                continue
            selected.append((page, y, text))
    selected.sort(key=lambda item: (item[0], item[1]))
    return [text for _, _, text in selected]


def _caption_candidates(
    figure: dict[str, Any],
    section_id: str | None,
    bookmark_nodes: list[dict[str, Any]],
    blocks_by_page: dict[int, list[dict[str, Any]]],
) -> list[str]:
    page = int(figure["page"])
    x0, y0, x1, y1 = [float(value) for value in figure["bbox"]]
    candidates: list[tuple[float, str]] = []
    for block in blocks_by_page.get(page, []):
        if block.get("kind") != "text":
            continue
        if _is_page_decoration(block):
            continue
        block_section = section_at_position(bookmark_nodes, page, float(block["bbox"][1]))
        if section_id and (not block_section or str(block_section["node_id"]) != section_id):
            continue
        text = _clean_text(str(block.get("text") or ""))
        compact = "".join(text.split())
        if not (2 <= len(compact) <= 80) or any(mark in text for mark in ("。", "；", "。")):
            continue
        bx0, by0, bx1, by1 = [float(value) for value in block["bbox"]]
        horizontal_overlap = max(0.0, min(x1, bx1) - max(x0, bx0))
        vertical_gap = min(abs(by0 - y1), abs(y0 - by1))
        if horizontal_overlap > 0 and vertical_gap <= 55:
            candidates.append((vertical_gap, text.strip(" ：:")))
    return [text for _, text in sorted(candidates, key=lambda item: item[0])]


def _manual_name_candidates(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        source = _clean_text(str(value or ""))
        for part in re.split(r"[，。；;：:]", source):
            candidate = part.strip(" -—·•●■□()（）[]【】")
            compact = "".join(candidate.split())
            if not (2 <= len(compact) <= 48):
                continue
            if candidate not in result:
                result.append(candidate)
    return result[:16]


def _section_id_for_figure(figure: dict[str, Any], bookmark_nodes: list[dict[str, Any]]) -> str | None:
    bbox = figure["bbox"]
    section = section_at_position(bookmark_nodes, int(figure["page"]), (float(bbox[1]) + float(bbox[3])) / 2)
    return str(section["node_id"]) if section else None


def _figure_position(figure: dict[str, Any]) -> tuple[int, float]:
    bbox = figure["bbox"]
    return int(figure["page"]), (float(bbox[1]) + float(bbox[3])) / 2


def _mid_position(left: tuple[int, float], right: tuple[int, float]) -> tuple[int, float]:
    if left[0] == right[0]:
        return left[0], (left[1] + right[1]) / 2
    return right[0], right[1] / 2


def _rectangle_gap(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    horizontal = max(0.0, max(left[0], right[0]) - min(left[2], right[2]))
    vertical = max(0.0, max(left[1], right[1]) - min(left[3], right[3]))
    return (horizontal**2 + vertical**2) ** 0.5


def _bounded_join(values: list[str], token_limit: int) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = _normalize(text)
        if not text or not key or key in seen:
            continue
        candidate = "\n".join([*result, text])
        if result and count_tokens(candidate) > token_limit:
            break
        result.append(text)
        seen.add(key)
    return "\n".join(result)


def _document_order(values: list[str], positions: list[tuple[int, float]]) -> list[str]:
    return [value for _, value in sorted(zip(positions, values), key=lambda item: item[0])]


def _is_section_label(text: str, section: dict[str, Any] | None) -> bool:
    return bool(section and _normalize(text) == _normalize(str(section.get("title") or "")))


def _is_page_decoration(block: dict[str, Any]) -> bool:
    x0, y0, x1, y1 = [float(value) for value in block.get("bbox") or [0, 0, 0, 0]]
    page_width = float(block.get("page_width") or 0)
    page_height = float(block.get("page_height") or 0)
    if y1 <= 55.0 or (page_height and y0 >= page_height * 0.95):
        return True
    return bool(page_width and (x0 >= page_width * 0.92 or x1 <= page_width * 0.08))


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()
