"""Crop icon chunks from small image objects embedded in the PDF.

The owner manual stores icons as individual raster objects.  Using those PDF
objects avoids mistaking text strokes, bullets, page numbers or callout borders
for icons.  No name suffix or page keyword is used as a gate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def should_extract_visual_regions(_page_text: str = "") -> bool:
    return os.getenv("MANUAL_ICON_EXTRACTION_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
    }


def extract_visual_regions(
    page_image_path: Path,
    output_dir: Path,
    page_number: int,
    page_text: str,
    words: list[dict] | None = None,
    pdf_width: float | int | None = None,
    pdf_height: float | int | None = None,
    image_blocks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return small embedded PDF images as generic icon candidates."""

    if not should_extract_visual_regions(page_text):
        return []
    try:
        import cv2
    except Exception:
        return []

    image = cv2.imread(str(page_image_path))
    if image is None:
        return []
    image_height, image_width = image.shape[:2]
    pdf_width = float(pdf_width or image_width)
    pdf_height = float(pdf_height or image_height)
    scale_x = image_width / max(pdf_width, 1.0)
    scale_y = image_height / max(pdf_height, 1.0)

    pdf_boxes = [
        [float(value) for value in block["bbox"]]
        for block in image_blocks or []
        if block.get("kind") == "image"
        and _is_icon_image_block(block, pdf_width, pdf_height)
    ]
    pdf_boxes = _deduplicate_pdf_boxes(pdf_boxes)
    output_dir.mkdir(parents=True, exist_ok=True)

    regions: list[dict[str, Any]] = []
    max_per_page = int(os.getenv("MANUAL_ICON_MAX_PER_PAGE", "64"))
    for index, pdf_box in enumerate(pdf_boxes[:max_per_page], start=1):
        x0, y0, x1, y1 = pdf_box
        padding_x = max(2, int((x1 - x0) * scale_x * 0.08))
        padding_y = max(2, int((y1 - y0) * scale_y * 0.08))
        left = max(0, int(x0 * scale_x) - padding_x)
        top = max(0, int(y0 * scale_y) - padding_y)
        right = min(image_width, int(x1 * scale_x) + padding_x)
        bottom = min(image_height, int(y1 * scale_y) + padding_y)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        crop = _scale_small_crop(cv2, crop)
        crop_path = output_dir / f"page_{page_number:03d}_icon_{index:03d}.png"
        if not cv2.imwrite(str(crop_path), crop):
            continue

        nearby = nearby_text_candidates(pdf_box, words or [])
        regions.append(
            {
                "region_id": f"p{page_number:04d}_icon_{index:04d}",
                "modality": "icon",
                "page": page_number,
                "bbox": pdf_box,
                "label_candidates": nearby[:8],
                "nearby_texts": nearby,
                "risk_level": _risk_level_for_text(" ".join(nearby)),
                "image_path": str(crop_path),
                "thumbnail_path": str(crop_path),
            }
        )
    return regions


def _is_icon_image_block(
    block: dict[str, Any],
    page_width: float,
    page_height: float,
) -> bool:
    bbox = block.get("bbox") or []
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox]
    width, height = x1 - x0, y1 - y0
    if width < 8 or height < 8:
        return False
    if width > page_width * 0.18 or height > page_height * 0.14:
        return False
    area_ratio = width * height / max(1.0, page_width * page_height)
    if area_ratio > 0.025:
        return False
    aspect = width / max(height, 1.0)
    if not 0.18 <= aspect <= 5.5:
        return False
    if y1 < page_height * 0.04 or y0 > page_height * 0.96:
        return False
    return True


def _deduplicate_pdf_boxes(boxes: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for box in sorted(boxes, key=lambda value: (value[1], value[0])):
        if any(_intersection_over_union(box, existing) >= 0.85 for existing in result):
            continue
        result.append(box)
    return result


def _intersection_over_union(left: list[float], right: list[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def nearby_text_candidates(pdf_box: list[float], words: list[dict]) -> list[str]:
    """Return spatially ranked text rows around one PDF-coordinate image bbox."""

    x0, y0, x1, y1 = pdf_box
    center_y = (y0 + y1) / 2
    row_tolerance = max(10.0, (y1 - y0) * 0.8)
    rows: dict[tuple[int, str], list[tuple[float, str]]] = {}
    row_gaps: dict[tuple[int, str], float] = {}
    for word in words:
        text = " ".join(str(word.get("text") or "").split()).strip()
        if not text:
            continue
        wx0, wx1 = float(word.get("x0") or 0), float(word.get("x1") or 0)
        wy0, wy1 = float(word.get("top") or 0), float(word.get("bottom") or 0)
        word_center_y = (wy0 + wy1) / 2
        same_row = (
            abs(word_center_y - center_y) <= row_tolerance
            and wx0 <= x1 + 160
            and wx1 >= x0 - 160
        )
        near_vertical = (
            min(abs(wy0 - y1), abs(y0 - wy1)) <= 35
            and wx1 >= x0 - 100
            and wx0 <= x1 + 160
        )
        if not (same_row or near_vertical):
            continue
        row_key = int(round(word_center_y / 4.0) * 4)
        if wx0 >= x1 - 3:
            side = "right"
            gap = max(0.0, wx0 - x1)
        elif wx1 <= x0 + 3:
            side = "left"
            gap = max(0.0, x0 - wx1)
        else:
            side = "overlap"
            gap = 0.0
        key = (row_key, side)
        rows.setdefault(key, []).append((wx0, text))
        row_gaps[key] = min(row_gaps.get(key, float("inf")), gap)

    candidates: list[str] = []
    side_priority = {"right": 0, "overlap": 1, "left": 2}
    for key in sorted(
        rows,
        key=lambda value: (
            abs(value[0] - center_y),
            side_priority[value[1]],
            row_gaps[value],
        ),
    ):
        value = "".join(text for _, text in sorted(rows[key]))
        value = value.strip(" |:：,，。；;")
        if value and value not in candidates:
            candidates.append(value[:240])
    return _join_wrapped_name_candidate(candidates)


# Compatibility alias for older unit tests/imports.
_nearby_text_candidates = nearby_text_candidates


def _join_wrapped_name_candidate(candidates: list[str]) -> list[str]:
    """Join a standard label split across two PDF text lines.

    Manual table cells commonly wrap ``自动紧急制动故障警告灯*`` as
    ``自动紧急制动故障警告`` and ``灯*``.  Only a short, known suffix is
    joined; ordinary neighbouring rows remain separate candidates.
    """

    if len(candidates) < 2:
        return candidates
    suffix = "".join(candidates[1].split())
    if suffix not in {"灯", "灯*", "告灯", "告灯*", "警告灯", "警告灯*", "指示灯", "指示灯*"}:
        return candidates
    first_compact = "".join(candidates[0].split())
    if first_compact.rstrip("*").endswith(("灯", "指示灯", "警告灯")):
        return candidates
    joined = f"{candidates[0].rstrip()}{candidates[1].lstrip()}"
    return [joined, *candidates[2:]]


def _scale_small_crop(cv2, crop):
    height, width = crop.shape[:2]
    longest = max(width, height)
    if longest >= 128:
        return crop
    scale = max(2, int(128 / max(longest, 1)))
    return cv2.resize(
        crop,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )


def _risk_level_for_text(text: str) -> str:
    if any(keyword in text for keyword in ("制动", "电池", "动力", "高压", "安全气囊", "转向")):
        return "high"
    if any(keyword in text for keyword in ("故障", "警告", "报警", "过热", "低压", "危险")):
        return "medium"
    return "low"
