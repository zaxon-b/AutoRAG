"""Extract native PDF image assets and every displayed occurrence.

The old ingestion path used an image block's bbox to crop a rendered page.
That mixed the object with page background and neighbouring text.  This module
keeps the PDF's native image bytes as the visual asset and stores page/bbox/
bookmark context separately as occurrences.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from service.bookmark_outline_service import section_at_position
from service.hierarchical_chunk_service import stable_chunk_id
from service.visual_region_extractor import nearby_text_candidates


def extract_native_image_catalog(
    *,
    pdf_path: Path,
    output_dir: Path,
    manual_id: str,
    pages: list[dict[str, Any]],
    bookmark_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact-deduplicated assets and all 1-based PDF occurrences."""

    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    page_metadata = {int(page["page_number"]): page for page in pages}
    assets_by_exact_key: dict[str, dict[str, Any]] = {}
    occurrences: list[dict[str, Any]] = []
    document = fitz.open(str(pdf_path))
    try:
        for page_number, page in enumerate(document, start=1):
            page_info = page_metadata.get(page_number, {})
            smask_by_xref = {
                int(item[0]): int(item[1] or 0)
                for item in page.get_images(full=True)
                if item and int(item[0] or 0) > 0
            }
            inline_blocks = [
                block
                for block in page.get_text("dict").get("blocks", [])
                if block.get("type") == 1 and block.get("image")
            ]
            for source_index, info in enumerate(
                page.get_image_info(hashes=True, xrefs=True),
                start=1,
            ):
                bbox = [float(value) for value in info.get("bbox") or []]
                if len(bbox) != 4:
                    continue
                digest = _digest_hex(info.get("digest"))
                xref = int(info.get("xref") or 0)
                pixel_width = int(info.get("width") or 0)
                pixel_height = int(info.get("height") or 0)
                exact_key = _exact_asset_key(
                    digest=digest,
                    xref=xref,
                    width=pixel_width,
                    height=pixel_height,
                    page_number=page_number,
                    source_index=source_index,
                )
                asset = assets_by_exact_key.get(exact_key)
                if asset is None:
                    asset_id = stable_chunk_id(manual_id, "image_asset", exact_key)
                    image_path = output_dir / f"{asset_id}.png"
                    smask_xref = smask_by_xref.get(xref, 0)
                    extracted = _save_native_image(
                        document=document,
                        xref=xref,
                        smask_xref=smask_xref,
                        inline_blocks=inline_blocks,
                        bbox=bbox,
                        target=image_path,
                    )
                    noise_reason = _noise_reason(image_path) if extracted else "native_image_unreadable"
                    asset = {
                        "asset_id": asset_id,
                        "manual_id": manual_id,
                        "source_xrefs": [],
                        "source_smask_xrefs": [],
                        "digest": digest or None,
                        "pixel_width": pixel_width or None,
                        "pixel_height": pixel_height or None,
                        "native_image_path": str(image_path) if extracted else None,
                        "classification": "ignore" if noise_reason else None,
                        "classification_confidence": 1.0 if noise_reason else None,
                        "classification_reason": noise_reason,
                        "noise_reason": noise_reason,
                        "phash": _perceptual_hash(image_path) if extracted else None,
                        "dominant_colour": _dominant_colour(image_path) if extracted else None,
                        "quality_score": _image_quality(image_path) if extracted else 0.0,
                        "occurrence_ids": [],
                    }
                    assets_by_exact_key[exact_key] = asset

                smask_xref = smask_by_xref.get(xref, 0)
                if xref > 0 and xref not in asset["source_xrefs"]:
                    asset["source_xrefs"].append(xref)
                if smask_xref > 0 and smask_xref not in asset["source_smask_xrefs"]:
                    asset["source_smask_xrefs"].append(smask_xref)

                transform = [float(value) for value in info.get("transform") or []]
                orientation = orientation_signature(transform)
                center_y = (bbox[1] + bbox[3]) / 2
                section = section_at_position(bookmark_nodes, page_number, center_y)
                nearby = nearby_text_candidates(bbox, page_info.get("words") or [])
                occurrence_seed = (
                    f"{page_number}|{source_index}|"
                    f"{','.join(f'{value:.3f}' for value in bbox)}|{exact_key}"
                )
                occurrence_id = stable_chunk_id(manual_id, "image_occurrence", occurrence_seed)
                occurrence = {
                    "occurrence_id": occurrence_id,
                    "manual_id": manual_id,
                    "asset_id": asset["asset_id"],
                    "page_number": page_number,
                    "bbox": bbox,
                    "transform": transform or None,
                    "orientation": orientation,
                    "section_title": section.get("path") if section else None,
                    "nearby_text": nearby,
                    "name_candidates": _clean_name_candidates(nearby),
                    "content_text": None,
                    "icon_group_id": None,
                    "visual_chunk_id": None,
                    "page_width": float(page.rect.width),
                    "page_height": float(page.rect.height),
                }
                occurrences.append(occurrence)
                asset["occurrence_ids"].append(occurrence_id)
    finally:
        document.close()

    assets = sorted(assets_by_exact_key.values(), key=lambda item: item["asset_id"])
    occurrences.sort(key=lambda item: (item["page_number"], item["bbox"][1], item["bbox"][0]))
    return assets, occurrences


def orientation_signature(transform: list[float] | tuple[float, ...] | None) -> str:
    """Reduce a PDF image transform to rotation plus reflection identity."""

    if not transform or len(transform) < 4:
        return "r0"
    a, b, c, d = [float(value) for value in transform[:4]]
    angle = int(round(math.degrees(math.atan2(b, a)) / 90.0) * 90) % 360
    reflected = a * d - b * c < 0
    return f"r{angle}{'_mirror' if reflected else ''}"


def oriented_image_path(
    asset: dict[str, Any],
    orientation: str,
    output_dir: Path,
) -> str | None:
    """Return a display-oriented prototype image without page pixels."""

    source_value = asset.get("native_image_path")
    if not source_value:
        return None
    source = Path(str(source_value))
    if not source.exists() or orientation == "r0":
        return str(source) if source.exists() else None
    target = output_dir / f"{asset['asset_id']}_{orientation}.png"
    if target.exists():
        return str(target)
    try:
        import cv2

        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            return str(source)
        if "_mirror" in orientation:
            image = cv2.flip(image, 1)
        angle = int(orientation.split("_", 1)[0].removeprefix("r") or 0)
        if angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target) if cv2.imwrite(str(target), image) else str(source)
    except Exception:
        return str(source)


def _save_native_image(
    *,
    document,
    xref: int,
    smask_xref: int,
    inline_blocks: list[dict[str, Any]],
    bbox: list[float],
    target: Path,
) -> bool:
    try:
        import fitz

        if xref > 0:
            base = fitz.Pixmap(document, xref)
            pixmap = base
            if smask_xref > 0:
                mask = fitz.Pixmap(document, smask_xref)
                if mask.width == base.width and mask.height == base.height:
                    pixmap = fitz.Pixmap(base, mask)
            if pixmap.colorspace and pixmap.colorspace.n > 3:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            pixmap.save(str(target))
            return target.exists() and target.stat().st_size > 0

        inline = _matching_inline_block(inline_blocks, bbox)
        if inline is None:
            return False
        image_bytes = bytes(inline.get("image") or b"")
        if not image_bytes:
            return False
        pixmap = fitz.Pixmap(image_bytes)
        if pixmap.colorspace and pixmap.colorspace.n > 3:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        pixmap.save(str(target))
        return target.exists() and target.stat().st_size > 0
    except Exception:
        target.unlink(missing_ok=True)
        return False


def _matching_inline_block(
    blocks: list[dict[str, Any]],
    bbox: list[float],
) -> dict[str, Any] | None:
    candidates = [
        (_bbox_iou(bbox, [float(value) for value in block.get("bbox") or []]), block)
        for block in blocks
        if len(block.get("bbox") or []) == 4
    ]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _noise_reason(path: Path) -> str | None:
    """Only remove deterministic non-content; semantic decoration goes to VLM."""

    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            return "empty_image"
        height, width = image.shape[:2]
        if width < 2 or height < 2:
            return "empty_image"
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            visible = alpha > 5
            if int(visible.sum()) == 0:
                return "fully_transparent"
            pixels = image[:, :, :3][visible]
        else:
            pixels = image.reshape(-1, image.shape[2] if image.ndim == 3 else 1)
        if pixels.size == 0:
            return "empty_image"
        if float(np.std(pixels.astype("float32"))) < 1.0:
            return "uniform_fill"
        return None
    except Exception:
        return "damaged_image"


def _perceptual_hash(path: Path) -> str | None:
    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        foreground = None
        if image.ndim == 3 and image.shape[2] == 4:
            foreground = image[:, :, 3] > 10
            alpha = image[:, :, 3:4].astype("float32") / 255.0
            rgb = image[:, :, :3].astype("float32")
            image = (rgb * alpha + 255.0 * (1.0 - alpha)).astype("uint8")
        elif image.ndim == 3:
            foreground = _border_difference_mask(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = _crop_to_foreground(gray, foreground)
        gray = _pad_square(gray, 255)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))[:8, :8]
        median = float(np.median(dct.reshape(-1)[1:]))
        bits = "".join("1" if float(value) > median else "0" for value in dct.reshape(-1))
        return f"{int(bits, 2):016x}"
    except Exception:
        return None


def _border_difference_mask(image):
    import numpy as np

    border = np.concatenate(
        [
            image[0, :, :3],
            image[-1, :, :3],
            image[:, 0, :3],
            image[:, -1, :3],
        ],
        axis=0,
    )
    background = np.median(border.astype("float32"), axis=0)
    difference = np.max(
        np.abs(image[:, :, :3].astype("float32") - background),
        axis=2,
    )
    mask = difference > 12.0
    return mask if 0 < int(mask.sum()) < mask.size * 0.98 else None


def _crop_to_foreground(image, mask):
    if mask is None:
        return image
    try:
        import cv2
        import numpy as np

        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            return image
        x, y, width, height = cv2.boundingRect(coords)
        return image[y : y + height, x : x + width]
    except Exception:
        return image


def _pad_square(image, value: int):
    import numpy as np

    height, width = image.shape[:2]
    size = max(height, width)
    canvas = np.full((size, size), value, dtype=image.dtype)
    top = (size - height) // 2
    left = (size - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def _dominant_colour(path: Path) -> dict[str, float] | None:
    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        alpha_mask = None
        if image.ndim == 3 and image.shape[2] == 4:
            alpha_mask = image[:, :, 3] > 20
            image = image[:, :, :3]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] >= 45
        if alpha_mask is not None:
            mask &= alpha_mask
        if int(mask.sum()) < 12:
            return {"hue": 0.0, "saturation": 0.0}
        hue_values = hsv[:, :, 0][mask].astype("float32")
        saturation = float(hsv[:, :, 1][mask].mean())
        angles = hue_values * (2.0 * math.pi / 180.0)
        hue = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
        hue = (hue % (2.0 * math.pi)) * 180.0 / (2.0 * math.pi)
        return {"hue": round(hue, 3), "saturation": round(saturation, 3)}
    except Exception:
        return None


def _image_quality(path: Path) -> float:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return 0.0
        height, width = image.shape[:2]
        gray_source = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
        gray = cv2.cvtColor(gray_source, cv2.COLOR_BGR2GRAY) if gray_source.ndim == 3 else gray_source
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        resolution = min(1.0, math.log1p(width * height) / math.log1p(300 * 300))
        sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(1500.0))
        return round(0.55 * resolution + 0.45 * sharpness_score, 6)
    except Exception:
        return 0.0


def _clean_name_candidates(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip(" ：:，,。")
        compact = "".join(text.split())
        if 2 <= len(compact) <= 64 and text not in result:
            result.append(text)
    return result[:12]


def _exact_asset_key(
    *,
    digest: str,
    xref: int,
    width: int,
    height: int,
    page_number: int,
    source_index: int,
) -> str:
    if digest:
        return f"digest:{digest}:{width}x{height}"
    if xref > 0:
        return f"xref:{xref}:{width}x{height}"
    return f"inline:{page_number}:{source_index}:{width}x{height}"


def _digest_hex(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
