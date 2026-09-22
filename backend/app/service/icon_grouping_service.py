"""Build persisted icon groups with pHash, visual vectors and batched VLM review."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import delete, update
from sqlalchemy.orm import Session

from models.manual import ManualChunk, ManualIconGroup


logger = logging.getLogger(__name__)

EXACT_PHASH_DISTANCE = max(0, int(os.getenv("ICON_GROUP_EXACT_PHASH_DISTANCE", "3")))
PHASH_CANDIDATE_DISTANCE = max(EXACT_PHASH_DISTANCE, int(os.getenv("ICON_GROUP_PHASH_DISTANCE", "10")))
VECTOR_CANDIDATE_SIMILARITY = float(os.getenv("ICON_GROUP_VECTOR_SIMILARITY", "0.88"))
MAX_VECTOR_NEIGHBOURS = max(1, int(os.getenv("ICON_GROUP_MAX_NEIGHBOURS", "5")))
VLM_GROUP_CONFIDENCE = float(os.getenv("ICON_GROUP_VLM_CONFIDENCE", "0.82"))
MAX_REVIEW_ITEMS = max(4, min(64, int(os.getenv("ICON_GROUP_VLM_MAX_ITEMS", "36"))))


def build_icon_groups(
    db: Session,
    manual_id: str,
    image_vectors: dict[str, list[float] | None] | None = None,
) -> int:
    """Group icon occurrences and store the group ID directly on each chunk."""

    rows = (
        db.query(ManualChunk)
        .filter(ManualChunk.manual_id == manual_id, ManualChunk.modality == "icon")
        .order_by(ManualChunk.page_number, ManualChunk.chunk_id)
        .all()
    )
    db.execute(
        update(ManualChunk)
        .where(ManualChunk.manual_id == manual_id, ManualChunk.modality == "icon")
        .values(icon_group_id=None)
    )
    db.execute(delete(ManualIconGroup).where(ManualIconGroup.manual_id == manual_id))
    db.flush()
    if not rows:
        db.commit()
        return 0

    vectors = image_vectors or {}
    row_by_id = {row.chunk_id: row for row in rows}
    features = {row.chunk_id: _icon_features(row, vectors.get(row.chunk_id)) for row in rows}
    candidate_pairs = list(_candidate_pairs(rows, features))

    # First collapse only almost-identical crops. This safe fallback also works
    # when no VLM is configured; looser pHash/vector neighbours still require
    # the visual-language review below.
    exact_parent = {row.chunk_id: row.chunk_id for row in rows}
    for left_id, right_id, signals in candidate_pairs:
        if signals["phash_distance"] <= EXACT_PHASH_DISTANCE and not signals["colour_conflict"]:
            _union(exact_parent, left_id, right_id)

    units: dict[str, list[ManualChunk]] = {}
    for row in rows:
        units.setdefault(_find(exact_parent, row.chunk_id), []).append(row)
    representative = {
        unit_id: _representative(members)
        for unit_id, members in units.items()
    }

    unit_edges: set[tuple[str, str]] = set()
    for left_id, right_id, signals in candidate_pairs:
        if signals["colour_conflict"]:
            continue
        left_unit, right_unit = _find(exact_parent, left_id), _find(exact_parent, right_id)
        if left_unit != right_unit:
            unit_edges.add(tuple(sorted((left_unit, right_unit))))

    final_unit_groups: list[list[str]] = []
    for component in _graph_components(list(units), unit_edges):
        if len(component) == 1:
            final_unit_groups.append(component)
            continue
        for review_batch in _bounded_review_batches(component, unit_edges):
            final_unit_groups.extend(_review_partition(review_batch, representative))

    member_groups: list[list[ManualChunk]] = []
    for unit_group in final_unit_groups:
        members = [member for unit_id in unit_group for member in units[unit_id]]
        members.sort(key=lambda row: (row.page_number, row.chunk_id))
        member_groups.append(members)

    for members in member_groups:
        group_id = _stable_group_id(manual_id, members)
        canonical_name = _canonical_name(members)
        representative_row = _representative(members)
        db.add(
            ManualIconGroup(
                icon_group_id=group_id,
                manual_id=manual_id,
                canonical_icon_name=canonical_name,
                representative_chunk_id=representative_row.chunk_id,
                member_count=len(members),
            )
        )
        db.flush()
        for member in members:
            member.icon_group_id = group_id
            if not member.object_name and canonical_name:
                member.object_name = canonical_name
    db.commit()
    return len(member_groups)


def _icon_features(row: ManualChunk, vector: list[float] | None) -> dict[str, Any]:
    path = Path(str(row.image_path or ""))
    return {
        "phash": _perceptual_hash(path),
        "vector": vector,
        "colour": _dominant_colour(path),
    }


def _candidate_pairs(
    rows: list[ManualChunk],
    features: dict[str, dict[str, Any]],
):
    candidates: list[tuple[float, str, str, dict[str, Any]]] = []
    for index, left in enumerate(rows):
        vector_neighbours: list[tuple[float, ManualChunk]] = []
        for right in rows[index + 1 :]:
            left_feature, right_feature = features[left.chunk_id], features[right.chunk_id]
            phash_distance = _hamming(left_feature["phash"], right_feature["phash"])
            similarity = _cosine(left_feature["vector"], right_feature["vector"])
            if similarity is not None:
                vector_neighbours.append((similarity, right))
            if phash_distance <= PHASH_CANDIDATE_DISTANCE:
                signals = _pair_signals(left_feature, right_feature, phash_distance, similarity)
                candidates.append((signals["priority"], left.chunk_id, right.chunk_id, signals))
        for similarity, right in sorted(vector_neighbours, key=lambda item: item[0], reverse=True)[:MAX_VECTOR_NEIGHBOURS]:
            if similarity < VECTOR_CANDIDATE_SIMILARITY:
                continue
            left_feature, right_feature = features[left.chunk_id], features[right.chunk_id]
            distance = _hamming(left_feature["phash"], right_feature["phash"])
            signals = _pair_signals(left_feature, right_feature, distance, similarity)
            candidates.append((signals["priority"], left.chunk_id, right.chunk_id, signals))

    seen: set[tuple[str, str]] = set()
    for _, left_id, right_id, signals in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = tuple(sorted((left_id, right_id)))
        if key in seen:
            continue
        seen.add(key)
        yield left_id, right_id, signals


def _pair_signals(left: dict[str, Any], right: dict[str, Any], phash_distance: int, similarity: float | None):
    colour_conflict = _colour_conflict(left.get("colour"), right.get("colour"))
    priority = (1.0 - min(phash_distance, 64) / 64.0) + max(0.0, similarity or 0.0)
    return {
        "phash_distance": phash_distance,
        "vector_similarity": similarity,
        "colour_conflict": colour_conflict,
        "priority": priority,
    }


def _graph_components(nodes: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    result: list[list[str]] = []
    remaining = set(nodes)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: list[str] = []
        remaining.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current] & remaining):
                remaining.remove(neighbour)
                stack.append(neighbour)
        result.append(sorted(component))
    return result


def _bounded_review_batches(component: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    """Keep one VLM contact sheet bounded without joining through weak chains."""

    if len(component) <= MAX_REVIEW_ITEMS:
        return [component]
    adjacency = {node: set() for node in component}
    for left, right in edges:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(component)
    batches: list[list[str]] = []
    while remaining:
        seed = max(remaining, key=lambda node: (len(adjacency[node] & remaining), node))
        batch = [seed]
        remaining.remove(seed)
        frontier = sorted(adjacency[seed] & remaining, key=lambda node: len(adjacency[node]), reverse=True)
        for node in frontier:
            if len(batch) >= MAX_REVIEW_ITEMS:
                break
            # Complete-link compatibility prevents A~B~C chains from becoming
            # a single visual class merely through transitivity.
            if all(tuple(sorted((node, member))) in edges for member in batch):
                batch.append(node)
                remaining.remove(node)
        batches.append(batch)
    return batches


def _review_partition(unit_ids: list[str], representatives: dict[str, ManualChunk]) -> list[list[str]]:
    if len(unit_ids) <= 1:
        return [unit_ids]
    api_key = _clean_env(os.getenv("VLM_API_KEY"))
    if not api_key:
        return [[unit_id] for unit_id in unit_ids]
    rows = [representatives[unit_id] for unit_id in unit_ids]
    verdict = _vlm_partition(rows, api_key)
    if verdict is None:
        return [[unit_id] for unit_id in unit_ids]

    assigned: set[int] = set()
    result: list[list[str]] = []
    for group in verdict:
        indices = group.get("indices") if isinstance(group, dict) else None
        confidence = float(group.get("confidence") or 0.0) if isinstance(group, dict) else 0.0
        valid = []
        for value in indices or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(unit_ids) and index not in assigned:
                valid.append(index)
        if len(valid) >= 2 and confidence >= VLM_GROUP_CONFIDENCE:
            result.append([unit_ids[index - 1] for index in valid])
            assigned.update(valid)
    for index, unit_id in enumerate(unit_ids, start=1):
        if index not in assigned:
            result.append([unit_id])
    return result


def _vlm_partition(rows: list[ManualChunk], api_key: str) -> list[dict[str, Any]] | None:
    sheet_path = _contact_sheet(rows)
    if sheet_path is None:
        return None
    try:
        from openai import OpenAI

        items = [
            {
                "index": index,
                "manual_name": row.object_name,
                "section_title": row.section_title,
            }
            for index, row in enumerate(rows, start=1)
        ]
        prompt = {
            "task": (
                "把编号图片按同一个视觉图标身份分组。警告、状态、操作、按钮图标都可能出现；"
                "只判断视觉身份，不按名称后缀分类。不同符号、不同状态颜色默认分开。"
                "不能确定相同的图片不要合并。只输出 JSON。"
            ),
            "items": items,
            "output": {"groups": [{"indices": [1, 2], "confidence": 0.0}]},
        }
        client = OpenAI(
            api_key=api_key,
            base_url=_clean_env(os.getenv("VLM_BASE_URL")) or _clean_env(os.getenv("DASHSCOPE_BASE_URL")),
        )
        response = client.chat.completions.create(
            model=_clean_env(os.getenv("VLM_MODEL")) or "qwen-vl-plus",
            temperature=0,
            max_tokens=max(512, len(rows) * 48),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": _image_data_url(sheet_path)}},
                    ],
                }
            ],
        )
        parsed = _parse_json_object(str(response.choices[0].message.content or "{}"))
        groups = parsed.get("groups")
        return groups if isinstance(groups, list) else None
    except Exception as error:
        logger.warning("Icon group VLM review failed: size=%s error=%s", len(rows), error)
        return None
    finally:
        sheet_path.unlink(missing_ok=True)


def _contact_sheet(rows: list[ManualChunk]) -> Path | None:
    try:
        import cv2
        import numpy as np

        cell_width, cell_height = 220, 190
        columns = min(6, max(1, math.ceil(math.sqrt(len(rows)))))
        row_count = math.ceil(len(rows) / columns)
        canvas = np.full((row_count * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
        for index, row in enumerate(rows, start=1):
            image = cv2.imread(str(row.image_path))
            if image is None:
                continue
            available_width, available_height = cell_width - 20, cell_height - 42
            scale = min(available_width / image.shape[1], available_height / image.shape[0])
            width = max(1, int(image.shape[1] * scale))
            height = max(1, int(image.shape[0] * scale))
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            column = (index - 1) % columns
            line = (index - 1) // columns
            left = column * cell_width + (cell_width - width) // 2
            top = line * cell_height + 34 + (available_height - height) // 2
            canvas[top : top + height, left : left + width] = resized
            cv2.putText(
                canvas,
                str(index),
                (column * cell_width + 8, line * cell_height + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        handle = tempfile.NamedTemporaryFile(prefix="icon_group_", suffix=".jpg", delete=False)
        handle.close()
        path = Path(handle.name)
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            path.unlink(missing_ok=True)
            return None
        return path
    except Exception:
        return None


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _representative(members: list[ManualChunk]) -> ManualChunk:
    return max(
        members,
        key=lambda row: (
            bool(row.object_name),
            _image_area(Path(str(row.image_path or ""))),
            -row.page_number,
        ),
    )


def _perceptual_hash(path: Path) -> int | None:
    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))[:8, :8]
        median = float(np.median(dct[1:, :]))
        bits = (dct > median).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bool(bit))
        return value
    except Exception:
        return None


def _dominant_colour(path: Path) -> tuple[float, float] | None:
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturated = hsv[:, :, 1] >= 55
        if int(saturated.sum()) < 20:
            return None
        hue = float(hsv[:, :, 0][saturated].mean())
        saturation = float(hsv[:, :, 1][saturated].mean())
        return hue, saturation
    except Exception:
        return None


def _colour_conflict(left: tuple[float, float] | None, right: tuple[float, float] | None) -> bool:
    if not left or not right or min(left[1], right[1]) < 70:
        return False
    distance = abs(left[0] - right[0])
    distance = min(distance, 180.0 - distance)
    return distance >= 24.0


def _hamming(left: int | None, right: int | None) -> int:
    if left is None or right is None:
        return 64
    return (left ^ right).bit_count()


def _cosine(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(value) ** 2 for value in left))
    norm_right = math.sqrt(sum(float(value) ** 2 for value in right))
    if not norm_left or not norm_right:
        return None
    return dot / (norm_left * norm_right)


def _find(parent: dict[str, str], value: str) -> str:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root, right_root = _find(parent, left), _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _canonical_name(members: list[ManualChunk]) -> str | None:
    values = [row.object_name for row in members if row.object_name]
    return Counter(values).most_common(1)[0][0] if values else None


def _stable_group_id(manual_id: str, members: list[ManualChunk]) -> str:
    raw = "|".join(sorted(row.chunk_id for row in members))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    safe_manual = re.sub(r"[^A-Za-z0-9_-]+", "-", manual_id).strip("-")[:48] or "manual"
    return f"{safe_manual}_icon_group_{digest}"


def _image_area(path: Path) -> int:
    try:
        import cv2

        image = cv2.imread(str(path))
        return int(image.shape[0] * image.shape[1]) if image is not None else 0
    except Exception:
        return 0


def _image_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _clean_env(value: str | None) -> str | None:
    cleaned = str(value or "").strip().strip('"').strip("'")
    return cleaned or None
