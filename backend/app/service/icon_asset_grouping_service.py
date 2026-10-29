"""Group native icon assets, choose strict manual names and visual prototypes."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from service.native_pdf_image_service import (
    _border_difference_mask,
    _crop_to_foreground,
    _pad_square,
    oriented_image_path,
)


logger = logging.getLogger(__name__)

EXACT_PHASH_DISTANCE = max(0, int(os.getenv("ICON_GROUP_EXACT_PHASH_DISTANCE", "3")))
PHASH_CANDIDATE_DISTANCE = max(
    EXACT_PHASH_DISTANCE,
    int(os.getenv("ICON_GROUP_PHASH_DISTANCE", "10")),
)
VECTOR_CANDIDATE_SIMILARITY = float(
    os.getenv("ICON_GROUP_VECTOR_SIMILARITY", "0.88")
)
VLM_GROUP_CONFIDENCE = float(os.getenv("ICON_GROUP_VLM_CONFIDENCE", "0.82"))
MAX_REVIEW_ITEMS = max(
    4,
    min(24, int(os.getenv("ICON_GROUP_VLM_MAX_ITEMS", "12"))),
)


def build_native_icon_groups(
    *,
    manual_id: str,
    assets: list[dict[str, Any]],
    occurrences: list[dict[str, Any]],
    image_vectors: dict[str, list[float] | None],
    variant_dir: Path,
) -> list[dict[str, Any]]:
    """Return final groups without writing occurrence-shaped icon chunks."""

    asset_by_id = {
        str(asset["asset_id"]): asset
        for asset in assets
        if asset.get("classification") == "icon"
    }
    occurrences_by_unit: dict[str, list[dict[str, Any]]] = {}
    for occurrence in occurrences:
        asset_id = str(occurrence["asset_id"])
        if asset_id not in asset_by_id:
            continue
        orientation = str(occurrence.get("orientation") or "r0")
        unit_id = f"{asset_id}:{orientation}"
        occurrences_by_unit.setdefault(unit_id, []).append(occurrence)

    units: dict[str, dict[str, Any]] = {}
    for unit_id, unit_occurrences in occurrences_by_unit.items():
        asset_id, orientation = unit_id.rsplit(":", 1)
        asset = asset_by_id[asset_id]
        path = oriented_image_path(asset, orientation, variant_dir)
        units[unit_id] = {
            "unit_id": unit_id,
            "asset_id": asset_id,
            "asset": asset,
            "orientation": orientation,
            "image_path": path,
            "occurrences": sorted(
                unit_occurrences,
                key=lambda item: (item["page_number"], item["bbox"][1], item["bbox"][0]),
            ),
            "phash": (
                asset.get("phash")
                if orientation == "r0"
                else _perceptual_hash(Path(str(path)))
            )
            if path
            else None,
            "colour": (
                asset.get("dominant_colour")
                if orientation == "r0"
                else _dominant_colour(Path(str(path)))
            )
            if path
            else None,
            "vector": image_vectors.get(asset_id),
            "quality_score": float(asset.get("quality_score") or 0.0),
        }
    if not units:
        return []

    candidate_edges = set(_candidate_edges(list(units.values())))
    final_unit_groups: list[list[str]] = []
    for component in _graph_components(list(units), candidate_edges):
        if len(component) == 1:
            final_unit_groups.append(component)
            continue
        final_unit_groups.extend(
            _review_component(component, units, candidate_edges)
        )

    groups: list[dict[str, Any]] = []
    for unit_ids in final_unit_groups:
        members = [units[unit_id] for unit_id in sorted(unit_ids)]
        member_occurrences = [
            occurrence
            for member in members
            for occurrence in member["occurrences"]
        ]
        member_occurrences.sort(
            key=lambda item: (item["page_number"], item["bbox"][1], item["bbox"][0])
        )
        group_id = _stable_group_id(manual_id, unit_ids)
        canonical_name = _canonical_name(group_id, members)
        prototypes = _select_prototypes(group_id, members)
        representative = _representative_occurrence(member_occurrences, prototypes)
        groups.append(
            {
                "icon_group_id": group_id,
                "manual_id": manual_id,
                "canonical_icon_name": canonical_name,
                "member_count": len(member_occurrences),
                "members": members,
                "occurrences": member_occurrences,
                "prototypes": prototypes,
                "representative_occurrence": representative,
            }
        )
    groups.sort(key=lambda item: item["icon_group_id"])
    return groups


def _candidate_edges(units: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Colour rejects first; pHash/vector only propose pairs for VLM review."""

    edges: list[tuple[str, str]] = []
    for index, left in enumerate(units):
        for right in units[index + 1 :]:
            # Same exact PDF asset with the same direction would already be one
            # unit. Different directions must still be reviewed, never merged
            # automatically.
            if _colour_conflict(left.get("colour"), right.get("colour")):
                continue
            phash_distance = _hamming(left.get("phash"), right.get("phash"))
            vector_similarity = _cosine(left.get("vector"), right.get("vector"))
            if phash_distance <= EXACT_PHASH_DISTANCE:
                edges.append(tuple(sorted((left["unit_id"], right["unit_id"]))))
            elif (
                phash_distance <= PHASH_CANDIDATE_DISTANCE
                or (vector_similarity is not None and vector_similarity >= VECTOR_CANDIDATE_SIMILARITY)
            ):
                edges.append(tuple(sorted((left["unit_id"], right["unit_id"]))))
    return list(dict.fromkeys(edges))


def _review_component(
    component: list[str],
    units: dict[str, dict[str, Any]],
    edges: set[tuple[str, str]],
) -> list[list[str]]:
    """Split a candidate component with the reviewed partitions only.

    The partition returned by the VLM is the final grouping decision; there is
    no second pairwise confirmation pass.  Overlapping review batches may
    disagree, so a pair is merged only when at least one batch put the two
    units in the same accepted partition and no batch separated them.
    """

    api_key = _clean_env(os.getenv("VLM_API_KEY"))
    if not api_key:
        raise RuntimeError(
            "Cross-fingerprint icon candidates require VLM group review; "
            "they cannot be finalized from pHash/vector thresholds alone."
        )

    pair_support: dict[tuple[str, str], int] = {}
    pair_conflict: set[tuple[str, str]] = set()
    for batch in _edge_cover_batches(component, edges):
        if len(batch) < 2:
            continue
        verdict = _vlm_partition([units[unit_id] for unit_id in batch], api_key)
        if verdict is None:
            raise RuntimeError(
                "VLM icon-group review failed for candidate units: "
                + ", ".join(batch)
            )
        groups, _ = _accepted_partitions(verdict, batch)
        for group in groups:
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    pair = tuple(sorted((left, right)))
                    pair_support[pair] = pair_support.get(pair, 0) + 1
        for left_index, left in enumerate(batch):
            for right in batch[left_index + 1 :]:
                pair = tuple(sorted((left, right)))
                if pair in pair_conflict:
                    continue
                same_group = any(left in group and right in group for group in groups)
                # An unassigned unit is conservatively its own group, so any
                # pair the review did not place together is a conflict.
                if not same_group:
                    pair_conflict.add(pair)

    parent = {unit_id: unit_id for unit_id in component}
    for pair, support in pair_support.items():
        if support > 0 and pair not in pair_conflict:
            _union(parent, pair[0], pair[1])

    result_by_root: dict[str, list[str]] = {}
    for unit_id in component:
        result_by_root.setdefault(_find(parent, unit_id), []).append(unit_id)
    return [sorted(values) for values in result_by_root.values()]


def _accepted_partitions(
    verdict: list[dict[str, Any]],
    batch: list[str],
) -> tuple[list[list[str]], set[str]]:
    """Turn one VLM verdict into valid unit groups.

    Every batch item is assigned at most once.  Items that are missing from the
    verdict, repeated across groups, below the confidence floor or grouped
    alone stay unassigned and therefore become conservative singleton groups.
    """

    index_groups: list[list[int]] = []
    for raw_group in verdict:
        if not isinstance(raw_group, dict):
            continue
        try:
            confidence = float(raw_group.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < VLM_GROUP_CONFIDENCE:
            continue
        indices: list[int] = []
        for value in raw_group.get("indices") or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(batch) and index not in indices:
                indices.append(index)
        if len(indices) >= 2:
            index_groups.append(indices)

    membership: dict[int, int] = {}
    ambiguous: set[int] = set()
    for group_index, indices in enumerate(index_groups):
        for index in indices:
            if index in membership and membership[index] != group_index:
                ambiguous.add(index)
            else:
                membership[index] = group_index

    groups: list[list[str]] = []
    for group_index, indices in enumerate(index_groups):
        members = [
            batch[index - 1]
            for index in indices
            if index not in ambiguous and membership.get(index) == group_index
        ]
        if len(members) >= 2:
            groups.append(members)
    assigned = {batch[index - 1] for index in membership if index not in ambiguous}
    return groups, assigned


def _edge_cover_batches(
    component: list[str],
    edges: set[tuple[str, str]],
) -> list[list[str]]:
    """Create bounded overlapping sheets so every candidate edge is reviewed."""

    component_set = set(component)
    pending = {
        edge
        for edge in edges
        if edge[0] in component_set and edge[1] in component_set
    }
    if len(component) <= MAX_REVIEW_ITEMS:
        return [sorted(component)]
    batches: list[list[str]] = []
    while pending:
        seed = min(pending)
        batch: list[str] = [seed[0], seed[1]]
        while len(batch) < MAX_REVIEW_ITEMS:
            neighbours = sorted(
                {
                    node
                    for left, right in pending
                    for node in (left, right)
                    if (left in batch or right in batch) and node not in batch
                }
            )
            if not neighbours:
                break
            batch.append(neighbours[0])
        batch_set = set(batch)
        covered = {
            edge
            for edge in pending
            if edge[0] in batch_set and edge[1] in batch_set
        }
        pending -= covered
        batches.append(batch)
    reviewed = {node for batch in batches for node in batch}
    batches.extend([[node] for node in sorted(component_set - reviewed)])
    return batches


def _vlm_partition(
    units: list[dict[str, Any]],
    api_key: str,
) -> list[dict[str, Any]] | None:
    sheet = _contact_sheet(units)
    if sheet is None:
        return None
    try:
        from openai import OpenAI

        prompt = {
            "task": (
                "审核候选预分组：把编号图片中视觉身份确实相同的icon分在一起。"
                "icon可以是警告、状态、操作、按钮等任何符号。只判断同一图标身份，"
                "不要按主题或文字相近合并；不同符号、不同状态颜色默认分开。"
                "PDF可能把同一图标以不同分辨率、尺寸、抗锯齿、压缩和轻微横纵拉伸"
                "重复保存；这些栅格化差异不能当作不同图标。比较外圈、内部符号、"
                "部件拓扑和颜色语义。不同出现位置独立提取出的相同candidate_name"
                "可以作为辅助证据，但名称本身不能覆盖明显的视觉冲突。"
                "可以拆开全部候选。只输出JSON。"
            ),
            "items": [
                {
                    "index": index,
                    "candidate_names": _unit_candidate_names(unit),
                    "pages": sorted(
                        {
                            int(item["page_number"])
                            for item in unit.get("occurrences") or []
                        }
                    )[:12],
                }
                for index, unit in enumerate(units, start=1)
            ],
            "output": {
                "groups": [
                    {
                        "indices": [1, 2],
                        "confidence": 0.0,
                    }
                ]
            },
        }
        client = OpenAI(
            api_key=api_key,
            base_url=_clean_env(os.getenv("VLM_BASE_URL"))
            or _clean_env(os.getenv("DASHSCOPE_BASE_URL")),
        )
        response = client.chat.completions.create(
            model=_clean_env(os.getenv("VLM_MODEL")) or "qwen-vl-plus",
            temperature=0,
            max_tokens=max(512, len(units) * 56),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": _image_data_url(sheet)}},
                    ],
                }
            ],
        )
        parsed = _parse_json_object(str(response.choices[0].message.content or "{}"))
        groups = parsed.get("groups")
        return groups if isinstance(groups, list) else None
    except Exception as error:
        logger.warning("Native icon group VLM review failed: size=%s error=%s", len(units), error)
        return None
    finally:
        sheet.unlink(missing_ok=True)


def _canonical_name(group_id: str, units: list[dict[str, Any]]) -> str | None:
    candidate_rows: list[dict[str, Any]] = []
    for unit in units:
        for occurrence in unit.get("occurrences") or []:
            candidates = _occurrence_name_candidates(occurrence)
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "candidate": candidate,
                        "page": occurrence.get("page_number"),
                        "section_title": occurrence.get("section_title"),
                        "priority": candidates.index(candidate),
                    }
                )
    if not candidate_rows:
        return None

    # "Consistent" means every occurrence's first spatial candidate agrees.
    first_by_occurrence: list[str] = []
    for unit in units:
        for occurrence in unit.get("occurrences") or []:
            first = next(iter(_occurrence_name_candidates(occurrence)), None)
            if first:
                first_by_occurrence.append(first)
    normalized_first = {_normalize_name(value): value for value in first_by_occurrence}
    if len(normalized_first) == 1:
        return next(iter(normalized_first.values()))

    api_key = _clean_env(os.getenv("VLM_API_KEY"))
    if not api_key:
        return None
    selected = _vlm_choose_candidate_name(group_id, units, candidate_rows, api_key)
    if selected is None:
        raise RuntimeError(
            f"VLM did not select a canonical manual-name candidate for icon group {group_id}"
        )
    return selected


def _vlm_choose_candidate_name(
    group_id: str,
    units: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    api_key: str,
) -> str | None:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        key = _normalize_name(str(row["candidate"]))
        if not key:
            continue
        entry = aggregated.setdefault(
            key,
            {
                "candidate": row["candidate"],
                "first_candidate_count": 0,
                "total_count": 0,
                "best_spatial_priority": int(row.get("priority") or 0),
                "earliest_page": int(row.get("page") or 10**9),
                "contexts": [],
            },
        )
        priority = int(row.get("priority") or 0)
        entry["total_count"] += 1
        entry["first_candidate_count"] += int(priority == 0)
        entry["best_spatial_priority"] = min(
            int(entry["best_spatial_priority"]),
            priority,
        )
        entry["earliest_page"] = min(
            int(entry["earliest_page"]),
            int(row.get("page") or 10**9),
        )
        context = {
            "page": row.get("page"),
            "section_title": row.get("section_title"),
            "spatial_priority": priority,
        }
        if context not in entry["contexts"] and len(entry["contexts"]) < 6:
            entry["contexts"].append(context)
        # Prefer the literal spelling that most recently appeared as a
        # first-position candidate; this preserves manual suffixes such as "*".
        if priority == 0:
            entry["candidate"] = row["candidate"]
    unique = sorted(
        aggregated.values(),
        key=lambda item: (
            int(item["earliest_page"]),
            int(item["best_spatial_priority"]),
            -int(item["first_candidate_count"]),
            -int(item["total_count"]),
            len(str(item["candidate"])),
        ),
    )[:36]
    for index, item in enumerate(unique, start=1):
        item["candidate_id"] = f"N{index}"
    sheet = _contact_sheet(units)
    if sheet is None:
        return None
    try:
        from openai import OpenAI

        prompt = {
            "task": (
                "为这一组相同icon选择手册中的标准名称。只能返回下列candidate_id之一，"
                "不能改写、拼接或创造名称。优先选择与图标同一表格单元格/同一行、"
                "能明确给该符号命名的候选；first_candidate_count表示该名称在多少个"
                "出现位置排空间第一。earliest_page表示该名称第一次出现的手册页；"
                "首次明确命名、空间第一的候选优先于后续章节中的动作或状态叙述。"
                "章节主题不是图标名称。"
                "同一图标在名称表中会出现一次、在后续正文中又出现多次；后续正文附近"
                "可能只有“点亮”“请检查”等叙述，不应因此否定名称表中的简短名词性标签。"
                "这不是按‘警告灯/指示灯’后缀硬编码，操作和状态图标同样按手册原名选择。"
                "只要候选列表非空就必须选出最符合图片和位置关系的candidate_id，不得"
                "返回null；即使部分出现位置只有叙述，也应选择名称表或首次明确命名位置"
                "中的原文候选。"
            ),
            "group_id": group_id,
            "candidates": unique,
            "output": {"candidate_id": "N1", "confidence": 0.0},
        }
        client = OpenAI(
            api_key=api_key,
            base_url=_clean_env(os.getenv("VLM_BASE_URL"))
            or _clean_env(os.getenv("DASHSCOPE_BASE_URL")),
        )
        for attempt in range(1, 3):
            prompt["attempt"] = attempt
            response = client.chat.completions.create(
                model=_clean_env(os.getenv("VLM_MODEL")) or "qwen-vl-plus",
                temperature=0,
                max_tokens=160,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                            {"type": "image_url", "image_url": {"url": _image_data_url(sheet)}},
                        ],
                    }
                ],
            )
            raw_response = str(response.choices[0].message.content or "{}")
            parsed = _parse_json_object(raw_response)
            nested = parsed.get("output")
            nested = nested if isinstance(nested, dict) else {}
            selected_id = str(
                parsed.get("candidate_id")
                or nested.get("candidate_id")
                or ""
            ).strip().upper()
            plain_response = raw_response.strip().strip("`").strip()
            if not selected_id and re.fullmatch(r"N\d+", plain_response, re.IGNORECASE):
                selected_id = plain_response.upper()
            if not selected_id:
                mentioned_ids = {
                    value.upper()
                    for value in re.findall(r"\bN\d+\b", raw_response, re.IGNORECASE)
                }
                if len(mentioned_ids) == 1:
                    selected_id = next(iter(mentioned_ids))
            selected = next(
                (row for row in unique if row["candidate_id"] == selected_id),
                None,
            )
            if selected is None:
                selected_name = parsed.get("object_name") or parsed.get("name")
                if not selected_name and plain_response:
                    selected_name = plain_response
                if selected_name:
                    selected = next(
                        (
                            row
                            for row in unique
                            if _normalize_name(str(row["candidate"]))
                            == _normalize_name(str(selected_name))
                        ),
                        None,
                    )
            if selected is None:
                mentioned_names = [
                    row
                    for row in unique
                    if str(row["candidate"]) in raw_response
                ]
                if len(mentioned_names) == 1:
                    selected = mentioned_names[0]
            if selected is not None:
                return str(selected["candidate"])
            logger.warning(
                "Canonical icon name VLM returned no valid candidate: "
                "group=%s attempt=%s selected_id=%r response=%r",
                group_id,
                attempt,
                selected_id,
                raw_response[:1000],
            )
        return None
    except Exception as error:
        logger.warning("Canonical icon name selection failed: group=%s error=%s", group_id, error)
        return None
    finally:
        sheet.unlink(missing_ok=True)


def _select_prototypes(
    group_id: str,
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the single representative image that online retrieval indexes.

    Kept as a list so the persisted prototype row and existing callers keep
    their shape, but only the top-ranked unit is returned: one group produces
    one icon chunk and one Elasticsearch document.
    """

    scored: list[tuple[float, dict[str, Any]]] = []
    for unit in units:
        similarities = [
            _unit_similarity(unit, other)
            for other in units
            if other["unit_id"] != unit["unit_id"]
        ]
        centrality = sum(similarities) / len(similarities) if similarities else 1.0
        score = 0.65 * centrality + 0.35 * float(unit.get("quality_score") or 0.0)
        scored.append((score, unit))
    scored.sort(key=lambda item: (item[0], item[1]["unit_id"]), reverse=True)

    score, unit = scored[0]
    if not unit.get("image_path"):
        return []
    return [
        {
            "prototype_id": f"{group_id}_prototype_1",
            "asset_id": unit["asset_id"],
            "orientation": unit["orientation"],
            "image_path": unit["image_path"],
            "phash": unit.get("phash"),
            "dominant_colour": unit.get("colour"),
            "quality_score": round(score, 6),
            "image_vector": unit.get("vector"),
            "prototype_rank": 1,
        }
    ]


def _representative_occurrence(
    occurrences: list[dict[str, Any]],
    prototypes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not occurrences:
        return None
    prototype_asset = str(prototypes[0]["asset_id"]) if prototypes else ""
    return max(
        occurrences,
        key=lambda item: (
            str(item.get("asset_id")) == prototype_asset,
            bool(item.get("name_candidates")),
            _bbox_area(item.get("bbox")),
            -int(item.get("page_number") or 0),
        ),
    )


def _unit_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    vector_similarity = _cosine(left.get("vector"), right.get("vector"))
    if vector_similarity is not None:
        return max(0.0, min(1.0, vector_similarity))
    return max(0.0, 1.0 - _hamming(left.get("phash"), right.get("phash")) / 64.0)


def _unit_candidate_names(unit: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for occurrence in unit.get("occurrences") or []:
        for name in _occurrence_name_candidates(occurrence):
            if name not in result:
                result.append(name)
    return result[:20]


def _contact_sheet(units: list[dict[str, Any]]) -> Path | None:
    try:
        import cv2
        import numpy as np

        cell_width, cell_height = 220, 190
        columns = min(6, max(1, math.ceil(math.sqrt(len(units)))))
        row_count = math.ceil(len(units) / columns)
        canvas = np.full(
            (row_count * cell_height, columns * cell_width, 3),
            255,
            dtype=np.uint8,
        )
        for index, unit in enumerate(units, start=1):
            image = cv2.imread(str(unit.get("image_path") or ""), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            if image.ndim == 3 and image.shape[2] == 4:
                alpha = image[:, :, 3:4].astype("float32") / 255.0
                image = (
                    image[:, :, :3].astype("float32") * alpha
                    + 255.0 * (1.0 - alpha)
                ).astype("uint8")
            available_width, available_height = cell_width - 20, cell_height - 42
            scale = min(
                available_width / max(1, image.shape[1]),
                available_height / max(1, image.shape[0]),
            )
            width = max(1, int(image.shape[1] * scale))
            height = max(1, int(image.shape[0] * scale))
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            column = (index - 1) % columns
            line = (index - 1) // columns
            left = column * cell_width + (cell_width - width) // 2
            top = line * cell_height + 34 + (available_height - height) // 2
            canvas[top : top + height, left : left + width] = resized[:, :, :3]
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
        handle = tempfile.NamedTemporaryFile(
            prefix="native_icon_group_",
            suffix=".jpg",
            delete=False,
        )
        handle.close()
        path = Path(handle.name)
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            path.unlink(missing_ok=True)
            return None
        return path
    except Exception:
        return None


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
            image = (
                image[:, :, :3].astype("float32") * alpha
                + 255.0 * (1.0 - alpha)
            ).astype("uint8")
        elif image.ndim == 3:
            foreground = _border_difference_mask(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = _crop_to_foreground(gray, foreground)
        gray = _pad_square(gray, 255)
        dct = cv2.dct(
            cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
        )[:8, :8]
        median = float(np.median(dct.reshape(-1)[1:]))
        bits = "".join("1" if float(value) > median else "0" for value in dct.reshape(-1))
        return f"{int(bits, 2):016x}"
    except Exception:
        return None


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
        angles = hue_values * (2.0 * math.pi / 180.0)
        hue = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
        hue = (hue % (2.0 * math.pi)) * 180.0 / (2.0 * math.pi)
        return {
            "hue": round(hue, 3),
            "saturation": round(float(hsv[:, :, 1][mask].mean()), 3),
        }
    except Exception:
        return None


def _colour_conflict(
    left: dict[str, float] | None,
    right: dict[str, float] | None,
) -> bool:
    if not left or not right:
        return False
    left_saturation = float(left.get("saturation") or 0.0)
    right_saturation = float(right.get("saturation") or 0.0)
    if min(left_saturation, right_saturation) < 70:
        return False
    distance = abs(float(left.get("hue") or 0.0) - float(right.get("hue") or 0.0))
    return min(distance, 180.0 - distance) >= 24.0


def _hamming(left: str | None, right: str | None) -> int:
    try:
        return (int(str(left), 16) ^ int(str(right), 16)).bit_count()
    except (TypeError, ValueError):
        return 64


def _cosine(
    left: list[float] | None,
    right: list[float] | None,
) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left_norm or not right_norm:
        return None
    return dot / (left_norm * right_norm)


def _graph_components(
    nodes: list[str],
    edges: set[tuple[str, str]],
) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(nodes)
    result: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current] & remaining):
                remaining.remove(neighbour)
                stack.append(neighbour)
        result.append(sorted(component))
    return result


def _find(parent: dict[str, str], value: str) -> str:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _stable_group_id(manual_id: str, unit_ids: list[str]) -> str:
    raw = "|".join(sorted(unit_ids))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    safe_manual = re.sub(r"[^A-Za-z0-9_-]+", "-", manual_id).strip("-")[:48] or "manual"
    return f"{safe_manual}_icon_group_{digest}"


def _clean_name(value: Any) -> str | None:
    text = " ".join(str(value or "").split()).strip(" ：:，,。")
    compact = "".join(text.split())
    return text if 2 <= len(compact) <= 64 else None


def _occurrence_name_candidates(
    occurrence: dict[str, Any],
) -> list[str]:
    """Keep the two closest literal labels, excluding bookmark/header text."""

    section_parts = {
        _normalize_name(part)
        for part in str(occurrence.get("section_title") or "").split(" > ")
        if part.strip()
    }
    result: list[str] = []
    for value in occurrence.get("name_candidates") or []:
        name = _clean_name(value)
        if not name:
            continue
        normalized = _normalize_name(name)
        if normalized in section_parts or normalized.isdigit():
            continue
        if name not in result:
            result.append(name)
        if len(result) >= 2:
            break
    return result


def _normalize_name(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _bbox_area(value: Any) -> float:
    if not isinstance(value, list) or len(value) != 4:
        return 0.0
    return max(0.0, float(value[2]) - float(value[0])) * max(
        0.0,
        float(value[3]) - float(value[1]),
    )


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


def _image_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _clean_env(value: str | None) -> str | None:
    cleaned = str(value or "").strip().strip('"').strip("'")
    return cleaned or None
