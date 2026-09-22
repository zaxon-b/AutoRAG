"""VLM classification for exact-deduplicated native PDF image assets."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from service.native_pdf_image_service import oriented_image_path
from service.visual_object_naming_service import _validate_operation_name


logger = logging.getLogger(__name__)

CLASSIFICATION_SCHEMA_VERSION = "native_visual_v2_icon_semantics"
CLASSIFICATION_BATCH_SIZE = max(
    1,
    min(8, int(os.getenv("VISUAL_ASSET_CLASSIFICATION_BATCH_SIZE", "4"))),
)
CLASSIFICATION_WORKERS = max(
    1,
    min(6, int(os.getenv("VISUAL_ASSET_CLASSIFICATION_WORKERS", "3"))),
)
INDIVIDUAL_ICON_CANDIDATE_REVIEW_MARKER = ":individual_icon_candidate_review:"


def classify_visual_assets(
    assets: list[dict[str, Any]],
    occurrences: list[dict[str, Any]],
    *,
    variant_dir: Path,
) -> None:
    """Mutate assets with icon/operation_figure/ignore decisions.

    Deterministic noise rows already carry ``classification=ignore`` and never
    consume a VLM call. Every remaining unique asset is classified once.
    """

    occurrences_by_asset: dict[str, list[dict[str, Any]]] = {}
    for occurrence in occurrences:
        occurrences_by_asset.setdefault(str(occurrence["asset_id"]), []).append(occurrence)

    targets: list[dict[str, Any]] = []
    target_by_asset_id: dict[str, dict[str, Any]] = {}
    for asset in assets:
        representative = _representative_occurrence(
            occurrences_by_asset.get(str(asset["asset_id"]), [])
        )
        if (
            asset.get("_classification_cached")
            and asset.get("classification") in {"icon", "operation_figure", "ignore"}
        ):
            if (
                asset.get("classification") != "icon"
                and not asset.get("noise_reason")
                and INDIVIDUAL_ICON_CANDIDATE_REVIEW_MARKER not in str(
                    asset.get("classification_reason") or ""
                )
                and representative is not None
                and asset.get("native_image_path")
            ):
                target_by_asset_id[str(asset["asset_id"])] = {
                    "asset": asset,
                    "occurrence": representative,
                    "display_path": oriented_image_path(
                        asset,
                        str(representative.get("orientation") or "r0"),
                        variant_dir,
                    ),
                }
            continue
        if asset.get("classification") == "ignore" and asset.get("noise_reason"):
            continue
        if representative is None or not asset.get("native_image_path"):
            asset.update(
                {
                    "classification": "ignore",
                    "classification_confidence": 1.0,
                    "classification_reason": "asset_has_no_occurrence_or_image",
                }
            )
            continue
        display_path = oriented_image_path(
            asset,
            str(representative.get("orientation") or "r0"),
            variant_dir,
        )
        target = {
            "asset": asset,
            "occurrence": representative,
            "display_path": display_path,
        }
        targets.append(target)
        target_by_asset_id[str(asset["asset_id"])] = target

    if not targets and not target_by_asset_id:
        return
    api_key = _clean_env(os.getenv("VLM_API_KEY"))
    if not api_key or not _enabled():
        raise RuntimeError(
            "Native visual asset classification requires an enabled VLM; "
            "size thresholds are not allowed to make the final icon/figure decision."
        )

    results: dict[str, dict[str, Any]] = {}
    if targets:
        batches = [
            targets[start : start + CLASSIFICATION_BATCH_SIZE]
            for start in range(0, len(targets), CLASSIFICATION_BATCH_SIZE)
        ]
        with ThreadPoolExecutor(
            max_workers=min(CLASSIFICATION_WORKERS, len(batches))
        ) as executor:
            futures = {
                executor.submit(_classify_batch, batch, api_key): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    results.update(future.result())
                except Exception as error:
                    logger.warning(
                        "Native visual asset classification failed: assets=%s error=%s",
                        [target["asset"]["asset_id"] for target in batch],
                        error,
                    )

    unresolved: list[str] = []
    for target in targets:
        asset = target["asset"]
        verdict = results.get(str(asset["asset_id"]))
        if verdict is None:
            unresolved.append(str(asset["asset_id"]))
            continue
        asset["classification"] = verdict["classification"]
        asset["classification_confidence"] = verdict["confidence"]
        reason = str(verdict.get("reason") or "").strip()
        asset["classification_reason"] = (
            f"{CLASSIFICATION_SCHEMA_VERSION}:{reason}"
        )
        asset["operation_object_name"] = verdict.get("object_name")
    if unresolved:
        raise RuntimeError(
            "VLM did not return a valid icon/operation_figure/ignore decision for "
            f"{len(unresolved)} native assets: {', '.join(unresolved[:8])}"
        )

    # Batch VLM calls can occasionally associate one item's image with a
    # neighbouring item's metadata. Size only selects suspicious non-icons for
    # a second look; the one-item VLM call still makes the final decision.
    # Re-review both ``ignore`` and ``operation_figure``: a table icon wrongly
    # labelled as a figure would otherwise silently bypass icon grouping.
    for target in target_by_asset_id.values():
        asset = target["asset"]
        occurrence = target["occurrence"]
        if (
            asset.get("classification") == "icon"
            or asset.get("noise_reason")
            or not _is_icon_sized_review_candidate(occurrence)
        ):
            continue
        verdict = _classify_batch([target], api_key).get(str(asset["asset_id"]))
        if verdict is None:
            raise RuntimeError(
                "Individual VLM re-review returned no valid decision for "
                f"{asset['asset_id']}"
            )
        asset["classification"] = verdict["classification"]
        asset["classification_confidence"] = verdict["confidence"]
        reason = str(verdict.get("reason") or "").strip()
        asset["classification_reason"] = (
            f"{CLASSIFICATION_SCHEMA_VERSION}"
            f"{INDIVIDUAL_ICON_CANDIDATE_REVIEW_MARKER}{reason}"
        )
        asset["operation_object_name"] = verdict.get("object_name")


def _classify_batch(
    batch: list[dict[str, Any]],
    api_key: str,
) -> dict[str, dict[str, Any]]:
    from openai import OpenAI

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "task": "逐项判断车主手册的原生图片资产类型，只输出JSON。",
                    "types": {
                        "icon": (
                            "独立的符号、按钮、状态标识、模式标识、操作标识、仪表指示标识"
                            "或警示标识。icon不等于警告灯，不能根据名称后缀硬编码。"
                            "静态符号、没有交互的状态灯、内部带字母/文字的标识仍然是icon。"
                        ),
                        "operation_figure": (
                            "有完整语义、适合被用户截图查询的照片、操作图、结构图、"
                            "部件图、界面图或原理示意图；无需继续细分类。"
                        ),
                        "ignore": (
                            "页眉页脚、Logo、纯装饰、背景、蒙版、线条碎片、"
                            "无法独立表达手册对象的局部素材。"
                        ),
                    },
                    "rules": [
                        "尺寸和页面占比只能作为提示，不能单独决定类型。",
                        "必须结合原生图片、书签路径和附近手册文字判断。",
                        "如果附近文字或表格单元格在给该独立符号命名，应判为icon；"
                        "不能因为它静态、尺寸小、没有交互或单独看语义不完整而判为ignore。",
                        "HEV、EV、ESC OFF、雪花、旗帜、圆圈字母等模式/状态标识均可属于icon。",
                        "icon的object_name必须为null，本步骤不命名icon。",
                        "operation_figure可给出简短object_name，但必须能从所给书签或附近原文得到；不确定则为null。",
                        "不要把有完整语义的操作/结构图片判成icon。",
                        "ignore仅用于没有独立对象身份、没有对应名称的页眉页脚、Logo、"
                        "纯装饰、背景、蒙版、线条或图片碎片。",
                    ],
                    "output": {
                        "items": [
                            {
                                "index": 1,
                                "classification": "icon|operation_figure|ignore",
                                "confidence": 0.0,
                                "object_name": "string|null",
                                "reason": "short string",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            ),
        }
    ]
    prepared: list[dict[str, Any]] = []
    for index, target in enumerate(batch, start=1):
        occurrence = target["occurrence"]
        metadata = {
            "index": index,
            "asset_id": target["asset"]["asset_id"],
            "section_title": occurrence.get("section_title"),
            "nearby_text": occurrence.get("nearby_text"),
            "name_candidates": occurrence.get("name_candidates"),
            "display_bbox": occurrence.get("bbox"),
            "page_size": [occurrence.get("page_width"), occurrence.get("page_height")],
        }
        prepared.append(metadata)
        content.extend(
            [
                {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(Path(str(target["display_path"])))},
                },
            ]
        )

    client = OpenAI(
        api_key=api_key,
        base_url=_clean_env(os.getenv("VLM_BASE_URL"))
        or _clean_env(os.getenv("DASHSCOPE_BASE_URL")),
    )
    response = client.chat.completions.create(
        model=_clean_env(os.getenv("VLM_MODEL")) or "qwen-vl-plus",
        temperature=0,
        max_tokens=max(640, len(batch) * 140),
        messages=[{"role": "user", "content": content}],
    )
    parsed = _parse_json_object(str(response.choices[0].message.content or "{}"))
    raw_by_index = {
        int(item.get("index")): item
        for item in parsed.get("items") or []
        if isinstance(item, dict) and str(item.get("index") or "").isdigit()
    }

    result: dict[str, dict[str, Any]] = {}
    for index, target in enumerate(batch, start=1):
        raw = raw_by_index.get(index, {})
        classification = str(raw.get("classification") or "").strip()
        if classification not in {"icon", "operation_figure", "ignore"}:
            continue
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        occurrence = target["occurrence"]
        source = "\n".join(
            [
                str(occurrence.get("section_title") or ""),
                *[str(value) for value in occurrence.get("nearby_text") or []],
            ]
        )
        object_name = None
        if classification == "operation_figure":
            object_name = _validate_operation_name(raw.get("object_name"), source)
        result[str(target["asset"]["asset_id"])] = {
            "classification": classification,
            "confidence": confidence,
            "object_name": object_name,
            "reason": " ".join(str(raw.get("reason") or "").split())[:1000] or None,
        }
    return result


def _representative_occurrence(
    occurrences: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not occurrences:
        return None
    return max(
        occurrences,
        key=lambda item: (
            bool(item.get("name_candidates")),
            bool(item.get("nearby_text")),
            -int(item.get("page_number") or 0),
            _display_area(item),
        ),
    )


def _display_area(occurrence: dict[str, Any]) -> float:
    bbox = occurrence.get("bbox") or [0, 0, 0, 0]
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def _is_icon_sized_review_candidate(occurrence: dict[str, Any]) -> bool:
    if not occurrence.get("name_candidates"):
        return False
    bbox = occurrence.get("bbox") or []
    if len(bbox) != 4:
        return False
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    page_width = max(1.0, float(occurrence.get("page_width") or 1.0))
    page_height = max(1.0, float(occurrence.get("page_height") or 1.0))
    return (
        width <= page_width * 0.20
        and height <= page_height * 0.16
        and width * height <= page_width * page_height * 0.03
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


def _enabled() -> bool:
    return str(os.getenv("VISUAL_ASSET_CLASSIFICATION_VLM_ENABLED", "true")).strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _clean_env(value: str | None) -> str | None:
    cleaned = str(value or "").strip().strip('"').strip("'")
    return cleaned or None
