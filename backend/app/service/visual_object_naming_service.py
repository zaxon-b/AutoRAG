"""Assign bounded visual object names in batched VLM requests.

Icon names are never generated freely: the model may only select one literal
candidate extracted from the manual around that occurrence.  Operation figure
names are short VLM summaries used as retrieval/display hints.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

ICON_NAME_BATCH_SIZE = max(1, min(4, int(os.getenv("VISUAL_ICON_NAME_BATCH_SIZE", "1"))))
OPERATION_NAME_BATCH_SIZE = max(1, min(8, int(os.getenv("VISUAL_OPERATION_NAME_BATCH_SIZE", "4"))))
NAME_WORKERS = max(1, min(6, int(os.getenv("VISUAL_OBJECT_NAME_WORKERS", "3"))))
OPERATION_NAME_MAX_CHARS = max(8, min(80, int(os.getenv("OPERATION_OBJECT_NAME_MAX_CHARS", "40"))))


def assign_visual_object_names(payloads: list[dict[str, Any]]) -> None:
    """Mutate icon/operation payloads with validated object_name values."""

    targets = [
        payload
        for payload in payloads
        if payload.get("modality") in {"icon", "operation_figure"}
        and payload.get("image_path")
        and Path(str(payload["image_path"])).exists()
    ]
    if not targets or not _vlm_api_key() or not _enabled():
        return

    icons = [payload for payload in targets if payload.get("modality") == "icon"]
    operations = [payload for payload in targets if payload.get("modality") == "operation_figure"]
    # Icon identity is easy to shift between adjacent table cells when several
    # tiny symbols share one VLM request.  Default to one icon per request;
    # operation figures remain safely batched because their grounded names are
    # checked against their own section/body text.
    batches = [
        icons[start : start + ICON_NAME_BATCH_SIZE]
        for start in range(0, len(icons), ICON_NAME_BATCH_SIZE)
    ]
    batches.extend(
        operations[start : start + OPERATION_NAME_BATCH_SIZE]
        for start in range(0, len(operations), OPERATION_NAME_BATCH_SIZE)
    )
    results: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=min(NAME_WORKERS, len(batches))) as executor:
        futures = {executor.submit(_name_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                results.update(future.result())
            except Exception as error:
                logger.warning(
                    "Visual object naming batch failed: chunks=%s error=%s",
                    [item.get("chunk_id") for item in batch],
                    error,
                )

    for payload in targets:
        name = results.get(str(payload.get("chunk_id") or ""))
        payload["object_name"] = name[:255] if name else None


def _name_batch(batch: list[dict[str, Any]]) -> dict[str, str | None]:
    from openai import OpenAI

    prepared: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "你在给车主手册中的视觉对象命名。逐项看图和手册原文，只输出 JSON 对象，不要 Markdown。\n"
                "icon：object_name 只能原样选择该项 candidate_names 中的一个字符串；无法确认则为 null，绝对不能改写或创造名称。\n"
                "icon 的唯一候选已经由 PDF 同一表格单元格或同一行定位；只有图片与该名称明显冲突时才返回 null。\n"
                "operation_figure：根据图和对应正文生成简短具体的对象名，不超过"
                f"{OPERATION_NAME_MAX_CHARS}个字符；不能命名为‘温馨提示’、‘注意’、‘警告’、‘图片’或‘示意图’。\n"
                "输出格式：{\"items\":[{\"index\":1,\"object_name\":\"名称或null\"}]}"
            ),
        }
    ]
    for index, payload in enumerate(batch, start=1):
        candidates = _candidate_names(payload)
        prepared.append(
            {
                "index": index,
                "chunk_id": str(payload.get("chunk_id") or ""),
                "modality": str(payload.get("modality") or ""),
                "candidates": candidates,
                "source": "\n".join(
                    part
                    for part in [
                        str(payload.get("section_title") or ""),
                        str(payload.get("content_text") or ""),
                        "\n".join(candidates),
                    ]
                    if part
                ),
            }
        )
        metadata = {
            "index": index,
            "modality": payload.get("modality"),
            "section_title": payload.get("section_title"),
            "candidate_names": candidates,
            "manual_text": str(payload.get("content_text") or "")[:1800],
        }
        content.extend(
            [
                {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": _image_data_url(Path(str(payload["image_path"]))) }},
            ]
        )

    client = OpenAI(api_key=_vlm_api_key(), base_url=_vlm_base_url())
    completion = client.chat.completions.create(
        model=_vlm_model(),
        messages=[{"role": "user", "content": content}],
        stream=False,
        temperature=0,
        max_tokens=max(512, len(batch) * 80),
    )
    parsed = _parse_json_object(str(completion.choices[0].message.content or "{}"))
    by_index = {
        int(item.get("index")): item.get("object_name")
        for item in parsed.get("items") or []
        if isinstance(item, dict) and str(item.get("index") or "").isdigit()
    }

    result: dict[str, str | None] = {}
    for item in prepared:
        raw_name = by_index.get(item["index"])
        if item["modality"] == "icon":
            result[item["chunk_id"]] = _validate_icon_name(raw_name, item["candidates"])
        else:
            result[item["chunk_id"]] = _validate_operation_name(raw_name, item["source"])
    return result


def _candidate_names(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for value in payload.get("_name_candidates") or []:
        text = " ".join(str(value or "").split()).strip(" ：:，,。")
        if 2 <= len("".join(text.split())) <= 48 and text not in result:
            result.append(text)
    # Icon candidates arrive in spatial priority order.  The first value is
    # the text in the same row/cell and is therefore the only manual name that
    # belongs to this occurrence.  Passing neighbouring rows to the VLM makes
    # tiny dashboard symbols easy to mislabel even when the correct label is
    # already ranked first.  The VLM still confirms the crop, but can no longer
    # select a different row's name.
    if payload.get("modality") == "icon":
        return result[:1]
    return result[:16]


def _validate_icon_name(value: Any, candidates: list[str]) -> str | None:
    if value is None:
        return None
    supplied = " ".join(str(value).split()).strip(" ：:，,。")
    return next((candidate for candidate in candidates if supplied == candidate), None)


def _validate_operation_name(value: Any, source: str | None = None) -> str | None:
    text = " ".join(str(value or "").split()).strip(" ：:，,。")
    if not text or len(text) > OPERATION_NAME_MAX_CHARS:
        return None
    normalized = re.sub(r"\s+", "", text)
    forbidden = {"温馨提示", "注意", "警告", "图片", "示意图", "null", "none", "未知", "无法确认"}
    if normalized in forbidden:
        return None
    if source and not _operation_name_is_grounded(text, source):
        return None
    return text


def _operation_name_is_grounded(name: str, source: str) -> bool:
    normalized_source = re.sub(r"\s+", "", source).lower()
    core = re.sub(r"(示意图|结构图|操作图|工作图|图)$", "", re.sub(r"\s+", "", name)).lower()
    if len(core) >= 2 and core in normalized_source:
        return True

    latin_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", name)]
    if any(term in normalized_source for term in latin_terms):
        return True

    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", core))
    ignored = {"工作", "模式", "系统", "操作", "示意", "结构", "装置", "车辆", "图片"}
    bigrams = {
        chinese[index : index + 2]
        for index in range(max(0, len(chinese) - 1))
        if chinese[index : index + 2] not in ignored
    }
    return bool(bigrams and any(term in normalized_source for term in bigrams))


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
    return str(os.getenv("MANUAL_ICON_NAME_VLM_ENABLED", "true")).strip().lower() not in {"0", "false", "no"}


def _vlm_api_key() -> str | None:
    return _clean_env(os.getenv("VLM_API_KEY"))


def _vlm_base_url() -> str | None:
    return _clean_env(os.getenv("VLM_BASE_URL")) or _clean_env(os.getenv("DASHSCOPE_BASE_URL"))


def _vlm_model() -> str:
    return _clean_env(os.getenv("VLM_MODEL")) or "qwen-vl-plus"


def _clean_env(value: str | None) -> str | None:
    cleaned = str(value or "").strip().strip('"').strip("'")
    return cleaned or None
