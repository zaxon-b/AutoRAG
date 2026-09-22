"""VLM image understanding for the general owner-manual image-QA path.

This first VLM call describes the upload and builds retrieval language.  Its
``image_type`` and ``confidence`` are metadata only: neither field is allowed
to stop visual retrieval or restrict the endpoint to dashboard icons.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI


IMAGE_TYPES = {
    "icon",
    "operation_figure",
    "component_photo",
    "page_snapshot",
    "unknown",
}
_IMAGE_TYPE_ALIASES = {
    "warning_icon": "icon",
    "button_icon": "icon",
    "button_symbol": "icon",
    "operation_illustration": "operation_figure",
    "app_screenshot": "page_snapshot",
    "page_screenshot": "page_snapshot",
}


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def _vlm_api_key() -> str | None:
    return _clean_env(os.getenv("VLM_API_KEY"))


def _vlm_base_url() -> str | None:
    return _clean_env(os.getenv("VLM_BASE_URL")) or _clean_env(os.getenv("DASHSCOPE_BASE_URL"))


def _vlm_model() -> str:
    return _clean_env(os.getenv("VLM_MODEL")) or "qwen-vl-plus"


def _image_data_url(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        mime = "image/webp"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _parse_json_object(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _as_string_list(value, *, limit: int = 8) -> list[str]:
    values = value if isinstance(value, list) else [value] if value else []
    result: list[str] = []
    for item in values:
        text = " ".join(str(item or "").split())
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _normalise_image_understanding(parsed: dict, question: str | None = None) -> dict:
    """Validate the seven first-pass fields and derive legacy display aliases."""

    image_type = str(parsed.get("image_type") or "unknown").strip().lower()
    image_type = _IMAGE_TYPE_ALIASES.get(image_type, image_type)
    if image_type not in IMAGE_TYPES:
        image_type = "unknown"
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0)))
    except (TypeError, ValueError):
        confidence = 0.0

    visual_description = " ".join(str(parsed.get("visual_description") or "").split())
    ocr_text = " ".join(str(parsed.get("ocr_text") or "").split())
    candidate_names = _as_string_list(parsed.get("candidate_names"), limit=5)
    visual_keywords = _as_string_list(parsed.get("visual_keywords"), limit=8)
    normalized_query = " ".join(str(parsed.get("normalized_query") or "").split())
    if not normalized_query:
        normalized_query = " ".join(
            part
            for part in [
                question or "",
                visual_description,
                ocr_text,
                " ".join(candidate_names),
                " ".join(visual_keywords),
            ]
            if part
        )

    result = {
        "visual_description": visual_description,
        "ocr_text": ocr_text,
        "candidate_names": candidate_names,
        "visual_keywords": visual_keywords,
        "normalized_query": normalized_query,
        "image_type": image_type,
        "confidence": confidence,
    }

    # Backward-compatible response fields.  They are derived metadata only and
    # are never consulted by the production router as a retrieval gate.
    result["recognized_object"] = candidate_names[0] if candidate_names else "unknown"
    result["risk_level"] = "unknown"
    result["is_target_icon"] = image_type == "icon"
    result["classification_confidence"] = confidence
    result["classification_reason"] = "image_type is descriptive metadata and does not route the request"
    return result


def _normalise_image_classification(parsed: dict) -> dict:
    """Compatibility wrapper retained for older imports and cached reports."""

    if "confidence" not in parsed and "classification_confidence" in parsed:
        parsed = {**parsed, "confidence": parsed.get("classification_confidence")}
    return _normalise_image_understanding(parsed)


def _enrich_query(parsed: dict, question: str | None = None) -> dict:
    """Compatibility wrapper; query enrichment now happens during validation."""

    return _normalise_image_understanding(parsed, question)


def understand_vehicle_image(image_path: Path, question: str | None = None) -> dict:
    """Describe an uploaded image without making a hard routing decision."""

    fallback = _normalise_image_understanding(
        {
            "visual_description": "图片理解服务暂时不可用。",
            "ocr_text": "",
            "candidate_names": [],
            "visual_keywords": [],
            "normalized_query": question or "车辆手册 图片 内容 含义 使用方法",
            "image_type": "unknown",
            "confidence": 0.0,
        },
        question,
    )
    fallback["vision_error"] = None
    if not _vlm_api_key():
        return {**fallback, "vision_error": "VLM_API_KEY is not configured."}

    prompt = """请理解用户上传的图片，为车主手册视觉检索生成结构化描述。只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。

要求：
- 只描述图片，不做是否继续检索或是否拒答的决定。
- image_type 只能是 icon、operation_figure、component_photo、page_snapshot、unknown 之一；它只是元数据。
- icon 包括警告、状态、操作、按钮等各种小图标，不能用名称后缀限制它。
- 与车辆/手册无关的照片、绘画或抽象图使用 unknown，不要为了填满类别而强行归为 operation_figure。
- visual_description 客观描述颜色、形状、对象、界面结构和上下文，避免下维修结论。
- 图片明显与车辆/车主手册无关时，在 visual_description 中明确写“与车辆/车主手册无关”；图片模糊到无法辨认时明确写“图片模糊无法辨认”。这仍然只是描述，不做路由决定。
- ocr_text 原样提取清晰可见的文字；没有可读文字时返回空字符串。
- candidate_names 给出 0-5 个可能的对象/功能/图示名称，不局限于仪表图标。
- visual_keywords 给出适合视觉对象解释的短关键词。
- normalized_query 面向当前车主手册检索，合并可见对象、OCR、候选名称和用户意图。
- confidence 表示对图片内容理解的信心，不是路由阈值。

{
  "visual_description": "图片可见内容",
  "ocr_text": "可见文字或空字符串",
  "candidate_names": ["候选名称1", "候选名称2"],
  "visual_keywords": ["关键词1", "关键词2"],
  "normalized_query": "用于检索车主手册的查询",
  "image_type": "icon|operation_figure|component_photo|page_snapshot|unknown",
  "confidence": 0.0
}
"""
    if question:
        prompt += f"\n用户问题：{question}"

    try:
        client = OpenAI(api_key=_vlm_api_key(), base_url=_vlm_base_url())
        completion = client.chat.completions.create(
            model=_vlm_model(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                }
            ],
            stream=False,
            max_tokens=512,
            temperature=0,
        )
        parsed = _parse_json_object(completion.choices[0].message.content or "{}")
        return {**fallback, **_normalise_image_understanding(parsed, question), "vision_error": None}
    except Exception as error:
        return {**fallback, "vision_error": f"{type(error).__name__}: {error}"}
