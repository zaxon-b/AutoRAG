"""VLM verification for general owner-manual visual candidates.

Retrieval similarity only proposes candidates.  This service compares the
uploaded image with each candidate image and may explicitly choose
none-of-the-above.  It accepts every visual modality actually present in the
current manual index, not just dashboard icons.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from service.icon_catalog_service import canonical_icon_name_for_item, canonicalize_evidence
from service.manual_storage_service import storage_root
from service.visual_candidate_service import VISUAL_MODALITIES, _perceptual_fingerprint


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


def verify_visual_candidates(
    query_image_path: Path,
    candidates: list[dict],
    *,
    vision: dict | None = None,
    question: str | None = None,
    top_k: int = 8,
) -> dict:
    """Select the most relevant manual visual chunk, or reject all of them."""

    if not candidates:
        return _verification_failure("No visual candidates were retrieved.", "vlm_visual_verification_skipped")
    if not _vlm_api_key():
        return _verification_failure("VLM_API_KEY is not configured.", "vlm_visual_verification_error")

    prepared: list[dict] = []
    for candidate in candidates[:top_k]:
        candidate = canonicalize_evidence(candidate)
        if candidate.get("modality") not in VISUAL_MODALITIES:
            continue
        local_path = _local_image_path(candidate.get("image_path") or candidate.get("thumbnail_path"))
        if not local_path or not local_path.exists():
            continue
        prepared.append(
            {
                "index": len(prepared) + 1,
                "chunk_id": candidate.get("chunk_id"),
                "title": _candidate_title(candidate),
                "section_title": candidate.get("section_title"),
                "object_name": candidate.get("object_name"),
                "content_text": candidate.get("content_text"),
                "page": candidate.get("page"),
                "modality": candidate.get("modality"),
                "icon_group_id": candidate.get("icon_group_id"),
                "agreement_routes": candidate.get("agreement_routes"),
                "image_path": local_path,
            }
        )
    if not prepared:
        return _verification_failure(
            "Retrieved candidates do not have readable local images.",
            "vlm_visual_verification_error",
        )

    ambiguous = _ambiguous_operation_candidates(query_image_path, prepared)
    if ambiguous:
        titles = "、".join(
            str(item.get("object_name") or item.get("title") or item.get("chunk_id"))
            for item in ambiguous[:4]
        )
        return _verification_failure(
            f"该操作图在手册中对应多个不同说明（{titles}），仅凭裁剪图无法确定具体出现位置；请上传带标题或周围正文的截图。",
            "ambiguous_operation_occurrence",
        )

    # Verification is deliberately image-only. First-pass names, candidate
    # titles and manual text are useful after selection, but can bias the VLM
    # into choosing a semantically related image that does not actually match.
    del vision, question
    selection_rule = (
        "Exactly one candidate is provided. If it is the same visual object, you MUST return selected_index=1 and reject_all=false; otherwise return selected_index=null and reject_all=true."
        if len(prepared) == 1
        else "Select the one matching candidate index, or reject all candidates when none matches."
    )
    prompt = {
        "task": "Compare the user image with owner-manual chunk images and select one only when it depicts the same visual identity/object, not merely a related topic.",
        "rules": [
            selection_rule,
            "Judge only the visible images. Candidate numbers carry no semantic meaning.",
            "icon, operation_figure, and page_snapshot are all valid selectable modalities.",
            "Use OCR and visible layout only when they are actually visible in both images.",
            "Compare concrete visible identity: internal symbol, outline, colour/state and component arrangement.",
            "Match at the same visual scope: a single-symbol query must not select a full page, dashboard or figure merely because that larger image contains a tiny similar symbol.",
            "When candidates share a base glyph, distinguishing marks such as OFF text, a slash, an exclamation mark, arrows or extra strokes take priority over the shared outline.",
            "Every matching feature you cite must be visibly present in both images. Never infer or hallucinate text or a modifier that is absent from the candidate image.",
            "Allow normal camera/display variation in hue, saturation, antialiasing, line thickness and background; do not reject the same glyph only because red appears orange or the background changed.",
            "Treat colour as decisive only when candidates share the same shape and colour is the visible feature that distinguishes different manual states.",
            "Do not match from generic colour, geometric shape, background or topical relevance alone.",
            "If the image is too blurry or none depicts the same object, set reject_all=true and selected_index=null.",
            "Never select a weak candidate merely because candidates were supplied.",
            "Return JSON only.",
        ],
        "output_schema": {
            "selected_index": (
                "integer 1 when the only candidate matches; otherwise null"
                if len(prepared) == 1
                else "candidate integer index or null"
            ),
            "reject_all": "boolean",
            "confidence": "0.0-1.0",
            "matching_features": ["short visible similarities supporting the selection"],
            "contradictions": ["visible differences that make the selection uncertain"],
            "reason": "short Chinese explanation grounded in visible comparison",
        },
    }

    content: list[dict] = [
        {"type": "text", "text": "用户上传图片："},
        {"type": "image_url", "image_url": {"url": _image_data_url(query_image_path)}},
        {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
    ]
    for item in prepared:
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"候选 {item['index']}：",
                },
                {"type": "image_url", "image_url": {"url": _image_data_url(item["image_path"])}},
            ]
        )

    try:
        client = OpenAI(api_key=_vlm_api_key(), base_url=_vlm_base_url())
        completion = client.chat.completions.create(
            model=_vlm_model(),
            messages=[{"role": "user", "content": content}],
            stream=False,
            temperature=0,
            max_tokens=512,
        )
        parsed = _parse_json_object(completion.choices[0].message.content or "{}")
        verdict = _validate_verification_payload(parsed, prepared)
        selected_id = verdict["selected_chunk_id"]
        matched = verdict["matched"]
        confidence = verdict["confidence"]
        selected = next((item for item in prepared if item.get("chunk_id") == selected_id), None)
        return {
            "matched": matched,
            "selected_chunk_id": selected_id,
            "selected_index": selected.get("index") if selected else None,
            "selected_title": (selected or {}).get("title"),
            # Compatibility alias used by historical icon evaluation code.
            "selected_icon_name": (selected or {}).get("title"),
            "confidence": confidence,
            "reason": parsed.get("reason"),
            "reject_all": verdict["reject_all"],
            "matching_features": verdict["matching_features"],
            "contradictions": verdict["contradictions"],
            "same_visual_identity": verdict["same_visual_identity"],
            "semantic_object_match": verdict["semantic_object_match"],
            "shared_visual_features": verdict["shared_visual_features"],
            "contradictory_visual_features": verdict["contradictory_visual_features"],
            "method": "vlm_visual_verification",
        }
    except Exception as error:
        return _verification_failure(
            f"{type(error).__name__}: {error}",
            "vlm_visual_verification_error",
        )


def verify_icon_candidates(query_image_path: Path, candidates: list[dict], top_k: int = 8) -> dict:
    """Compatibility wrapper for historical icon-only evaluation scripts."""

    return verify_visual_candidates(query_image_path, candidates, top_k=top_k)


def _verification_failure(reason: str, method: str) -> dict:
    return {
        "matched": False,
        "selected_chunk_id": None,
        "selected_index": None,
        "selected_title": None,
        "selected_icon_name": None,
        "confidence": 0.0,
        "reason": reason,
        "reject_all": True,
        "matching_features": [],
        "contradictions": [],
        "same_visual_identity": False,
        "semantic_object_match": False,
        "shared_visual_features": [],
        "contradictory_visual_features": [],
        "method": method,
    }


def _ambiguous_operation_candidates(query_image_path: Path, prepared: list[dict]) -> list[dict]:
    """Reject identical operation crops that carry different manual meanings."""

    query_fingerprint = _perceptual_fingerprint(str(query_image_path))
    if not query_fingerprint:
        return []
    matches = [
        item
        for item in prepared
        if item.get("modality") == "operation_figure"
        and _perceptual_fingerprint(str(item.get("image_path") or "")) == query_fingerprint
    ]
    semantic_keys = {
        _normalized_semantic_key(item)
        for item in matches
        if _normalized_semantic_key(item)
    }
    return matches if len(matches) >= 2 and len(semantic_keys) >= 2 else []


def _normalized_semantic_key(item: dict) -> str:
    value = item.get("object_name") or item.get("content_text") or item.get("section_title") or ""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).casefold())


def _validate_verification_payload(parsed: dict, prepared: list[dict]) -> dict:
    """Map a constrained candidate index to server-owned chunk metadata."""

    selected_id = None
    selected_index = None
    if parsed.get("selected_index") is not None:
        try:
            selected_index = int(parsed["selected_index"])
            selected_id = _chunk_id_for_index(prepared, selected_index)
        except (TypeError, ValueError):
            selected_id = None
    reject_all_value = parsed.get("reject_all")
    reject_all = reject_all_value is True
    matching = _as_string_list(parsed.get("matching_features"), limit=8)
    contradictions = _as_string_list(parsed.get("contradictions"), limit=8)
    # Fail closed when reject_all is missing or not a real JSON boolean.
    matched = reject_all_value is False and selected_id is not None
    if not matched:
        selected_id = None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "matched": matched,
        "selected_chunk_id": selected_id,
        "selected_index": selected_index if matched else None,
        "confidence": confidence,
        "reject_all": reject_all,
        "matching_features": matching,
        "contradictions": contradictions,
        # Compatibility fields retained for existing reports; they are derived
        # rather than additional independent VLM gates.
        "same_visual_identity": matched,
        "semantic_object_match": None,
        "shared_visual_features": matching,
        "contradictory_visual_features": contradictions,
    }


def _as_string_list(value, *, limit: int) -> list[str]:
    values = value if isinstance(value, list) else [value] if value else []
    result: list[str] = []
    for item in values:
        text = " ".join(str(item or "").split())
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _candidate_title(candidate: dict) -> str | None:
    return (
        canonical_icon_name_for_item(candidate)
        or candidate.get("object_name")
        or candidate.get("section_title")
    )


def _chunk_id_for_index(items: list[dict], index: int) -> str | None:
    item = next((item for item in items if item["index"] == index), None)
    return item.get("chunk_id") if item else None


def _local_image_path(value: str | None) -> Path | None:
    if not value:
        return None
    if value.startswith("/storage/"):
        return storage_root() / value.removeprefix("/storage/")
    path = Path(value)
    return path if path.is_absolute() else storage_root() / path


def _image_data_url(path: Path) -> str:
    """Encode an image for VLM inspection, enlarging only tiny inputs in memory."""

    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            longest = max(height, width)
            if 0 < longest < 256:
                scale = 256.0 / longest
                image = cv2.resize(
                    image,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_NEAREST,
                )
                encoded, buffer = cv2.imencode(".png", image)
                if encoded:
                    return f"data:image/png;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
    except Exception:
        # The original image is still a valid fallback; VLM verification must
        # never fail only because optional display upscaling failed.
        pass
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        mime = "image/webp"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


def _parse_json_object(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        return json.loads(cleaned[start : end + 1]) if start >= 0 and end > start else {}


def _icon_name_from_caption(caption: str | None) -> str | None:
    """Compatibility helper retained for old imports."""

    if not caption:
        return None
    match = re.search(r"疑似仪表图标[:：]\s*([^。]+)", caption)
    return match.group(1).strip() if match else None
