"""Text QA and unified owner-manual image QA routes."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from schemas.qa import ImageQaResponse, TextQaRequest
from service.answer_generation_service import complete_answer, stream_answer
from service.icon_catalog_service import canonical_icon_name_for_item, canonicalize_evidence, sort_visual_evidence
from service.manual_storage_service import DEFAULT_MANUAL_ID, save_query_image
from service.retrieval_service import retrieve_evidence, retrieve_visual_anchor_text
from service.rrf_fusion_service import rrf_fuse
from service.vision_rerank_service import verify_visual_candidates
from service.vision_understanding_service import understand_vehicle_image
from service.visual_candidate_service import (
    VISUAL_MODALITIES,
    deduplicate_visual_candidates,
    union_group_fusion,
)
from service.visual_similarity_service import (
    retrieve_visual_chunks_by_image_embedding,
    retrieve_visual_chunks_by_multimodal,
)
from utils.database import get_db


router = APIRouter(prefix="/qa", tags=["qa"])

VLM_VISUAL_VERIFICATION_THRESHOLD = float(
    os.getenv(
        "VLM_VISUAL_VERIFICATION_THRESHOLD",
        os.getenv("VLM_ICON_VERIFICATION_THRESHOLD", "0.75"),
    )
)
# Compatibility alias for callers that still import the previous setting.
VLM_ICON_VERIFICATION_THRESHOLD = VLM_VISUAL_VERIFICATION_THRESHOLD
ENABLE_VLM_VISUAL_VERIFICATION = os.getenv(
    "ENABLE_VLM_VISUAL_VERIFICATION", "false"
).strip().lower() in {"1", "true", "yes", "on"}


@router.post("/text")
async def text_qa(request: TextQaRequest, db: Session = Depends(get_db)):
    """Answer a text question through the unchanged hybrid text pipeline."""

    evidence = retrieve_evidence(db, request.manual_id, request.question)
    return StreamingResponse(
        stream_answer(request.question, evidence),
        media_type="text/event-stream",
    )


@router.post("/image", response_model=ImageQaResponse)
async def image_qa(
    manual_id: str = Form(DEFAULT_MANUAL_ID),
    question: str | None = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Answer questions about any visual object supported by the manual.

    ``image_type`` from first-pass understanding is logged and exposed to the
    client, but it never blocks retrieval. Production uses the first candidate
    from the evaluated two-route RRF. Candidate-changing VLM verification is
    retained behind an opt-in flag for offline experiments.
    """

    suffix = (image.filename or "").split(".")[-1].lower()
    if suffix and suffix not in {"png", "jpg", "jpeg", "webp"}:
        raise HTTPException(status_code=400, detail="Only png, jpg, jpeg and webp images are supported.")

    content = await image.read()
    image_path = save_query_image(content, image.filename or "query.png", manual_id)
    vision = understand_vehicle_image(image_path, question)

    # The two production embedding routes run concurrently. Each one queries visual
    # modalities directly and returns at most 20 raw visual chunks.
    raw_routes = await _retrieve_visual_routes(db, manual_id, image_path, top_k=20)
    image_candidates = deduplicate_visual_candidates(
        db, manual_id, raw_routes["image_embedding"], max_items=20
    )
    multimodal_candidates = deduplicate_visual_candidates(
        db, manual_id, raw_routes["multimodal_embedding"], max_items=20
    )
    pre_vlm_candidates = union_group_fusion(
        image_candidates,
        [],
        multimodal_candidates,
        top_k=5,
        image_weight=1.0,
        multimodal_weight=1.0,
    )

    if vision.get("vision_error"):
        return _insufficient_image_response(
            vision,
            status="vision_understanding_failed",
            reason="图片理解服务调用失败，当前无法可靠完成视觉核对。",
        )
    if not pre_vlm_candidates:
        status, reason = _no_visual_candidate_failure_message(vision)
        return _insufficient_image_response(
            vision,
            status=status,
            reason=reason,
        )

    anchor_method = "rrf_visual_anchor"
    if ENABLE_VLM_VISUAL_VERIFICATION:
        verification = verify_visual_candidates(
            image_path,
            pre_vlm_candidates,
            vision=vision,
            question=question,
            top_k=5,
        )
        anchored_candidates = _apply_vlm_verification(pre_vlm_candidates, verification)
        visual_anchor = _verified_visual_anchor(anchored_candidates)
        if visual_anchor is None:
            status, reason = _verification_failure_message(vision, verification)
            return _insufficient_image_response(
                vision,
                status=status,
                reason=reason,
                visual_match=_build_visual_match(anchored_candidates),
            )
        anchor_method = "vlm_visual_anchor"
    else:
        visual_anchor = _apply_retrieval_anchor(pre_vlm_candidates[0])

    # Body retrieval is deliberately downstream of visual verification.  It
    # is one BM25/Dense -> RRF -> Cross-Encoder path, not parallel special and
    # generic RAG routes.
    visual_evidence = dict(canonicalize_evidence(visual_anchor))
    visual_evidence["evidence_role"] = "visual"
    visual_evidence["match_method"] = _merge_match_methods(
        visual_evidence.get("match_method"), anchor_method
    )
    visual_match = _build_visual_match([visual_evidence])
    text_query = _build_image_text_query(vision, question, visual_anchor)
    text_evidence = retrieve_visual_anchor_text(
        db,
        manual_id,
        text_query,
        _build_visual_anchor(visual_anchor),
        top_k=4,
    )[:4]
    text_evidence = [{**item, "evidence_role": "text"} for item in text_evidence[:4]]
    evidence = [visual_evidence, *text_evidence]

    if not text_evidence:
        return _verified_image_response(
            vision=vision,
            text_query=text_query,
            visual_evidence=visual_evidence,
            text_evidence=[],
            visual_match=visual_match,
            answer=_build_insufficient_text_answer(visual_evidence),
            evidence_status="insufficient_text_evidence",
        )

    answer_question = _build_image_answer_question(
        vision,
        question,
        text_query,
        visual_match,
    )
    answer = complete_answer(answer_question, evidence)
    return _verified_image_response(
        vision=vision,
        text_query=text_query,
        visual_evidence=visual_evidence,
        text_evidence=text_evidence,
        visual_match=visual_match,
        answer=answer,
        evidence_status="verified",
    )


def _verified_image_response(
    *,
    vision: dict,
    text_query: str,
    visual_evidence: dict,
    text_evidence: list[dict],
    visual_match: dict | None,
    answer: str,
    evidence_status: str,
) -> dict:
    """Build a compatible response after a production visual anchor is selected."""

    title = _visual_title(visual_evidence) or vision.get("recognized_object") or "unknown"
    return {
        "recognized_object": title,
        "image_type": vision.get("image_type"),
        "is_target_icon": vision.get("is_target_icon"),
        "classification_confidence": vision.get("classification_confidence"),
        "classification_reason": vision.get("classification_reason"),
        "visual_description": vision.get("visual_description"),
        "visual_keywords": vision.get("visual_keywords"),
        "candidate_names": vision.get("candidate_names"),
        "normalized_query": text_query,
        "risk_level": visual_evidence.get("risk_level") or vision.get("risk_level") or "unknown",
        "confidence": (
            visual_match.get("vlm_confidence") or vision.get("confidence")
            if visual_match
            else vision.get("confidence")
        ),
        "vision_error": vision.get("vision_error"),
        "visual_match": visual_match,
        "answer": answer,
        "suggestion": (
            "请结合视觉证据确认手册对象，并优先参考正文证据中的官方说明；"
            "涉及安全、制动、电池或动力系统异常时，请安全停车并联系官方售后。"
        ),
        "evidence": [visual_evidence, *text_evidence],
        "visual_evidence": visual_evidence,
        "text_evidence": text_evidence,
        "evidence_status": evidence_status,
    }


async def _retrieve_visual_routes(
    db: Session,
    manual_id: str,
    image_path: Path,
    *,
    top_k: int = 20,
) -> dict[str, list[dict]]:
    """Run the two production visual retrieval routes concurrently."""

    del db

    image_task = asyncio.to_thread(
        retrieve_visual_chunks_by_image_embedding,
        manual_id,
        image_path,
        top_k,
        embedding_variant="alpha_white",
    )
    multimodal_task = asyncio.to_thread(
        retrieve_visual_chunks_by_multimodal,
        manual_id,
        image_path,
        top_k,
    )
    image_items, multimodal_items = await asyncio.gather(
        image_task,
        multimodal_task,
    )
    return {
        "image_embedding": image_items,
        "multimodal_embedding": multimodal_items,
        # OpenCV remains in the evaluator as a diagnostic ablation. It reduced
        # production ranking quality in the 90-case regression and is not run
        # on the online path.
        "opencv": [],
    }


def _build_image_text_query(vision: dict, question: str | None, visual_anchor: dict) -> str:
    """Build the single body-search query after a visual anchor is verified."""

    parts: list[str] = []
    for value in (
        question,
        vision.get("normalized_query"),
        str(vision.get("visual_description") or "")[:240],
        str(vision.get("ocr_text") or "")[:300],
        _visual_title(visual_anchor),
        str(visual_anchor.get("section_title") or "")[:300],
        str(visual_anchor.get("content_text") or "")[:300],
        " ".join(str(item) for item in (vision.get("candidate_names") or [])[:5]),
        " ".join(str(item) for item in (vision.get("visual_keywords") or [])[:8]),
    ):
        _append_unique(parts, value)
    return " ".join(parts) or "车辆手册 图片 内容 含义 使用方法"


def _build_image_retrieval_queries(vision: dict, question: str | None) -> list[str]:
    """Compatibility helper returning the pre-anchor understanding query."""

    parts: list[str] = []
    for value in (
        question,
        vision.get("normalized_query"),
        vision.get("visual_description"),
        vision.get("ocr_text"),
        " ".join(str(item) for item in (vision.get("candidate_names") or [])[:5]),
        " ".join(str(item) for item in (vision.get("visual_keywords") or [])[:8]),
    ):
        _append_unique(parts, value)
    return [" ".join(parts) or "车辆手册 图片 内容 含义 使用方法"]


def _append_unique(items: list[str], value) -> None:
    text = " ".join(str(value or "").split())
    if text and text not in items:
        items.append(text)


def _build_visual_anchor(item: dict) -> dict:
    """Expose only fields that are meaningful for associated body retrieval."""

    return {
        "chunk_id": item.get("chunk_id"),
        "object_name": item.get("object_name"),
        "section_title": item.get("section_title"),
        "content_text": item.get("content_text"),
        "page": item.get("page"),
        "modality": item.get("modality"),
        "icon_group_id": item.get("icon_group_id"),
    }


def _retrieve_merged_evidence(
    db: Session,
    manual_id: str,
    queries: list[str],
    visual_evidence: list[dict] | None = None,
    multimodal_evidence: list[dict] | None = None,
    diagnostics: dict | None = None,
) -> list[dict]:
    """Compatibility wrapper around the unified post-anchor text retriever.

    The ``multimodal_evidence`` argument is intentionally ignored: raw
    multimodal results are agreement signals only and never enter final
    evidence.  No cross-modal RRF is performed here.
    """

    del multimodal_evidence
    anchor = _verified_visual_anchor(visual_evidence or [])
    if anchor is None:
        return []
    query = " ".join(query for query in queries if query).strip()
    text_evidence = retrieve_visual_anchor_text(
        db,
        manual_id,
        query or _visual_title(anchor) or "车辆手册 图片说明",
        _build_visual_anchor(anchor),
        top_k=4,
    )
    visual = {**canonicalize_evidence(anchor), "evidence_role": "visual"}
    text_items = [{**item, "evidence_role": "text"} for item in text_evidence[:4]]
    if diagnostics is not None:
        diagnostics["text_support_candidates"] = text_items
        diagnostics["text_evidence"] = text_items
    return [visual, *text_items]


def _apply_vlm_verification(visual_evidence: list[dict], verification: dict | None) -> list[dict]:
    """Mark and promote only the candidate explicitly selected by the VLM."""

    if not visual_evidence:
        return []
    verification = verification or {}
    selected_id = verification.get("selected_chunk_id") if verification.get("matched") else None
    promoted: list[dict] = []
    others: list[dict] = []
    for item in visual_evidence:
        copied = canonicalize_evidence(item)
        selected = bool(selected_id) and copied.get("chunk_id") == selected_id
        copied["vlm_verified"] = selected
        copied["vlm_confidence"] = verification.get("confidence")
        copied["vlm_reason"] = verification.get("reason")
        if selected:
            selected_title = verification.get("selected_title") or verification.get("selected_icon_name")
            if selected_title:
                copied["verified_title"] = selected_title
            copied["score"] = max(
                float(copied.get("score") or 0),
                float(verification.get("confidence") or 0) * 100,
            )
            copied["match_method"] = _merge_match_methods(
                copied.get("match_method"), verification.get("method")
            )
            promoted.append(copied)
        else:
            others.append(copied)
    return promoted + others


def _apply_retrieval_anchor(candidate: dict) -> dict:
    """Mark the first candidate from the evaluated production RRF as the anchor."""

    anchored = canonicalize_evidence(candidate)
    anchored["retrieval_anchor"] = True
    anchored["match_method"] = _merge_match_methods(
        anchored.get("match_method"), "rrf_visual_anchor"
    )
    return anchored


def _verified_visual_anchor(visual_evidence: list[dict]) -> dict | None:
    return next((item for item in visual_evidence if _is_strong_visual_match(item)), None)


def _build_visual_match(visual_evidence: list[dict]) -> dict | None:
    if not visual_evidence:
        return None
    top = canonicalize_evidence(visual_evidence[0])
    return {
        "matched": _is_strong_visual_match(top),
        "object_name": top.get("object_name"),
        "section_title": top.get("section_title"),
        "score": top.get("score"),
        "page": top.get("page"),
        "modality": top.get("modality"),
        "risk_level": top.get("risk_level"),
        "image_path": top.get("image_path"),
        "thumbnail_path": top.get("thumbnail_path"),
        "icon_group_id": top.get("icon_group_id"),
        "match_method": top.get("match_method"),
        "vlm_verified": top.get("vlm_verified"),
        "vlm_confidence": top.get("vlm_confidence"),
        "vlm_reason": top.get("vlm_reason"),
        "retrieval_anchor": top.get("retrieval_anchor"),
    }


def _is_strong_visual_match(match: dict | None, vision: dict | None = None) -> bool:
    """Accept either the production RRF anchor or an enabled VLM-verified anchor."""

    if not match or match.get("modality") not in VISUAL_MODALITIES:
        return False
    if vision is not None and vision.get("vision_error"):
        return False
    return match.get("retrieval_anchor") is True or (
        match.get("vlm_verified") is True
        and float(match.get("vlm_confidence") or 0) >= VLM_VISUAL_VERIFICATION_THRESHOLD
    )


def _should_refuse_image_answer(visual_match: dict | None, vision: dict) -> bool:
    """Compatibility name for the verified-anchor evidence check."""

    return not _is_strong_visual_match(visual_match, vision)


def _is_target_icon_image(vision: dict | None) -> bool:
    """Legacy metadata helper; production routing never calls this function."""

    return bool(vision) and vision.get("image_type") == "icon"


def _build_image_answer_question(
    vision: dict,
    question: str | None,
    primary_query: str | None,
    visual_match: dict | None,
) -> str:
    title = (
        (visual_match or {}).get("object_name")
        or (visual_match or {}).get("section_title")
        or "unknown"
    )
    return (
        "这是一次车主手册图片问答。"
        f"用户问题：{question or '这张图片是什么，应该如何理解或使用？'}。"
        f"VLM初诊描述：{vision.get('visual_description') or ''}。"
        f"VLM初诊OCR：{vision.get('ocr_text') or ''}。"
        f"经视觉召回与融合排序确定，视觉锚点为“{title}”，"
        f"modality={(visual_match or {}).get('modality')}，page={(visual_match or {}).get('page')}。"
        f"统一正文检索query：{primary_query or ''}。"
        "证据中role=visual的唯一职责是确认图片对应的手册对象；"
        "role=text证据才用于说明对象含义、使用方法、触发条件和处理建议。"
        "只基于这些证据回答；正文没有支持的结论必须明确说明无法确认。"
    )


def _build_insufficient_text_answer(visual_evidence: dict) -> str:
    """Return a deterministic, grounded answer when body retrieval is empty."""

    title = _visual_title(visual_evidence) or "该视觉对象"
    return (
        f"已在当前手册中确认图片对应“{title}”，但未检索到足以支持用户问题的正文证据，"
        "因此无法可靠说明其含义、使用方法、触发条件或处理建议。"
    )


def _verification_failure_message(vision: dict, verification: dict | None) -> tuple[str, str]:
    verification = verification or {}
    method = str(verification.get("method") or "")
    reason = str(verification.get("reason") or "").strip()
    if method == "ambiguous_operation_occurrence":
        return (
            "visual_occurrence_ambiguous",
            reason
            or "同一操作图在手册中对应多个不同说明，请上传带标题或周围正文的截图。",
        )
    combined = f"{vision.get('visual_description') or ''} {reason}".lower()
    if "error" in method:
        return "visual_verification_failed", "视觉候选二次验证调用失败，当前不会基于相似度猜测答案。"
    if any(token in combined for token in ("模糊", "看不清", "无法辨认", "blur", "illegible")):
        return "image_too_blurry", "图片过于模糊，无法与手册候选可靠比较，请重新上传更清晰的图片。"
    if any(token in combined for token in ("与汽车无关", "非车辆", "unrelated", "not vehicle")):
        return "image_unrelated", "图片内容可能与当前汽车手册无关，未找到可支持回答的视觉证据。"
    return (
        "manual_visual_not_found",
        "图片可能是有效的车辆或手册内容，但当前手册未找到通过二次验证的可靠视觉证据。",
    )


def _no_visual_candidate_failure_message(vision: dict) -> tuple[str, str]:
    """Distinguish obvious upload problems before candidate verification."""

    description = " ".join(
        str(value or "")
        for value in (
            vision.get("visual_description"),
            vision.get("ocr_text"),
            " ".join(str(item) for item in (vision.get("visual_keywords") or [])),
        )
    ).lower()
    if any(token in description for token in ("模糊", "看不清", "无法辨认", "blur", "illegible")):
        return (
            "image_too_blurry",
            "图片过于模糊，两路视觉召回无法建立可靠候选，请重新上传更清晰的图片。",
        )
    unrelated_tokens = (
        "与车辆无关",
        "与汽车无关",
        "与车主手册无关",
        "非车辆",
        "房屋",
        "风景",
        "食物",
        "动物",
        "house",
        "landscape",
    )
    if vision.get("image_type") == "unknown" and any(
        token in description for token in unrelated_tokens
    ):
        return (
            "image_unrelated",
            "图片内容明显与车辆或当前车主手册无关，未找到可支持回答的视觉证据。",
        )
    return (
        "manual_visual_not_found",
        "图片可能是有效的车辆或手册内容，但两路视觉召回未在当前手册中找到可靠匹配。",
    )


def _insufficient_image_response(
    vision: dict,
    *,
    status: str,
    reason: str,
    visual_match: dict | None = None,
) -> dict:
    return {
        "recognized_object": "unknown",
        "image_type": vision.get("image_type"),
        "is_target_icon": vision.get("is_target_icon"),
        "classification_confidence": vision.get("classification_confidence"),
        "classification_reason": vision.get("classification_reason"),
        "visual_description": vision.get("visual_description"),
        "visual_keywords": vision.get("visual_keywords"),
        "candidate_names": vision.get("candidate_names"),
        "normalized_query": vision.get("normalized_query"),
        "risk_level": "unknown",
        "confidence": vision.get("confidence", 0.0),
        "vision_error": vision.get("vision_error"),
        "visual_match": visual_match,
        "answer": f"当前手册未找到可靠视觉证据，无法给出确定结论。\n\n{reason}",
        "suggestion": "请确认手册/车型是否正确；如果图片模糊，请重新上传清晰图片。",
        "evidence": [],
        "visual_evidence": None,
        "text_evidence": [],
        "evidence_status": status,
    }


def _build_non_icon_refusal(vision: dict) -> str:
    """Legacy alias with corrected, non-icon-agnostic wording."""

    return _insufficient_image_response(
        vision,
        status="manual_visual_not_found",
        reason="当前手册没有通过验证的视觉对象。",
    )["answer"]


def _build_low_confidence_refusal(vision: dict, visual_match: dict | None, evidence: list[dict]) -> str:
    del evidence
    reason = (visual_match or {}).get("vlm_reason") or "候选未通过可靠的图片二次验证。"
    return _insufficient_image_response(
        vision,
        status="manual_visual_not_found",
        reason=str(reason),
        visual_match=visual_match,
    )["answer"]


def _visual_title(item: dict) -> str | None:
    return (
        canonical_icon_name_for_item(item)
        or item.get("object_name")
        or item.get("section_title")
        or item.get("verified_title")
    )


def _icon_name_from_evidence(item: dict) -> str | None:
    """Compatibility alias for older icon evaluation code."""

    return _visual_title(item)


def _merge_match_methods(left: str | None, right: str | None) -> str | None:
    methods: list[str] = []
    for value in (left, right):
        for method in str(value or "").split("+"):
            if method and method not in methods:
                methods.append(method)
    return "+".join(methods) if methods else None


def _fuse_visual_evidence(image_embedding_evidence: list[dict], opencv_evidence: list[dict]) -> list[dict]:
    """Historical two-route icon-evaluation compatibility helper."""

    fused = rrf_fuse(
        [image_embedding_evidence, opencv_evidence],
        weights=[1.4, 0.2],
        top_k=10,
    )
    prepared: list[dict] = []
    for item in fused:
        copied = canonicalize_evidence(item)
        copied["_visual_rank_score"] = float(copied.get("rrf_score") or 0)
        prepared.append(copied)
    ranked = sort_visual_evidence(prepared)
    for item in ranked:
        item.pop("_visual_rank_score", None)
    return ranked[:5]


def _canonical_name_from_verification(verification: dict) -> str | None:
    """Compatibility helper returning the generic selected title."""

    return verification.get("selected_title") or verification.get("selected_icon_name")
