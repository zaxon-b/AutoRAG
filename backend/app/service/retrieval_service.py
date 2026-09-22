"""Hybrid retrieval over text leaves followed by recursive parent merging."""

from __future__ import annotations

import re
import time
from typing import Any

from sqlalchemy.orm import Session

from models.manual import ManualChunk, ManualImageOccurrence
from service.auto_merge_service import auto_merge_text_evidence
from service.embedding_service import embed_text
from service.es_index_service import search_keyword_chunks, search_text_vector_chunks
from service.manual_storage_service import to_storage_url
from service.rrf_fusion_service import rrf_fuse
from service.text_cross_encoder_service import TEXT_RERANK_MODEL, rerank_text_candidates


TEXT_CANDIDATE_K = 20
TEXT_BM25_WEIGHT = 0.65
TEXT_DENSE_WEIGHT = 1.0
TEXT_RERANK_CANDIDATE_K = 50
TEXT_MODALITIES = ["text"]

_DOMAIN_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ESC", "电子车身稳定系统", "车身电子稳定系统", "电子稳定控制系统"),
    ("后备箱", "后备厢", "行李箱", "行李箱盖"),
    ("胎压", "标准胎压", "轮胎气压", "轮胎充气压力"),
    ("V2L", "车外放电", "对外放电", "车辆对负载放电"),
    ("紧急启动", "应急启动", "应急启动车辆"),
    ("按钮", "按键", "开关"),
    ("正常启动", "启动车辆", "启动流程", "车辆上电"),
    ("充电方式", "充电模式", "车辆充电"),
)


def expand_domain_query(query: str) -> str:
    normalized = query.casefold()
    additions: list[str] = []
    for group in _DOMAIN_SYNONYM_GROUPS:
        if any(alias.casefold() in normalized for alias in group):
            additions.extend(alias for alias in group if alias.casefold() not in normalized and alias not in additions)
    return " ".join([query.strip(), *additions]).strip()


def retrieve_text_pipeline(
    db: Session,
    manual_id: str,
    query: str,
    top_k: int = 5,
    candidate_k: int = TEXT_CANDIDATE_K,
    final_route: str = "cross_encoder",
    visual_anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if final_route not in {"rrf", "cross_encoder"}:
        raise ValueError(f"unsupported text final route: {final_route}")
    expanded_query = expand_domain_query(query)
    vector = embed_text(expanded_query)
    keyword = search_keyword_chunks(manual_id, expanded_query, top_k=candidate_k)
    dense = search_text_vector_chunks(manual_id, vector, top_k=candidate_k)
    fused = rrf_fuse(
        [keyword, dense],
        weights=[TEXT_BM25_WEIGHT, TEXT_DENSE_WEIGHT],
        top_k=min(TEXT_RERANK_CANDIDATE_K, candidate_k * 2),
    )
    anchor_candidates = _same_page_text_candidates(db, manual_id, visual_anchor)
    candidate_pool = _deduplicate_candidates([*fused, *anchor_candidates, *keyword, *dense])

    cross_encoder_error: str | None = None
    cross_encoder_elapsed_seconds = 0.0
    if final_route == "cross_encoder":
        started = time.monotonic()
        try:
            reranked = rerank_text_candidates(query, candidate_pool)
        except Exception as error:
            cross_encoder_error = f"{type(error).__name__}: {error}"
            reranked = [dict(item) for item in fused]
        cross_encoder_elapsed_seconds = round(time.monotonic() - started, 3)
    else:
        reranked = [dict(item) for item in fused]

    merge_input = [
        _format_evidence(
            item,
            score=item.get("cross_encoder_score")
            if item.get("cross_encoder_score") is not None
            else item.get("rrf_score") or item.get("score"),
        )
        for item in reranked[: max(top_k * 4, 12)]
    ]
    auto_merged = auto_merge_text_evidence(db, manual_id, merge_input)
    evidence = auto_merged[:top_k]
    return {
        "query": query,
        "expanded_query": expanded_query,
        "keyword": keyword,
        "dense": dense,
        "fused": fused,
        "visual_anchor_candidates": anchor_candidates,
        "candidate_pool_size": len(candidate_pool),
        "rrf": fused,
        "cross_encoder": reranked,
        "reranked": reranked,
        "auto_merged": auto_merged,
        "cross_encoder_model": TEXT_RERANK_MODEL if final_route == "cross_encoder" else None,
        "cross_encoder_error": cross_encoder_error,
        "cross_encoder_elapsed_seconds": cross_encoder_elapsed_seconds,
        "final_route": final_route,
        "evidence": evidence,
    }


def retrieve_evidence(db: Session, manual_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    try:
        return retrieve_text_pipeline(db, manual_id, query, top_k=top_k)["evidence"]
    except Exception:
        fallback = _fallback_pg_search(db, manual_id, query, max(top_k * 4, 12))
        return auto_merge_text_evidence(db, manual_id, fallback)[:top_k]


def retrieve_visual_anchor_text(
    db: Session,
    manual_id: str,
    query: str,
    visual_anchor: dict[str, Any],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    result = retrieve_text_pipeline(
        db,
        manual_id,
        query,
        top_k=min(top_k, 4),
        candidate_k=20,
        final_route="cross_encoder",
        visual_anchor=visual_anchor,
    )
    return [{**item, "evidence_role": "text"} for item in result["evidence"][:4]]


def retrieve_text_support(
    db: Session,
    manual_id: str,
    query: str,
    top_k: int = 4,
    anchor_pages: set[int] | None = None,
) -> list[dict[str, Any]]:
    page = min(anchor_pages) if anchor_pages else None
    anchor = {"page": page} if page else None
    return retrieve_text_pipeline(db, manual_id, query, top_k=top_k, visual_anchor=anchor)["evidence"]


def _same_page_text_candidates(
    db: Session,
    manual_id: str,
    visual_anchor: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not visual_anchor:
        return []
    pages = _visual_anchor_pages(db, manual_id, visual_anchor)
    if not pages:
        return []
    rows = (
        db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.page_number.in_(pages),
            ManualChunk.modality == "text",
            ManualChunk.is_leaf.is_(True),
        )
        .order_by(ManualChunk.id)
        .all()
    )
    return [_row_to_candidate(row, "visual_anchor_same_page") for row in rows]


def _visual_anchor_pages(
    db: Session,
    manual_id: str,
    visual_anchor: dict[str, Any],
) -> list[int]:
    """Use every occurrence page for a persisted icon group.

    The group establishes visual identity, while each occurrence keeps its own
    section and content.  Adding all occurrence pages prevents the representative
    image from hiding a different manual context for the same icon.
    """

    page = int(visual_anchor.get("page") or visual_anchor.get("page_number") or 0)
    pages = [page] if page > 0 else []
    group_id = str(visual_anchor.get("icon_group_id") or "")
    if visual_anchor.get("modality") == "icon" and group_id:
        group_pages = (
            db.query(ManualImageOccurrence.page_number)
            .filter(
                ManualImageOccurrence.manual_id == manual_id,
                ManualImageOccurrence.icon_group_id == group_id,
            )
            .distinct()
            .all()
        )
        pages.extend(int(value) for (value,) in group_pages if value)
    return list(dict.fromkeys(pages))


def _fallback_pg_search(db: Session, manual_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    terms = [term for term in re.split(r"\s+", query) if term]
    rows = (
        db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality == "text",
            ManualChunk.is_leaf.is_(True),
        )
        .all()
    )
    scored: list[tuple[float, ManualChunk]] = []
    for row in rows:
        title = row.section_title or ""
        body = row.content_text or ""
        haystack = f"{title}\n{body}"
        score = sum(4.0 if term in title else 1.0 for term in terms if term in haystack)
        if query and query in haystack:
            score += 3.0
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_format_evidence(_row_to_candidate(row, "pg_text_leaf_fallback"), score) for score, row in scored[:top_k]]


def _row_to_candidate(row: ManualChunk, method: str) -> dict[str, Any]:
    return {
        "chunk_id": row.chunk_id,
        "manual_id": row.manual_id,
        "page": row.page_number,
        "modality": row.modality,
        "section_title": row.section_title,
        "content_text": row.content_text,
        "object_name": row.object_name,
        "parent_chunk_id": row.parent_chunk_id,
        "is_leaf": row.is_leaf,
        "token_count": row.token_count,
        "risk_level": row.risk_level,
        "image_path": row.image_path,
        "thumbnail_path": row.thumbnail_path,
        "icon_group_id": row.icon_group_id,
        "score": row.score,
        "match_method": method,
    }


def _format_evidence(chunk: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id") or chunk.get("es_doc_id") or "",
        "page": int(chunk.get("page") or chunk.get("page_number") or 0),
        "modality": "text",
        "section_title": chunk.get("section_title"),
        "content_text": chunk.get("content_text"),
        "object_name": None,
        "parent_chunk_id": chunk.get("parent_chunk_id"),
        "is_leaf": chunk.get("is_leaf"),
        "token_count": chunk.get("token_count"),
        "risk_level": chunk.get("risk_level"),
        "image_path": to_storage_url(chunk.get("image_path")),
        "thumbnail_path": to_storage_url(chunk.get("thumbnail_path")),
        "icon_group_id": None,
        "score": score if score is not None else chunk.get("score"),
        "match_method": chunk.get("match_method"),
        "evidence_role": "text",
    }


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for item in candidates:
        if item.get("modality") != "text" or item.get("is_leaf") is not True:
            continue
        chunk_id = str(item.get("chunk_id") or "")
        fingerprint = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(item.get("content_text") or "").casefold())[:800]
        if not chunk_id or chunk_id in seen_ids or (fingerprint and fingerprint in seen_content):
            continue
        result.append(dict(item))
        seen_ids.add(chunk_id)
        if fingerprint:
            seen_content.add(fingerprint)
    return result
