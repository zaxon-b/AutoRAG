"""Read the persisted icon catalog without duplicating chunk member data."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from models.manual import ManualChunk, ManualIconGroup
from service.icon_grouping_service import build_icon_groups
from service.manual_storage_service import to_storage_url


GROUPABLE_VISUAL_MODALITIES = ["icon"]


def clear_icon_group_cache() -> None:
    _manual_icon_groups.cache_clear()


def persist_icon_groups(
    db: Session,
    manual_id: str,
    image_vectors: dict[str, list[float] | None] | None = None,
) -> int:
    count = build_icon_groups(db, manual_id, image_vectors=image_vectors)
    clear_icon_group_cache()
    return count


def canonical_icon_name_from_text(*values: object, allow_generic: bool = False) -> str | None:
    """Compatibility helper: accept only a literal concise value, never infer a suffix."""

    del allow_generic
    for value in values:
        text = " ".join(str(value or "").split()).strip(" ：:，,。")
        if 2 <= len(text) <= 255:
            return text
    return None


def canonical_icon_name_for_item(item: dict[str, Any]) -> str | None:
    if str(item.get("modality") or "") != "icon":
        return None
    return canonical_icon_name_from_text(item.get("object_name"))


def canonicalize_evidence(item: dict[str, Any]) -> dict[str, Any]:
    copied = dict(item)
    if copied.get("page") is None and copied.get("page_number") is not None:
        copied["page"] = copied.get("page_number")
    copied.pop("page_number", None)
    if copied.get("image_path"):
        copied["image_path"] = to_storage_url(copied["image_path"])
    if copied.get("thumbnail_path"):
        copied["thumbnail_path"] = to_storage_url(copied["thumbnail_path"])
    return copied


def annotate_icon_groups(
    db: Session,
    manual_id: str,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunk_ids = [str(item.get("chunk_id")) for item in evidence if item.get("chunk_id")]
    rows = (
        db.query(ManualChunk)
        .filter(ManualChunk.manual_id == manual_id, ManualChunk.chunk_id.in_(chunk_ids))
        .all()
        if chunk_ids
        else []
    )
    row_by_id = {row.chunk_id: row for row in rows}
    result: list[dict[str, Any]] = []
    for item in evidence:
        copied = canonicalize_evidence(item)
        row = row_by_id.get(str(copied.get("chunk_id") or ""))
        if row and row.modality == "icon":
            copied["icon_group_id"] = row.icon_group_id
            copied["object_name"] = row.object_name
        result.append(copied)
    return result


def expand_with_icon_group_representatives(
    db: Session,
    manual_id: str,
    evidence: list[dict[str, Any]],
    max_items: int = 8,
) -> list[dict[str, Any]]:
    return prefer_icon_group_representative(annotate_icon_groups(db, manual_id, evidence))[:max_items]


def prefer_icon_group_representative(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-ranked candidate for each persisted icon group."""

    result: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_chunks: set[str] = set()
    for item in sort_visual_evidence(evidence):
        copied = canonicalize_evidence(item)
        group_id = str(copied.get("icon_group_id") or "") if copied.get("modality") == "icon" else ""
        chunk_id = str(copied.get("chunk_id") or "")
        if group_id and group_id in seen_groups:
            continue
        if not group_id and chunk_id and chunk_id in seen_chunks:
            continue
        if group_id:
            seen_groups.add(group_id)
        if chunk_id:
            seen_chunks.add(chunk_id)
        result.append(copied)
    return result


def sort_visual_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (canonicalize_evidence(item) for item in evidence),
        key=lambda item: (
            float(item.get("_visual_rank_score") or item.get("gated_fusion_score") or item.get("rrf_score") or item.get("score") or 0.0)
            + visual_rank_bonus(item),
            bool(item.get("object_name")),
        ),
        reverse=True,
    )


def visual_rank_bonus(item: dict[str, Any]) -> float:
    bonus = 0.0
    if item.get("modality") == "icon" and item.get("icon_group_id"):
        bonus += 0.01
    if item.get("vlm_verified"):
        bonus += 0.05
    return bonus


@lru_cache(maxsize=32)
def _manual_icon_groups(_bind_key: str, manual_id: str) -> dict[str, dict[str, Any]]:
    from utils.database import SessionLocal

    db = SessionLocal()
    try:
        groups = db.query(ManualIconGroup).filter(ManualIconGroup.manual_id == manual_id).all()
        chunks = (
            db.query(ManualChunk)
            .filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.modality == "icon",
                ManualChunk.icon_group_id.isnot(None),
            )
            .all()
        )
        members_by_group: dict[str, list[dict[str, Any]]] = {}
        for row in chunks:
            members_by_group.setdefault(str(row.icon_group_id), []).append(_row_to_evidence(row))
        return {
            group.icon_group_id: {
                "icon_group_id": group.icon_group_id,
                "canonical_icon_name": group.canonical_icon_name,
                "representative_chunk_id": group.representative_chunk_id,
                "member_count": group.member_count,
                "members": members_by_group.get(group.icon_group_id, []),
            }
            for group in groups
        }
    finally:
        db.close()


def _row_to_evidence(row: ManualChunk) -> dict[str, Any]:
    return canonicalize_evidence(
        {
            "chunk_id": row.chunk_id,
            "page": row.page_number,
            "modality": row.modality,
            "section_title": row.section_title,
            "content_text": row.content_text,
            "object_name": row.object_name,
            "icon_group_id": row.icon_group_id,
            "risk_level": row.risk_level,
            "bbox": row.bbox,
            "image_path": row.image_path,
            "thumbnail_path": row.thumbnail_path,
        }
    )


def _group_metadata(group: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any]:
    if not group:
        return {}
    return {
        "icon_group_id": group.get("icon_group_id"),
        "object_name": item.get("object_name") or group.get("canonical_icon_name"),
    }


def _representative_score(item: dict[str, Any]) -> float:
    return float(item.get("score") or 0.0) + (0.01 if item.get("object_name") else 0.0)


def _db_bind_key(db: Session) -> str:
    try:
        return str(db.get_bind().url)
    except Exception:
        return "default"


def _merge_methods(left: str | None, right: str | None) -> str | None:
    parts: list[str] = []
    for value in (left, right):
        for part in str(value or "").split("+"):
            part = part.strip()
            if part and part not in parts:
                parts.append(part)
    return "+".join(parts) or None
