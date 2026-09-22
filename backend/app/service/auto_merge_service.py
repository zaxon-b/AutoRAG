"""Recursively replace sufficiently covered text children with their parent chunk."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from models.manual import ManualChunk


MIN_MATCHED_CHILDREN = int(os.getenv("AUTO_MERGE_MIN_MATCHED_CHILDREN", "2"))
MIN_CHILD_RATIO = float(os.getenv("AUTO_MERGE_MIN_CHILD_RATIO", "0.5"))
MAX_PARENT_TOKENS = int(os.getenv("AUTO_MERGE_MAX_PARENT_TOKENS", "1800"))
MAX_EVIDENCE_TOKENS = int(os.getenv("AUTO_MERGE_MAX_EVIDENCE_TOKENS", "5000"))


def auto_merge_text_evidence(
    db: Session,
    manual_id: str,
    candidates: list[dict[str, Any]],
    *,
    min_matched_children: int = MIN_MATCHED_CHILDREN,
    min_child_ratio: float = MIN_CHILD_RATIO,
    max_parent_tokens: int = MAX_PARENT_TOKENS,
    max_total_tokens: int = MAX_EVIDENCE_TOKENS,
) -> list[dict[str, Any]]:
    """Apply threshold-based bottom-up merging and enforce the final token budget."""

    current = _unique_text_candidates(candidates)
    while current:
        ids = [str(item.get("chunk_id") or "") for item in current if item.get("chunk_id")]
        rows = (
            db.query(ManualChunk)
            .filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.modality == "text",
                ManualChunk.chunk_id.in_(ids),
            )
            .all()
        )
        row_by_id = {row.chunk_id: row for row in rows}
        matched_by_parent: dict[str, list[str]] = defaultdict(list)
        for chunk_id in ids:
            row = row_by_id.get(chunk_id)
            if row and row.parent_chunk_id:
                matched_by_parent[row.parent_chunk_id].append(chunk_id)
        if not matched_by_parent:
            break

        parent_ids = list(matched_by_parent)
        parents = (
            db.query(ManualChunk)
            .filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.modality == "text",
                ManualChunk.is_leaf.is_(False),
                ManualChunk.chunk_id.in_(parent_ids),
            )
            .all()
        )
        all_children: dict[str, int] = defaultdict(int)
        for parent_id, _ in (
            db.query(ManualChunk.parent_chunk_id, ManualChunk.chunk_id)
            .filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.modality == "text",
                ManualChunk.parent_chunk_id.in_(parent_ids),
            )
            .all()
        ):
            if parent_id:
                all_children[str(parent_id)] += 1

        merged_any = False
        for parent in sorted(parents, key=lambda row: min(_index_of(current, child) for child in matched_by_parent[row.chunk_id])):
            matched_ids = list(dict.fromkeys(matched_by_parent[parent.chunk_id]))
            total_children = all_children.get(parent.chunk_id, 0)
            if len(matched_ids) < min_matched_children or total_children <= 0:
                continue
            if len(matched_ids) / total_children < min_child_ratio:
                continue
            parent_tokens = int(parent.token_count or 0)
            if parent_tokens <= 0 or parent_tokens > max_parent_tokens:
                continue

            matched_set = set(matched_ids)
            matched_items = [item for item in current if item.get("chunk_id") in matched_set]
            insertion_index = min(index for index, item in enumerate(current) if item.get("chunk_id") in matched_set)
            best_score = max((float(item.get("score") or 0.0) for item in matched_items), default=0.0)
            methods = [str(item.get("match_method") or "") for item in matched_items]
            parent_item = _row_to_evidence(parent, best_score, _merge_methods(*methods, "auto_merge"))
            current = [item for item in current if item.get("chunk_id") not in matched_set]
            current.insert(insertion_index, parent_item)
            merged_any = True
        if not merged_any:
            break

    return _fit_token_budget(current, max_total_tokens)


def _unique_text_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("modality") != "text":
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        result.append(dict(item))
    return result


def _row_to_evidence(row: ManualChunk, score: float, match_method: str) -> dict[str, Any]:
    return {
        "chunk_id": row.chunk_id,
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
        "score": score,
        "match_method": match_method,
        "evidence_role": "text",
    }


def _fit_token_budget(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    total = 0
    for item in items:
        size = _item_tokens(item)
        if result and total + size > limit:
            continue
        if not result and size > limit:
            # Keep the best evidence instead of returning nothing; prompt construction
            # still applies its own hard truncation.
            result.append(item)
            break
        result.append(item)
        total += size
    return result


def _item_tokens(item: dict[str, Any]) -> int:
    value = item.get("token_count")
    if isinstance(value, int) and value >= 0:
        return value
    from service.hierarchical_chunk_service import count_tokens

    return count_tokens(str(item.get("content_text") or ""))


def _index_of(items: list[dict[str, Any]], chunk_id: str) -> int:
    return next(index for index, item in enumerate(items) if item.get("chunk_id") == chunk_id)


def _merge_methods(*methods: str) -> str:
    parts: list[str] = []
    for method in methods:
        for part in str(method or "").split("+"):
            part = part.strip()
            if part and part not in parts:
                parts.append(part)
    return "+".join(parts)
