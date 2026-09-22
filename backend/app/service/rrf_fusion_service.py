from collections.abc import Iterable
from typing import Any


MERGED_FIELDS = (
    "section_title",
    "content_text",
    "object_name",
    "parent_chunk_id",
    "is_leaf",
    "token_count",
    "icon_group_id",
    "risk_level",
    "image_path",
    "thumbnail_path",
    "modality",
    "page",
)


def rrf_fuse(
    ranked_lists: Iterable[list[dict[str, Any]]],
    *,
    weights: list[float] | None = None,
    rank_constant: int = 60,
    id_key: str = "chunk_id",
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse ranked lists without comparing incompatible raw score scales."""
    fused: dict[str, dict[str, Any]] = {}
    route_weights = weights or []

    for route_index, items in enumerate(ranked_lists):
        weight = route_weights[route_index] if route_index < len(route_weights) else 1.0
        for rank, item in enumerate(items, start=1):
            item_id = str(item.get(id_key) or "")
            if not item_id:
                continue
            contribution = weight / (rank_constant + rank)
            existing = fused.get(item_id)
            if existing is None:
                copied = dict(item)
                copied["_rrf_score"] = contribution
                copied["_rrf_routes"] = [_route_name(item, route_index)]
                fused[item_id] = copied
                continue

            existing["_rrf_score"] = float(existing.get("_rrf_score") or 0) + contribution
            existing["_rrf_routes"].append(_route_name(item, route_index))
            existing["score"] = max(
                float(existing.get("score") or 0),
                float(item.get("score") or 0),
            )
            existing["match_method"] = _merge_methods(
                existing.get("match_method"),
                item.get("match_method"),
            )
            for key in MERGED_FIELDS:
                if existing.get(key) is None and item.get(key) is not None:
                    existing[key] = item[key]

    ranked = sorted(
        fused.values(),
        key=lambda item: float(item.get("_rrf_score") or 0),
        reverse=True,
    )
    for item in ranked:
        routes = item.pop("_rrf_routes", [])
        item["rrf_score"] = round(float(item.pop("_rrf_score", 0.0)), 6)
        if routes:
            item["match_method"] = _merge_methods(
                item.get("match_method"),
                "+".join(routes),
            )
    return ranked[:top_k]


def _route_name(item: dict[str, Any], route_index: int) -> str:
    return str(item.get("match_method") or "").strip() or f"route_{route_index + 1}"


def _merge_methods(left: str | None, right: str | None) -> str | None:
    methods: list[str] = []
    for value in (left, right):
        for method in str(value or "").split("+"):
            if method and method not in methods:
                methods.append(method)
    return "+".join(methods) if methods else None
