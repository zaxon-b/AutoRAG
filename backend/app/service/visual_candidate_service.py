"""Deduplicate and fuse visual chunks for the production image-QA path.

The persisted visual modalities are exactly ``icon``, ``operation_figure``
and ``page_snapshot``.

Every retrieval route is deduplicated independently. Icon occurrences use
``icon_group_id``; operation figures and page snapshots keep their occurrence
``chunk_id`` because the same pixels may have different manual context.
Production fusion admits the union of the image-vector and multimodal routes,
then combines their ranks with RRF. OpenCV is retained only as an offline
ablation and cannot introduce a production candidate.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

from sqlalchemy.orm import Session

from service.icon_catalog_service import (
    _db_bind_key,
    _group_metadata,
    _manual_icon_groups,
    _merge_methods,
    _representative_score,
    annotate_icon_groups,
    canonicalize_evidence,
)
from service.manual_storage_service import storage_root
from service.rrf_fusion_service import rrf_fuse


# These values come from the actual demo_manual DB/ES index, not from an
# aspirational schema.  Extend the set only after ingestion creates that
# modality and both indexes are verified to contain it.
VISUAL_MODALITIES = frozenset({"icon", "operation_figure", "page_snapshot"})
ICON_MODALITIES = frozenset({"icon"})

# Compatibility aliases for historical evaluation scripts.  Production code
# uses the generic names above.
VISUAL_RETRIEVAL_MODALITIES = VISUAL_MODALITIES
TARGET_ICON_MODALITIES = ICON_MODALITIES

RRF_RANK_CONSTANT = 60
# Elasticsearch image-vector scores are normalised to 0-100 by
# visual_similarity_service.  A nearest-neighbour index always returns a
# nearest item, even for an unrelated upload, so the primary pool needs a
# conservative floor before secondary routes or the VLM see it.  The audited
# 60-image set has a minimum Top1 score of 73.9; 50 leaves a wide margin while
# rejecting the unrelated smoke image (28.8).
MIN_IMAGE_EMBEDDING_CANDIDATE_SCORE = float(
    os.getenv("MIN_IMAGE_EMBEDDING_CANDIDATE_SCORE", "50")
)


def deduplicate_visual_candidates(
    db: Session,
    manual_id: str,
    evidence: list[dict],
    *,
    max_items: int | None = None,
) -> list[dict]:
    """Return one ranked candidate per visual object.

    Deduplication identity follows the production priority requested by the
    visual-QA design:

    * icon chunks: ``icon_group_id`` when available;
    * operation figures and page snapshots: ``chunk_id``.

    Icon routes keep the matched prototype image for VLM verification and add
    the group's canonical metadata. Non-icon chunks remain themselves. An
    operation figure is never rewritten into an icon merely because both
    happened to be linked by catalog data.
    """

    annotated = annotate_icon_groups(db, manual_id, evidence)
    icon_groups = _manual_icon_groups(_db_bind_key(db), manual_id)
    buckets: dict[str, dict] = {}

    for rank, raw_item in enumerate(annotated, start=1):
        item = canonicalize_evidence(raw_item)
        modality = str(item.get("modality") or "")
        if modality not in VISUAL_MODALITIES:
            continue

        candidate = dict(item)
        if modality in ICON_MODALITIES:
            group_id = str(item.get("icon_group_id") or "")
            group = icon_groups.get(group_id) if group_id else None
            if group is not None:
                # ES may return any of the group's 1-N prototypes. Preserve that
                # prototype image for VLM verification while attaching the
                # single persisted group chunk's canonical metadata.
                candidate.update(_group_metadata(group, candidate))
                candidate["icon_group_id"] = group_id

        fingerprint = _perceptual_fingerprint(
            candidate.get("image_path") or candidate.get("thumbnail_path")
        )
        key = _candidate_key(candidate, fingerprint=fingerprint)
        existing = buckets.get(key)
        if existing is None:
            candidate["_candidate_group_key"] = key
            candidate["_route_rank"] = rank
            candidate["image_fingerprint"] = fingerprint
            candidate["score"] = max(
                float(candidate.get("score") or 0),
                float(item.get("score") or 0),
            )
            candidate["matched_chunk_ids"] = _unique_values(
                [candidate.get("chunk_id"), item.get("chunk_id")]
            )
            candidate["matched_modalities"] = _unique_values(
                [candidate.get("modality"), item.get("modality")]
            )
            buckets[key] = candidate
            continue

        buckets[key] = _merge_candidate(existing, item)

    ranked = sorted(buckets.values(), key=lambda item: int(item.get("_route_rank") or 10**9))
    for position, item in enumerate(ranked, start=1):
        item["group_rank"] = position
    return ranked[:max_items] if max_items else ranked


def group_visual_candidates(
    db: Session,
    manual_id: str,
    evidence: list[dict],
    *,
    max_items: int | None = None,
) -> list[dict]:
    """Compatibility wrapper for the old icon-oriented function name."""

    return deduplicate_visual_candidates(db, manual_id, evidence, max_items=max_items)


def fuse_group_routes(
    grouped_routes: Iterable[list[dict]],
    *,
    weights: list[float],
    top_k: int,
) -> list[dict]:
    """Keep ordinary RRF available for evaluation ablations only."""

    return rrf_fuse(
        grouped_routes,
        weights=weights,
        rank_constant=RRF_RANK_CONSTANT,
        id_key="_candidate_group_key",
        top_k=top_k,
    )


def gated_group_fusion(
    image_groups: list[dict],
    opencv_groups: list[dict],
    multimodal_groups: list[dict],
    *,
    top_k: int = 8,
    image_weight: float = 1.4,
    opencv_weight: float = 0.2,
    multimodal_weight: float = 1.2,
    minimum_image_score: float = MIN_IMAGE_EMBEDDING_CANDIDATE_SCORE,
) -> list[dict]:
    """Apply gated agreement scoring to independently deduplicated routes.

    Only a sufficiently similar image-vector candidate may enter the returned
    pool.  This prevents an unrelated upload from inheriting the forced Top1
    nearest neighbour.  A matching multimodal hit adds the stronger
    consistency bonus; a matching OpenCV hit adds a smaller low-level
    shape/colour bonus.  The score floor is only a rejection prerequisite:
    raw similarity can never establish a verified match without the VLM.
    """

    opencv_ranks = _rank_map(opencv_groups)
    multimodal_ranks = _rank_map(multimodal_groups)
    opencv_by_key = _item_map(opencv_groups)
    multimodal_by_key = _item_map(multimodal_groups)
    fused: list[dict] = []

    for image_rank, item in enumerate(image_groups, start=1):
        primary_image_score = float(item.get("score") or 0)
        if primary_image_score < minimum_image_score:
            continue
        key = str(item.get("_candidate_group_key") or "")
        if not key:
            continue
        score = image_weight / (RRF_RANK_CONSTANT + image_rank)
        routes = ["image_embedding"]
        copied = dict(item)
        copied["image_embedding_score"] = primary_image_score

        multimodal_rank = multimodal_ranks.get(key)
        if multimodal_rank is not None:
            score += multimodal_weight / (RRF_RANK_CONSTANT + multimodal_rank)
            copied = _merge_candidate(copied, multimodal_by_key[key])
            routes.append("multimodal_embedding")

        opencv_rank = opencv_ranks.get(key)
        if opencv_rank is not None:
            score += opencv_weight / (RRF_RANK_CONSTANT + opencv_rank)
            copied = _merge_candidate(copied, opencv_by_key[key])
            routes.append("opencv_visual_match")

        copied["rrf_score"] = score
        copied["gated_fusion_score"] = score
        copied["agreement_routes"] = routes
        copied["match_method"] = _merge_methods(copied.get("match_method"), "gated_visual_fusion")
        fused.append(copied)

    fused.sort(key=lambda item: float(item.get("gated_fusion_score") or 0), reverse=True)
    return fused[:top_k]


def union_group_fusion(
    image_groups: list[dict],
    opencv_groups: list[dict],
    multimodal_groups: list[dict],
    *,
    top_k: int = 10,
    image_weight: float = 1.0,
    multimodal_weight: float = 1.0,
    opencv_weight: float = 0.0,
) -> list[dict]:
    """Fuse the image and multimodal candidate union without a hard score gate.

    Either embedding route may introduce a candidate. OpenCV remains auxiliary:
    it can boost an object already present in that union but cannot introduce an
    object by itself. All contributions are rank based so incompatible raw model
    scores are never compared.
    """

    candidate_limit = len(image_groups) + len(multimodal_groups)
    if candidate_limit <= 0:
        return []
    fused = rrf_fuse(
        [image_groups, multimodal_groups],
        weights=[image_weight, multimodal_weight],
        rank_constant=RRF_RANK_CONSTANT,
        id_key="_candidate_group_key",
        top_k=candidate_limit,
    )
    image_ranks = _rank_map(image_groups)
    multimodal_ranks = _rank_map(multimodal_groups)
    opencv_ranks = _rank_map(opencv_groups)
    opencv_by_key = _item_map(opencv_groups)
    for item in fused:
        key = str(item.get("_candidate_group_key") or "")
        routes = []
        if key in image_ranks:
            routes.append("image_embedding")
        if key in multimodal_ranks:
            routes.append("multimodal_embedding")
        opencv_rank = opencv_ranks.get(key)
        if opencv_rank is not None and opencv_weight > 0:
            item["rrf_score"] = float(item.get("rrf_score") or 0) + (
                opencv_weight / (RRF_RANK_CONSTANT + opencv_rank)
            )
            item.update(_merge_candidate(item, opencv_by_key[key]))
            routes.append("opencv_visual_match")
        item["agreement_routes"] = routes
        item["union_fusion_score"] = float(item.get("rrf_score") or 0)
        item["match_method"] = _merge_methods(item.get("match_method"), "union_visual_fusion")
    fused.sort(key=lambda item: float(item.get("union_fusion_score") or 0), reverse=True)
    return fused[:top_k]


def _icon_representative(group: dict | None) -> dict | None:
    if not group:
        return None
    members = [
        dict(item)
        for item in group.get("members") or []
        if item.get("modality") in ICON_MODALITIES
    ]
    if not members:
        return None
    representative_id = str(group.get("representative_chunk_id") or "")
    representative = next(
        (item for item in members if str(item.get("chunk_id") or "") == representative_id),
        max(members, key=_representative_score),
    )
    representative["is_group_representative"] = True
    return representative


def _candidate_key(item: dict, *, fingerprint: str | None = None) -> str:
    del fingerprint  # Fingerprints are diagnostics, never occurrence identity.
    modality = str(item.get("modality") or "")
    if modality in ICON_MODALITIES:
        group_id = str(item.get("icon_group_id") or "")
        if group_id:
            return f"group:{group_id}"
    chunk_id = str(item.get("chunk_id") or "")
    return f"chunk:{chunk_id or id(item)}"


def _perceptual_fingerprint(value: str | None) -> str | None:
    path = _local_image_path(value)
    if path is None or not path.exists():
        return None
    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
            dct = cv2.dct(resized)[:8, :8]
            median = float(np.median(dct.reshape(-1)[1:]))
            bits = "".join("1" if float(value) >= median else "0" for value in dct.reshape(-1))
            return f"phash-{int(bits, 2):016x}"
    except Exception:
        pass
    return f"sha1-{hashlib.sha1(path.read_bytes()).hexdigest()[:16]}"


def _local_image_path(value: str | None) -> Path | None:
    if not value:
        return None
    if value.startswith("/storage/"):
        return storage_root() / value.removeprefix("/storage/")
    path = Path(value)
    return path if path.is_absolute() else storage_root() / path


def _rank_map(items: list[dict]) -> dict[str, int]:
    return {
        str(item.get("_candidate_group_key")): rank
        for rank, item in enumerate(items, start=1)
        if item.get("_candidate_group_key")
    }


def _item_map(items: list[dict]) -> dict[str, dict]:
    return {
        str(item.get("_candidate_group_key")): item
        for item in items
        if item.get("_candidate_group_key")
    }


def _unique_values(values: Iterable) -> list:
    result: list = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _merge_candidate(left: dict, right: dict) -> dict:
    merged = dict(left)
    merged["score"] = max(float(left.get("score") or 0), float(right.get("score") or 0))
    merged["match_method"] = _merge_methods(left.get("match_method"), right.get("match_method"))
    merged["matched_chunk_ids"] = _unique_values(
        [
            *list(left.get("matched_chunk_ids") or []),
            left.get("chunk_id"),
            *list(right.get("matched_chunk_ids") or []),
            right.get("chunk_id"),
        ]
    )
    merged["matched_modalities"] = _unique_values(
        [
            *list(left.get("matched_modalities") or []),
            left.get("modality"),
            *list(right.get("matched_modalities") or []),
            right.get("modality"),
        ]
    )
    return merged
