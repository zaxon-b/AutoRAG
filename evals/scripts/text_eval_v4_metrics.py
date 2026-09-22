"""Metrics for the claim/leaf text evaluation protocol introduced in v4."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


DEFAULT_CUTOFFS = (1, 3, 5)


def unique(values: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def claim_support_ids(claim: dict[str, Any], *, include_parents: bool) -> set[str]:
    ids = {
        str(item.get("chunk_id"))
        for item in claim.get("supporting_leaf_chunks") or []
        if item.get("chunk_id")
    }
    if include_parents:
        ids.update(
            str(item.get("chunk_id"))
            for item in claim.get("supporting_auto_merge_parents") or []
            if item.get("chunk_id")
        )
    return ids


def leaf_equivalence_groups(record: dict[str, Any]) -> list[set[str]]:
    """Return distinct OR-groups of independently sufficient leaf chunks.

    Claims backed by the same leaf alternatives collapse into one group. This
    prevents two facts in one chunk from pretending that two chunks were needed,
    while equivalent chunks for one fact remain alternatives rather than two
    mandatory gold items.
    """

    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for claim in record.get("required_claims") or []:
        group = frozenset(claim_support_ids(claim, include_parents=False))
        if group and group not in seen:
            seen.add(group)
            groups.append(set(group))
    return groups


def case_leaf_group_metrics(
    record: dict[str, Any],
    candidate_ids: list[str],
    cutoff: int,
) -> dict[str, float]:
    """Score ranked leaf chunks against OR-equivalent gold leaf groups."""

    gold_groups = leaf_equivalence_groups(record)
    if not gold_groups:
        return {}
    candidates = unique(candidate_ids)[:cutoff]
    relevant_ids = set().union(*gold_groups)
    matched_groups = sum(bool(set(candidates) & group) for group in gold_groups)
    relevant_count = sum(chunk_id in relevant_ids for chunk_id in candidates)
    precision = relevant_count / cutoff if cutoff else 0.0
    recall = matched_groups / len(gold_groups)
    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(candidates, start=1)
            if chunk_id in relevant_ids
        ),
        None,
    )
    return {
        "hit": float(matched_groups > 0),
        "complete_hit": float(matched_groups == len(gold_groups)),
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "gold_groups": float(len(gold_groups)),
        "matched_groups": float(matched_groups),
    }


def case_claim_metrics(
    record: dict[str, Any],
    candidate_ids: list[str],
    cutoff: int,
    *,
    include_parents: bool = False,
) -> dict[str, float]:
    """Score one ranked list without mixing chunk precision and claim recall."""

    claims = list(record.get("required_claims") or [])
    if not claims:
        return {}
    candidates = unique(candidate_ids)[:cutoff]
    claim_gold = [
        claim_support_ids(claim, include_parents=include_parents)
        for claim in claims
    ]
    relevant_leaf_ids = {
        str(item.get("chunk_id"))
        for claim in claims
        for item in claim.get("supporting_leaf_chunks") or []
        if item.get("chunk_id")
    }
    acceptable_ids = set().union(*claim_gold)
    recalled = [bool(set(candidates) & gold) for gold in claim_gold]
    recalled_count = sum(recalled)
    relevant_count = sum(chunk_id in relevant_leaf_ids for chunk_id in candidates)
    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(candidates, start=1)
            if chunk_id in acceptable_ids
        ),
        None,
    )

    covered: set[int] = set()
    redundant_relevant = 0
    for chunk_id in candidates:
        hit_claims = {
            index for index, gold in enumerate(claim_gold) if chunk_id in gold
        }
        if hit_claims and hit_claims.issubset(covered):
            redundant_relevant += 1
        covered.update(hit_claims)

    return {
        "any_claim_hit": float(recalled_count > 0),
        "complete_claim_hit": float(recalled_count == len(claims)),
        "relevant_leaf_precision": relevant_count / cutoff if cutoff else 0.0,
        "macro_claim_recall": recalled_count / len(claims),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "redundant_relevant_chunks": float(redundant_relevant),
        "gold_claims": float(len(claims)),
        "recalled_claims": float(recalled_count),
    }


def aggregate_route_cutoffs(
    records: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, dict[str, dict[str, float]]]:
    record_by_id = {record["id"]: record for record in records}
    collected: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    leaf_collected: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in result_rows:
        record = record_by_id.get(row.get("id"))
        if record is None or record.get("should_refuse"):
            continue
        for route_name, route in (row.get("routes") or {}).items():
            candidate_ids = list(route.get("top_chunk_ids") or [])
            for cutoff in cutoffs:
                collected[route_name][cutoff].append(
                    case_claim_metrics(record, candidate_ids, cutoff)
                )
                leaf_collected[route_name][cutoff].append(
                    case_leaf_group_metrics(record, candidate_ids, cutoff)
                )

    output: dict[str, dict[str, dict[str, float]]] = {}
    for route_name, cutoff_map in collected.items():
        output[route_name] = {}
        for cutoff, cases in cutoff_map.items():
            leaf_cases = leaf_collected[route_name][cutoff]
            total_leaf_groups = sum(
                case["gold_groups"] for case in leaf_cases
            )
            matched_leaf_groups = sum(
                case["matched_groups"] for case in leaf_cases
            )
            total_claims = sum(case["gold_claims"] for case in cases)
            recalled_claims = sum(case["recalled_claims"] for case in cases)
            output[route_name][f"at_{cutoff}"] = {
                "count": float(len(cases)),
                "leaf_hit": _mean(case["hit"] for case in leaf_cases),
                "leaf_complete_hit": _mean(
                    case["complete_hit"] for case in leaf_cases
                ),
                "leaf_precision": _mean(
                    case["precision"] for case in leaf_cases
                ),
                "macro_leaf_recall": _mean(
                    case["recall"] for case in leaf_cases
                ),
                "micro_leaf_recall": (
                    matched_leaf_groups / total_leaf_groups
                    if total_leaf_groups
                    else 0.0
                ),
                "leaf_f1": _mean(case["f1"] for case in leaf_cases),
                "leaf_reciprocal_rank": _mean(
                    case["reciprocal_rank"] for case in leaf_cases
                ),
                "any_claim_hit": _mean(case["any_claim_hit"] for case in cases),
                "complete_claim_hit": _mean(
                    case["complete_claim_hit"] for case in cases
                ),
                "relevant_leaf_precision": _mean(
                    case["relevant_leaf_precision"] for case in cases
                ),
                "macro_claim_recall": _mean(
                    case["macro_claim_recall"] for case in cases
                ),
                "micro_claim_recall": (
                    recalled_claims / total_claims if total_claims else 0.0
                ),
                "reciprocal_rank": _mean(
                    case["reciprocal_rank"] for case in cases
                ),
                "mean_redundant_relevant_chunks": _mean(
                    case["redundant_relevant_chunks"] for case in cases
                ),
            }
    return output


def aggregate_auto_merge_context(
    records: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    chunk_metadata: dict[str, dict[str, Any]],
    *,
    top_k: int,
    final_route: str,
) -> dict[str, Any]:
    record_by_id = {record["id"]: record for record in records}
    comparisons: list[dict[str, Any]] = []
    for row in result_rows:
        record = record_by_id.get(row.get("id"))
        if record is None or record.get("should_refuse"):
            continue
        pre_ids = list(
            ((row.get("routes") or {}).get(final_route) or {}).get(
                "top_chunk_ids"
            )
            or []
        )[:top_k]
        post_ids = list(row.get("evidence_chunk_ids") or [])[:top_k]
        pre_metrics = case_claim_metrics(
            record,
            pre_ids,
            top_k,
            include_parents=False,
        )
        post_metrics = case_claim_metrics(
            record,
            post_ids,
            top_k,
            include_parents=True,
        )
        pre_tokens = sum(
            int((chunk_metadata.get(chunk_id) or {}).get("token_count") or 0)
            for chunk_id in pre_ids
        )
        post_tokens = sum(
            int((chunk_metadata.get(chunk_id) or {}).get("token_count") or 0)
            for chunk_id in post_ids
        )
        parent_ids = [
            chunk_id
            for chunk_id in post_ids
            if (chunk_metadata.get(chunk_id) or {}).get("is_leaf") is False
        ]
        comparisons.append(
            {
                "id": record["id"],
                "pre_chunk_ids": pre_ids,
                "post_chunk_ids": post_ids,
                "pre": pre_metrics,
                "post": post_metrics,
                "pre_tokens": pre_tokens,
                "post_tokens": post_tokens,
                "parent_chunk_ids": parent_ids,
                "coverage_delta": (
                    post_metrics.get("macro_claim_recall", 0.0)
                    - pre_metrics.get("macro_claim_recall", 0.0)
                ),
            }
        )

    return {
        "count": len(comparisons),
        "pre_auto_merge": _aggregate_context_side(comparisons, "pre", "pre_tokens"),
        "post_auto_merge": _aggregate_context_side(
            comparisons,
            "post",
            "post_tokens",
        ),
        "merge_trigger_rate": _mean(
            float(bool(item["parent_chunk_ids"])) for item in comparisons
        ),
        "coverage_improved_count": sum(
            item["coverage_delta"] > 0 for item in comparisons
        ),
        "coverage_unchanged_count": sum(
            item["coverage_delta"] == 0 for item in comparisons
        ),
        "coverage_reduced_count": sum(
            item["coverage_delta"] < 0 for item in comparisons
        ),
        "cases": comparisons,
    }


def _aggregate_context_side(
    comparisons: list[dict[str, Any]],
    metrics_field: str,
    tokens_field: str,
) -> dict[str, float]:
    cases = [item[metrics_field] for item in comparisons]
    total_claims = sum(case.get("gold_claims", 0.0) for case in cases)
    recalled_claims = sum(case.get("recalled_claims", 0.0) for case in cases)
    return {
        "any_claim_hit": _mean(case.get("any_claim_hit", 0.0) for case in cases),
        "complete_claim_hit": _mean(
            case.get("complete_claim_hit", 0.0) for case in cases
        ),
        "macro_claim_recall": _mean(
            case.get("macro_claim_recall", 0.0) for case in cases
        ),
        "micro_claim_recall": (
            recalled_claims / total_claims if total_claims else 0.0
        ),
        "mean_tokens": _mean(float(item[tokens_field]) for item in comparisons),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
