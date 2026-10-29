"""Evaluate the owner-manual text and multimodal RAG pipeline.

The evaluator calls the same retrieval, VLM rerank, refusal and answer
generation functions used by the API. It reads the current datasets from the
repository-level ``evals/datasets`` directory and writes run artifacts under
``evals/results/runs``.

Examples (inside the application container)::

    python -m evaluate_owner_manual --mode all
    python -m evaluate_owner_manual --mode text --skip-generation
    python -m evaluate_owner_manual --mode image --limit-image 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from models.manual import ManualChunk
from router.qa_rt import (
    _apply_retrieval_anchor,
    _apply_vlm_verification,
    _build_image_answer_question,
    _build_insufficient_text_answer,
    _build_image_text_query,
    _build_low_confidence_refusal,
    _build_visual_anchor,
    _build_visual_match,
    _is_strong_visual_match,
    _should_refuse_image_answer,
    _verified_visual_anchor,
)
from service.answer_generation_service import complete_answer
from service.retrieval_service import retrieve_text_pipeline, retrieve_visual_anchor_text
from service.visual_candidate_service import (
    deduplicate_visual_candidates,
    gated_group_fusion,
    union_group_fusion,
)
from service.vision_rerank_service import verify_visual_candidates
from service.vision_understanding_service import understand_vehicle_image
from service.visual_similarity_service import (
    retrieve_visual_chunks_by_image_embedding,
    retrieve_visual_chunks_by_multimodal,
    retrieve_visual_chunks_by_opencv,
)
from utils.database import SessionLocal


APP_DIR = Path(__file__).resolve().parent
REPOSITORY_EVAL_DIR = APP_DIR.parent.parent / "evals"
EVAL_DIR = REPOSITORY_EVAL_DIR if REPOSITORY_EVAL_DIR.exists() else APP_DIR / "evals"
DEFAULT_TEXT_DATASET = EVAL_DIR / "datasets" / "owner_manual_text_eval_v4.jsonl"
DEFAULT_IMAGE_DATASET = EVAL_DIR / "datasets" / "owner_manual_image_eval_native_v4.jsonl"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results" / "runs"

REFUSAL_PATTERN = re.compile(
    r"无法.{0,20}(?:确认|回答|判断|识别)"
    r"|无法给出.{0,20}(?:结论|答案)"
    r"|(?:当前)?手册未找到.{0,20}(?:证据|匹配)"
    r"|未检索到.{0,30}(?:正文)?证据"
    r"|(?:当前|现有|提供的)?(?:文档|手册)?(?:证据|信息)(?:明显)?不足"
    r"|手册(?:中|内容)?(?:未提供|没有提供|不包含)"
    r"|不能从.{0,20}(?:确认|得出)"
    r"|没有任何.{0,16}(?:包含|涉及|提及|给出|提供)"
    r"|不应猜测|无法可靠(?:回答|说明)|未通过高置信"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate owner-manual RAG retrieval, answers and refusal behavior.")
    parser.add_argument("--mode", choices=["all", "text", "image"], default="all")
    parser.add_argument("--text-dataset", type=Path, default=DEFAULT_TEXT_DATASET)
    parser.add_argument("--image-dataset", type=Path, default=DEFAULT_IMAGE_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--result-name", default="owner_manual_eval", help="Base name for checkpoints and result files.")
    parser.add_argument("--latest-only", action="store_true", help="Do not keep timestamped result copies.")
    parser.add_argument("--json-only", action="store_true", help="Write machine-readable JSON without a per-run Markdown report.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-text", type=int, default=0, help="0 means all text samples.")
    parser.add_argument("--limit-image", type=int, default=0, help="0 means all image samples.")
    parser.add_argument(
        "--image-timeout-seconds",
        type=int,
        default=180,
        help="Maximum wall time for one image case; 0 disables the evaluator timeout.",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        default=[],
        help="Evaluate only the requested image case ID; repeat for multiple cases.",
    )
    parser.add_argument("--skip-generation", action="store_true", help="Run retrieval/refusal stages without answer generation.")
    parser.add_argument(
        "--skip-first-pass-vlm",
        action="store_true",
        help="Skip only the first-pass image understanding used to build the body query.",
    )
    parser.add_argument(
        "--enable-candidate-vlm",
        action="store_true",
        help="Enable candidate-changing VLM verification; production keeps it disabled.",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help=(
            "Legacy switch: skip both first-pass understanding and candidate "
            "verification. It does not describe the production image-QA path."
        ),
    )
    parser.add_argument(
        "--skip-text-support",
        action="store_true",
        help="Stop image evaluation after VLM anchor selection; useful for an uncontaminated visual-only ablation.",
    )
    parser.add_argument(
        "--image-vector-variant",
        choices=["native", "alpha_white"],
        default="alpha_white",
        help="Production defaults to alpha_white; native is an explicit ablation.",
    )
    parser.add_argument(
        "--visual-fusion-profile",
        choices=[
            "legacy_hard_gate",
            "rrf_equal",
            "rrf_multimodal_weighted",
            "rrf_image_weighted",
            "rrf_equal_opencv",
            "rrf_multimodal_weighted_opencv",
            "rrf_image_weighted_opencv",
        ],
        default="rrf_equal",
        help="Production defaults to equal-weight RRF; other profiles are explicit ablations.",
    )
    parser.add_argument(
        "--enable-opencv-ablation",
        action="store_true",
        help="Run the OpenCV route for offline diagnosis; production does not rank with it.",
    )
    parser.add_argument(
        "--vlm-top-k",
        type=int,
        default=5,
        help="Number of fused candidates shown to the VLM after retrieval ablation selects the cutoff.",
    )
    parser.add_argument(
        "--text-final-route",
        choices=["rrf", "cross_encoder"],
        default="cross_encoder",
        help="Choose the text Top-5 evidence route used for answer generation.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume rows from the checkpoint in results-dir.")
    parser.add_argument("--report-from", type=Path, help="Recompute metrics/report from an existing result JSON without model calls.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return records


class ImageEvaluationTimeout(BaseException):
    pass


@contextmanager
def image_evaluation_timeout(seconds: int):
    """Prevent one stalled external model call from blocking the full run."""

    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame) -> None:
        raise ImageEvaluationTimeout(
            f"image case exceeded evaluator timeout of {seconds} seconds"
        )

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def resolve_image_path(value: str, dataset_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (dataset_path.parent / path).resolve()


def unique(values: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def retrieval_at_k(
    candidate_keys: list[str],
    gold_keys: list[str],
    k: int,
    *,
    evidence_units: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    candidates = unique(candidate_keys)[:k]
    gold = set(unique(gold_keys))
    if not gold:
        return {}
    true_positives = len(set(candidates) & gold)
    # Precision@k uses the requested cutoff as denominator.  A route that
    # returns fewer than k candidates must not receive an artificially higher
    # score merely because its list is short after group deduplication.
    precision = true_positives / k if k > 0 else 0.0
    if evidence_units:
        unit_gold = [set(unique(unit.get("acceptable_chunk_ids") or [])) for unit in evidence_units]
        unit_gold = [keys for keys in unit_gold if keys]
        recalled_units = sum(bool(set(candidates) & keys) for keys in unit_gold)
        recall = recalled_units / len(unit_gold) if unit_gold else 0.0
    else:
        recalled_units = true_positives
        recall = true_positives / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rank = next((index for index, key in enumerate(unique(candidate_keys), start=1) if key in gold), None)
    return {
        f"hit_at_{k}": float(true_positives > 0),
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        f"f1_at_{k}": f1,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "gold_evidence_units": float(len(evidence_units) if evidence_units else len(gold)),
        "recalled_evidence_units": float(recalled_units),
    }


def page_from_chunk_id(chunk_id: str | None) -> str | None:
    match = re.search(r"_p(\d{3})_", str(chunk_id or ""))
    return str(int(match.group(1))) if match else None


def page_from_item(item: dict[str, Any]) -> str | None:
    """Read the persisted page first; legacy chunk IDs are only a fallback."""

    page = item.get("page")
    if page is None:
        page = item.get("page_number")
    if page is not None:
        try:
            return str(int(page))
        except (TypeError, ValueError):
            pass
    return page_from_chunk_id(item.get("chunk_id"))


def route_result(
    items: list[dict[str, Any]],
    gold_keys: list[str],
    k: int,
    *,
    key_for_chunk: dict[str, str] | None = None,
    gold_pages: list[int] | None = None,
    evidence_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chunk_ids = unique(item.get("chunk_id") for item in items)
    if key_for_chunk is None:
        keys = chunk_ids
    else:
        keys = unique(key_for_chunk.get(chunk_id, chunk_id) for chunk_id in chunk_ids)
    result = {
        "top_chunk_ids": chunk_ids[:10],
        "top_keys": keys[:10],
        "metrics": retrieval_at_k(keys, gold_keys, k, evidence_units=evidence_units),
    }
    if gold_pages:
        page_metrics = retrieval_at_k(
            unique(page_from_item(item) for item in items),
            [str(page) for page in gold_pages],
            k,
        )
        result["metrics"].update({f"page_{name}": value for name, value in page_metrics.items()})
    return result


def normalise_text(value: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9.%℃°]+", str(value or ""))).lower()


def ngrams(value: str, n: int = 2) -> set[str]:
    text = normalise_text(value)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def coverage_score(reference: str, candidate: str) -> float:
    reference_normalised = normalise_text(reference)
    candidate_normalised = normalise_text(candidate)
    if not reference_normalised:
        return 1.0
    if reference_normalised in candidate_normalised:
        return 1.0
    reference_grams = ngrams(reference)
    candidate_grams = ngrams(candidate)
    score = len(reference_grams & candidate_grams) / len(reference_grams) if reference_grams else 0.0
    reference_numbers = set(re.findall(r"\d+(?:\.\d+)?", reference_normalised))
    if reference_numbers and not reference_numbers.issubset(set(re.findall(r"\d+(?:\.\d+)?", candidate_normalised))):
        score *= 0.5
    return min(1.0, score)


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(item.get(field) or "")
        for item in evidence
        for field in ("section_title", "object_name", "content_text")
    )


def detect_refusal(answer: str) -> bool:
    # Judge the leading conclusion only.  Scanning the whole answer incorrectly
    # marks a valid direct answer as refusal when a later caveat says that one
    # secondary detail cannot be confirmed.
    cleaned = re.sub(r"^[#*\s]+", "", answer or "")
    first_sentence = re.split(r"[。！？\n]", cleaned, maxsplit=1)[0][:260]
    return bool(REFUSAL_PATTERN.search(first_sentence))


def score_answer(record: dict[str, Any], answer: str, evidence: list[dict[str, Any]], predicted_refuse: bool) -> dict[str, float]:
    if record.get("should_refuse"):
        has_next_step = bool(re.search(r"建议|请提供|请确认|查询|联系|补充", answer or ""))
        return {
            "claim_coverage": float(predicted_refuse),
            "claim_pass_rate": float(predicted_refuse),
            "groundedness_proxy": float(predicted_refuse),
            "execution_score": (float(predicted_refuse) + float(has_next_step)) / 2,
            "answer_quality_proxy": float(predicted_refuse),
        }

    claims = record.get("reference_claims") or [record.get("reference_answer") or ""]
    claim_scores = [coverage_score(str(claim), answer) for claim in claims]
    claim_coverage = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0
    claim_pass_rate = sum(score >= 0.5 for score in claim_scores) / len(claim_scores) if claim_scores else 0.0
    support_source = evidence_text(evidence)
    groundedness = coverage_score(answer, support_source) if answer and support_source else 0.0
    direct_answer = float(bool(answer.strip()) and not predicted_refuse)
    citation = float(bool(re.search(r"第\s*\d+\s*页|证据\s*\d+|\[\d+\]", answer or "")))
    execution = (direct_answer + claim_pass_rate + citation) / 3
    quality = claim_coverage * 0.65 + groundedness * 0.35
    return {
        "claim_coverage": claim_coverage,
        "claim_pass_rate": claim_pass_rate,
        "groundedness_proxy": groundedness,
        "execution_score": execution,
        "answer_quality_proxy": quality,
    }


def evaluate_text_record(
    db,
    record: dict[str, Any],
    top_k: int,
    skip_generation: bool,
    final_route: str,
) -> dict[str, Any]:
    started = time.monotonic()
    query = record["question"]
    manual_id = record["manual_id"]
    gold = record.get("relevant_chunk_ids") or []
    retrieval = retrieve_text_pipeline(
        db,
        manual_id,
        query,
        top_k=top_k,
        final_route=final_route,
    )
    keyword = retrieval["keyword"]
    dense = retrieval["dense"]
    rrf = retrieval["rrf"]
    cross_encoder = retrieval["cross_encoder"]
    evidence = retrieval["evidence"]
    routes = {
        "bm25": route_result(keyword, gold, top_k, gold_pages=record.get("reference_pages"), evidence_units=record.get("evidence_units")),
        "dense": route_result(dense, gold, top_k, gold_pages=record.get("reference_pages"), evidence_units=record.get("evidence_units")),
        "rrf": route_result(rrf, gold, top_k, gold_pages=record.get("reference_pages"), evidence_units=record.get("evidence_units")),
    }
    if final_route == "cross_encoder":
        routes["cross_encoder"] = route_result(
            cross_encoder,
            gold,
            top_k,
            gold_pages=record.get("reference_pages"),
            evidence_units=record.get("evidence_units"),
        )

    answer = ""
    generation_error = None
    if not skip_generation:
        try:
            answer = complete_answer(query, evidence)
        except Exception as error:  # keep retrieval results even if an external LLM fails
            generation_error = f"{type(error).__name__}: {error}"
    predicted_refuse = detect_refusal(answer) if not skip_generation else False
    scores = score_answer(record, answer, evidence, predicted_refuse) if not skip_generation else {}
    return {
        "id": record["id"],
        "kind": "text",
        "category": record.get("category"),
        "difficulty": record.get("difficulty"),
        "should_refuse": bool(record.get("should_refuse")),
        "predicted_refuse": predicted_refuse,
        "answer_refuse_detected": predicted_refuse,
        "final_route": retrieval["final_route"],
        "routes": routes,
        "answer": answer,
        "answer_scores": scores,
        "expanded_query": retrieval["expanded_query"],
        "candidate_pool_size": retrieval["candidate_pool_size"],
        "cross_encoder_model": retrieval["cross_encoder_model"],
        "cross_encoder_error": retrieval["cross_encoder_error"],
        "cross_encoder_elapsed_seconds": retrieval["cross_encoder_elapsed_seconds"],
        "evidence": evidence,
        "evidence_chunk_ids": [item.get("chunk_id") for item in evidence],
        "generation_error": generation_error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def visual_candidate_snapshot(items: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    fields = (
        "chunk_id",
        "icon_group_id",
        "prototype_id",
        "asset_id",
        "object_name",
        "section_title",
        "page",
        "modality",
        "score",
        "rrf_score",
        "gated_fusion_score",
        "group_rank",
        "match_method",
        "agreement_routes",
        "matched_chunk_ids",
        "matched_modalities",
        "image_fingerprint",
        "collapsed_icon_group_ids",
        "vlm_verified",
        "vlm_confidence",
        "vlm_reason",
    )
    return [{field: item.get(field) for field in fields if item.get(field) is not None} for item in items[:limit]]


def evaluate_image_record(
    db,
    record: dict[str, Any],
    dataset_path: Path,
    top_k: int,
    skip_generation: bool,
    first_pass_vlm: bool,
    candidate_vlm: bool,
    use_opencv: bool,
    image_vector_variant: str,
    visual_fusion_profile: str,
    skip_text_support: bool,
    vlm_top_k: int,
) -> dict[str, Any]:
    started = time.monotonic()
    image_path = resolve_image_path(record["image_path"], dataset_path)
    manual_id = record["manual_id"]
    question = record.get("question") or "这张图片是什么，应该如何理解或使用？"
    expected_group = record.get("expected_icon_group_id")
    expected_groups = list(
        record.get("expected_icon_group_ids")
        or ([expected_group] if expected_group else [])
    )
    expected_visual_chunks = (
        record.get("expected_visual_chunk_ids")
        or record.get("relevant_visual_chunk_ids")
        or []
    )
    visual_gold_type = record.get("visual_gold_type") or (
        "visual_chunk" if expected_visual_chunks else "none"
    )
    # Every persisted visual object now has exactly one chunk (an icon group
    # owns one icon chunk), so icon and operation hits are scored by the same
    # visual chunk ID.  Group IDs remain available for diagnostics only.
    visual_gold = list(expected_visual_chunks)
    legacy_non_icon = record.get("negative_type") == "manual_non_icon_figure"

    if not first_pass_vlm:
        vision = {
            "recognized_object": "unknown",
            "visual_description": "First-pass vision skipped by evaluator.",
            "ocr_text": "",
            "candidate_names": [],
            "visual_keywords": [],
            "normalized_query": question,
            "risk_level": "unknown",
            "confidence": 0.0,
            "image_type": "unknown",
            "is_target_icon": False,
            "classification_confidence": 0.0,
            "classification_reason": "First-pass vision skipped by evaluator.",
            "vision_error": None,
        }
    else:
        vision = understand_vehicle_image(image_path, question)

    # Evaluation uses the same production routes and the same route-level
    # deduplication/fusion.  No image_type gate is consulted, including for the
    # 20 legacy PDF-figure rows.
    image_items = retrieve_visual_chunks_by_image_embedding(
        manual_id,
        image_path,
        top_k=20,
        embedding_variant=image_vector_variant,
    )
    opencv_items = (
        retrieve_visual_chunks_by_opencv(db, manual_id, image_path, top_k=20)
        if use_opencv
        else []
    )
    multimodal_items = retrieve_visual_chunks_by_multimodal(manual_id, image_path, top_k=20)

    image_groups = deduplicate_visual_candidates(db, manual_id, image_items, max_items=20)
    opencv_groups = deduplicate_visual_candidates(db, manual_id, opencv_items, max_items=20)
    multimodal_groups = deduplicate_visual_candidates(db, manual_id, multimodal_items, max_items=20)
    equal_fusion = union_group_fusion(
        image_groups,
        opencv_groups,
        multimodal_groups,
        top_k=10,
        image_weight=1.0,
        multimodal_weight=1.0,
    )
    multimodal_weighted_fusion = union_group_fusion(
        image_groups,
        opencv_groups,
        multimodal_groups,
        top_k=10,
        image_weight=0.6,
        multimodal_weight=1.0,
    )
    image_weighted_fusion = union_group_fusion(
        image_groups,
        opencv_groups,
        multimodal_groups,
        top_k=10,
        image_weight=1.0,
        multimodal_weight=0.6,
    )
    equal_opencv_fusion = union_group_fusion(
        image_groups,
        opencv_groups,
        multimodal_groups,
        top_k=10,
        image_weight=1.0,
        multimodal_weight=1.0,
        opencv_weight=0.2,
    )
    multimodal_weighted_opencv_fusion = union_group_fusion(
        image_groups,
        opencv_groups,
        multimodal_groups,
        top_k=10,
        image_weight=0.6,
        multimodal_weight=1.0,
        opencv_weight=0.2,
    )
    image_weighted_opencv_fusion = union_group_fusion(
        image_groups,
        opencv_groups,
        multimodal_groups,
        top_k=10,
        image_weight=1.0,
        multimodal_weight=0.6,
        opencv_weight=0.2,
    )
    legacy_gate = gated_group_fusion(
        image_groups,
        opencv_groups,
        multimodal_groups,
        top_k=10,
    )
    fusion_profiles = {
        "legacy_hard_gate": legacy_gate,
        "rrf_equal": equal_fusion,
        "rrf_multimodal_weighted": multimodal_weighted_fusion,
        "rrf_image_weighted": image_weighted_fusion,
        "rrf_equal_opencv": equal_opencv_fusion,
        "rrf_multimodal_weighted_opencv": multimodal_weighted_opencv_fusion,
        "rrf_image_weighted_opencv": image_weighted_opencv_fusion,
    }
    pre_vlm_candidates = fusion_profiles[visual_fusion_profile]
    verification = None
    visual_candidates = list(pre_vlm_candidates)
    if candidate_vlm and not vision.get("vision_error") and visual_candidates:
        verification = verify_visual_candidates(
            image_path,
            visual_candidates,
            vision=vision,
            question=question,
            top_k=vlm_top_k,
        )
        visual_candidates = _apply_vlm_verification(visual_candidates, verification)
        visual_anchor = _verified_visual_anchor(visual_candidates)
    elif visual_candidates:
        # Production closes with the fused Top1 and marks it as the retrieval
        # anchor, so the downstream body search runs without waiting for a
        # verification flag that will never be set.
        visual_anchor = _apply_retrieval_anchor(visual_candidates[0])
    else:
        visual_anchor = None

    visual_match = _build_visual_match([visual_anchor] if visual_anchor else visual_candidates)
    guardrail_refuse = _should_refuse_image_answer(visual_match, vision)
    accepted_visual_evidence = [visual_anchor] if visual_anchor else []

    # A grouped icon can legitimately return a representative occurrence from
    # another page. Page accuracy is meaningful only for exact visual chunks.
    visual_pages = (
        [record["source_pdf_page"]]
        if visual_gold_type == "visual_chunk" and record.get("source_pdf_page")
        else None
    )
    routes = {
        "pure_visual_image_embedding": route_result(image_groups, visual_gold, top_k, gold_pages=visual_pages),
        "multimodal_embedding": route_result(multimodal_groups, visual_gold, top_k, gold_pages=visual_pages),
        "rrf_image_1_multimodal_1": route_result(equal_fusion, visual_gold, top_k, gold_pages=visual_pages),
        "rrf_image_0_6_multimodal_1": route_result(multimodal_weighted_fusion, visual_gold, top_k, gold_pages=visual_pages),
        "rrf_image_1_multimodal_0_6": route_result(image_weighted_fusion, visual_gold, top_k, gold_pages=visual_pages),
        "rrf_image_1_multimodal_1_opencv_0_2": route_result(equal_opencv_fusion, visual_gold, top_k, gold_pages=visual_pages),
        "rrf_image_0_6_multimodal_1_opencv_0_2": route_result(multimodal_weighted_opencv_fusion, visual_gold, top_k, gold_pages=visual_pages),
        "rrf_image_1_multimodal_0_6_opencv_0_2": route_result(image_weighted_opencv_fusion, visual_gold, top_k, gold_pages=visual_pages),
        "legacy_hard_gate": route_result(legacy_gate, visual_gold, top_k, gold_pages=visual_pages),
        "full_chain_pre_vlm": route_result(pre_vlm_candidates, visual_gold, top_k, gold_pages=visual_pages),
        "full_chain_visual": route_result(accepted_visual_evidence, visual_gold, top_k, gold_pages=visual_pages),
    }
    diagnostic_routes = {
        "opencv": route_result(opencv_groups, visual_gold, top_k, gold_pages=visual_pages),
    }

    final_evidence: list[dict] = []
    text_evidence: list[dict] = []
    text_query = vision.get("normalized_query") or question
    if visual_anchor is not None and not skip_text_support:
        text_query = _build_image_text_query(vision, question, visual_anchor)
        text_evidence = retrieve_visual_anchor_text(
            db,
            manual_id,
            text_query,
            _build_visual_anchor(visual_anchor),
            top_k=4,
        )
        final_evidence = [
            {**visual_anchor, "evidence_role": "visual"},
            *[{**item, "evidence_role": "text"} for item in text_evidence[:4]],
        ]
    text_evidence_units = list(record.get("text_evidence_units") or [])
    text_gold = (
        unique(
            chunk_id
            for unit in text_evidence_units
            for chunk_id in unit.get("acceptable_chunk_ids") or []
        )
        if text_evidence_units
        else record.get("relevant_text_chunk_ids") or []
    )
    support_pages = unique(str(item.get("page")) for item in record.get("gold_evidence") or [])
    text_support_top_k = 4
    text_support_route = route_result(
        text_evidence,
        text_gold,
        text_support_top_k,
        gold_pages=[int(page) for page in support_pages] if support_pages else None,
        evidence_units=text_evidence_units or None,
    )

    answer = ""
    generation_error = None
    if not skip_generation:
        try:
            if guardrail_refuse:
                answer = _build_low_confidence_refusal(vision, visual_match, final_evidence)
            elif not text_evidence:
                answer = _build_insufficient_text_answer(visual_anchor or {})
            else:
                answer_question = _build_image_answer_question(
                    vision,
                    question,
                    text_query,
                    visual_match,
                )
                answer = complete_answer(answer_question, final_evidence)
        except Exception as error:
            generation_error = f"{type(error).__name__}: {error}"
    answer_refuse_detected = detect_refusal(answer) if not skip_generation else False
    final_refuse = guardrail_refuse or answer_refuse_detected
    scores = (
        score_answer(record, answer, final_evidence, final_refuse)
        if not skip_generation and not legacy_non_icon
        else {}
    )
    return {
        "id": record["id"],
        "kind": "image",
        "category": record.get("category"),
        "variant": record.get("variant") or "negative",
        "negative_type": record.get("negative_type"),
        "expected_refusal_reason": record.get("expected_refusal_reason") or record.get("refusal_reason"),
        "image_path": record["image_path"],
        "expected_icon": record.get("expected_icon"),
        "expected_icon_group_id": expected_group,
        "expected_icon_group_ids": expected_groups,
        "visual_gold_type": visual_gold_type,
        "expected_visual_title": record.get("expected_visual_title"),
        "expected_visual_modality": record.get("expected_visual_modality"),
        "expected_visual_chunk_ids": list(expected_visual_chunks),
        "source_pdf_page": record.get("source_pdf_page"),
        "source_occurrence_id": record.get("source_occurrence_id"),
        "source_asset_id": record.get("source_asset_id"),
        "gold_review_status": record.get("gold_review_status"),
        "image_vector_variant": image_vector_variant,
        "visual_fusion_profile": visual_fusion_profile,
        "vlm_candidate_top_k": vlm_top_k,
        "text_support_skipped": skip_text_support,
        "should_refuse": bool(record.get("should_refuse")),
        "visual_regression_evaluable": bool(
            record.get("visual_regression_evaluable", bool(visual_gold))
        ),
        # The current 20 PDF figures are not a valid general refusal set.
        "refusal_evaluable": bool(record.get("refusal_evaluable", False)),
        "dataset_compatibility": record.get("dataset_compatibility") or (
            "legacy_non_icon_unscored" if legacy_non_icon else "icon_visual_regression"
        ),
        # predicted_refuse intentionally remains the deterministic production
        # guardrail decision.  LLM lexical refusal is reported separately.
        "predicted_refuse": guardrail_refuse,
        "guardrail_refuse": guardrail_refuse,
        "answer_refuse_detected": answer_refuse_detected,
        "image_type": vision.get("image_type"),
        "is_target_icon": vision.get("is_target_icon"),
        "classification_confidence": vision.get("classification_confidence"),
        "classification_reason": vision.get("classification_reason"),
        "recognized_object": visual_match.get("title") if not guardrail_refuse and visual_match else vision.get("recognized_object"),
        "visual_match": visual_match,
        "verification": verification,
        "vlm_evaluated": verification is not None,
        "routes": routes,
        "diagnostic_routes": diagnostic_routes,
        "text_support_route": text_support_route,
        "text_support_top_k": text_support_top_k,
        "text_query": text_query,
        "route_candidates": {
            "pure_visual_image_embedding": visual_candidate_snapshot(image_groups),
            "multimodal_embedding": visual_candidate_snapshot(multimodal_groups),
            "rrf_image_1_multimodal_1": visual_candidate_snapshot(equal_fusion),
            "rrf_image_0_6_multimodal_1": visual_candidate_snapshot(multimodal_weighted_fusion),
            "rrf_image_1_multimodal_0_6": visual_candidate_snapshot(image_weighted_fusion),
            "rrf_image_1_multimodal_1_opencv_0_2": visual_candidate_snapshot(equal_opencv_fusion),
            "rrf_image_0_6_multimodal_1_opencv_0_2": visual_candidate_snapshot(multimodal_weighted_opencv_fusion),
            "rrf_image_1_multimodal_0_6_opencv_0_2": visual_candidate_snapshot(image_weighted_opencv_fusion),
            "legacy_hard_gate": visual_candidate_snapshot(legacy_gate),
            "full_chain_pre_vlm": visual_candidate_snapshot(pre_vlm_candidates),
            "full_chain_visual": visual_candidate_snapshot(accepted_visual_evidence),
            "opencv_diagnostic": visual_candidate_snapshot(opencv_groups),
        },
        "answer": answer,
        "evidence_status": (
            "manual_visual_not_found"
            if guardrail_refuse
            else "insufficient_text_evidence"
            if not text_evidence
            else "verified"
        ),
        "answer_scores": scores,
        "evidence": final_evidence,
        "evidence_chunk_ids": [item.get("chunk_id") for item in final_evidence],
        "generation_error": generation_error,
        "vision_error": vision.get("vision_error"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_route_collection(
    rows: list[dict[str, Any]],
    field: str = "routes",
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("should_refuse"):
            continue
        for route_name, route in row.get(field, {}).items():
            for metric_name, value in route.get("metrics", {}).items():
                values[route_name][metric_name].append(float(value))
    return {
        route_name: {metric_name: mean(metric_values) for metric_name, metric_values in metric_map.items()}
        for route_name, metric_map in values.items()
    }


def aggregate_routes(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return aggregate_route_collection(rows, "routes")


def aggregate_single_route(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("should_refuse"):
            continue
        for metric_name, value in row.get(field, {}).get("metrics", {}).items():
            values[metric_name].append(float(value))
    return {metric_name: mean(metric_values) for metric_name, metric_values in values.items()}


def aggregate_visual_cutoffs(
    rows: list[dict[str, Any]],
    cutoffs: tuple[int, ...] = (1, 3, 5),
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate visual-gold and page metrics at smaller cutoffs."""

    route_specs = (
        ("pure_visual_image_embedding", "routes", "pure_visual_image_embedding"),
        ("multimodal_embedding", "routes", "multimodal_embedding"),
        ("opencv", "diagnostic_routes", "opencv_diagnostic"),
        ("rrf_image_1_multimodal_1", "routes", "rrf_image_1_multimodal_1"),
        ("rrf_image_0_6_multimodal_1", "routes", "rrf_image_0_6_multimodal_1"),
        ("rrf_image_1_multimodal_0_6", "routes", "rrf_image_1_multimodal_0_6"),
        ("rrf_image_1_multimodal_1_opencv_0_2", "routes", "rrf_image_1_multimodal_1_opencv_0_2"),
        ("rrf_image_0_6_multimodal_1_opencv_0_2", "routes", "rrf_image_0_6_multimodal_1_opencv_0_2"),
        ("rrf_image_1_multimodal_0_6_opencv_0_2", "routes", "rrf_image_1_multimodal_0_6_opencv_0_2"),
        ("legacy_hard_gate", "routes", "legacy_hard_gate"),
        ("full_chain_pre_vlm", "routes", "full_chain_pre_vlm"),
        ("full_chain_visual", "routes", "full_chain_visual"),
    )
    collected: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        if row.get("should_refuse"):
            continue
        # Visual hits are always scored against the unique visual chunk, for
        # icon groups and operation figures alike.
        gold_keys = [
            str(value)
            for value in row.get("expected_visual_chunk_ids") or []
            if value
        ]
        if not gold_keys:
            continue
        gold_pages = (
            [str(row["source_pdf_page"])]
            if row.get("visual_gold_type") == "visual_chunk" and row.get("source_pdf_page")
            else []
        )
        snapshots = row.get("route_candidates") or {}
        for route_name, field, snapshot_name in route_specs:
            route_key = "opencv" if route_name == "opencv" else route_name
            top_keys = list((row.get(field) or {}).get(route_key, {}).get("top_keys") or [])
            candidates = list(snapshots.get(snapshot_name) or [])
            top_pages = [str(item.get("page")) for item in candidates if item.get("page")]
            for cutoff in cutoffs:
                metrics = retrieval_at_k(top_keys, gold_keys, cutoff)
                # The shared retrieval helper reports the reciprocal rank over
                # the complete saved list.  For this cutoff table we need the
                # truncated variant: a relevant group below k contributes 0.
                truncated_rank = next(
                    (rank for rank, key in enumerate(unique(top_keys)[:cutoff], start=1) if key in set(gold_keys)),
                    None,
                )
                metrics["reciprocal_rank"] = 1.0 / truncated_rank if truncated_rank else 0.0
                if gold_pages:
                    page_metrics = retrieval_at_k(top_pages, gold_pages, cutoff)
                    metrics.update({f"page_{name}": value for name, value in page_metrics.items()})
                for metric_name, value in metrics.items():
                    collected[route_name][f"at_{cutoff}"][metric_name].append(float(value))
    return {
        route_name: {
            cutoff: {metric: mean(values) for metric, values in metrics.items()}
            for cutoff, metrics in cutoff_map.items()
        }
        for route_name, cutoff_map in collected.items()
    }


def aggregate_visual_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    routes = aggregate_routes(rows)
    opencv = aggregate_route_collection(rows, "diagnostic_routes").get("opencv", {})
    return {
        "count": len(rows),
        "routes": {**routes, "opencv": opencv},
        "visual_cutoffs": aggregate_visual_cutoffs(rows),
    }


def aggregate_vlm_verification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate candidate coverage from the VLM's conditional selection skill."""

    answerable = [
        row
        for row in rows
        if not row.get("should_refuse") and row.get("expected_visual_chunk_ids")
    ]
    evaluated = [row for row in answerable if row.get("vlm_evaluated")]
    candidate_cutoff = int(next((row.get("vlm_candidate_top_k") for row in evaluated), 0) or 0)
    covered = 0
    selected_correct = 0
    rejected_with_gold_present = 0
    wrong_selection_with_gold_present = 0
    for row in evaluated:
        gold = set(
            str(value)
            for value in row.get("expected_visual_chunk_ids") or []
            if value
        )
        candidate_keys = set(
            (row.get("routes") or {})
            .get("full_chain_pre_vlm", {})
            .get("top_keys", [])[:candidate_cutoff]
        )
        selected_keys = set(
            (row.get("routes") or {})
            .get("full_chain_visual", {})
            .get("top_keys", [])[:1]
        )
        has_gold = bool(candidate_keys & gold)
        is_correct = bool(selected_keys & gold)
        if has_gold:
            covered += 1
            if is_correct:
                selected_correct += 1
            elif not (row.get("verification") or {}).get("matched"):
                rejected_with_gold_present += 1
            else:
                wrong_selection_with_gold_present += 1
    total = len(evaluated)
    return {
        "status": "available" if total else "not_run",
        "answerable_count": total,
        "candidate_cutoff": candidate_cutoff,
        "candidate_covered_count": covered,
        "candidate_coverage": covered / total if total else 0.0,
        "selected_correct_count": selected_correct,
        "conditional_selection_accuracy": selected_correct / covered if covered else 0.0,
        "end_to_end_anchor_accuracy": selected_correct / total if total else 0.0,
        "false_rejection_with_gold_present_count": rejected_with_gold_present,
        "wrong_selection_with_gold_present_count": wrong_selection_with_gold_present,
    }


def aggregate_llm_judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judgments = [row["llm_judge"] for row in rows if not row.get("should_refuse") and row.get("llm_judge")]
    if not judgments:
        return {}
    return {
        "judge_model": next(
            (judgment.get("judge_model") for judgment in judgments if judgment.get("judge_model")),
            "current Codex model",
        ),
        "scale": "0.0 to 1.0 in 0.2 increments",
        "count": len(judgments),
        **{
            field: mean(float(judgment[field]) for judgment in judgments)
            for field in ("answer_correctness", "groundedness", "answer_execution")
        },
    }


def refusal_metrics(rows: list[dict[str, Any]], field: str = "predicted_refuse") -> dict[str, float | int]:
    if not rows:
        return {
            "status": "not_available",
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
        }
    tp = sum(row["should_refuse"] and row.get(field, False) for row in rows)
    fp = sum(not row["should_refuse"] and row.get(field, False) for row in rows)
    fn = sum(row["should_refuse"] and not row.get(field, False) for row in rows)
    tn = sum(not row["should_refuse"] and not row.get(field, False) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


def latency_metrics(rows: list[dict[str, Any]], field: str = "elapsed_seconds") -> dict[str, float]:
    values = [float(row.get(field) or 0) for row in rows if row.get(field) is not None]
    return {
        "mean_seconds": mean(values),
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
        "max_seconds": max(values, default=0.0),
    }


def aggregate_answer_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    answerable = [row for row in rows if not row.get("should_refuse") and row.get("answer_scores")]
    names = ["claim_coverage", "claim_pass_rate", "groundedness_proxy", "execution_score", "answer_quality_proxy"]
    return {name: mean(row["answer_scores"].get(name, 0.0) for row in answerable) for name in names}


def variant_metrics(rows: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    variants = unique(
        row.get("variant")
        for row in rows
        if not row.get("should_refuse") and row.get("variant")
    )
    for variant in variants:
        selected = [row for row in rows if row.get("variant") == variant and not row.get("should_refuse")]
        metric_name = f"hit_at_{top_k}"
        result[variant] = {
            f"full_chain_visual_{metric_name}": mean(
                row.get("routes", {}).get("full_chain_visual", {}).get("metrics", {}).get(metric_name, 0.0)
                for row in selected
            ),
            "count": len(selected),
        }
    return result


def aggregate(text_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    difficulty_routes = {
        difficulty: aggregate_routes([row for row in text_rows if row.get("difficulty") == difficulty])
        for difficulty in ("easy", "medium", "hard")
    }
    image_true_refusal_rows = [
        row
        for row in image_rows
        if row.get("should_refuse") and row.get("refusal_evaluable")
    ]
    # Every answerable row is a valid negative example for the refusal
    # decision; otherwise false refusals would be silently excluded and
    # refusal precision would be artificially inflated.
    image_decision_rows = [
        row
        for row in image_rows
        if not row.get("should_refuse") or row.get("refusal_evaluable")
    ]
    image_answerable_groups = {
        compatibility: [
            row
            for row in image_rows
            if row.get("dataset_compatibility") == compatibility and not row.get("should_refuse")
        ]
        for compatibility in (
            "icon_visual_regression",
            "general_visual_answerable",
            "real_icon_strong_positive",
        )
    }
    image_ablation_by_dataset = {
        compatibility: aggregate_visual_ablation(rows)
        for compatibility, rows in image_answerable_groups.items()
        if rows
    }
    image_routes_by_dataset = {
        compatibility: {
            **image_ablation_by_dataset[compatibility],
            "text_support": aggregate_single_route(rows, "text_support_route"),
            "answers": aggregate_answer_scores(rows),
        }
        for compatibility, rows in image_answerable_groups.items()
        if rows
    }
    image_answerable_rows = [row for rows in image_answerable_groups.values() for row in rows]
    image_answerable_ablation = aggregate_visual_ablation(image_answerable_rows)
    image_rows_by_visual_type = {
        visual_type: [
            row
            for row in image_answerable_rows
            if row.get("visual_gold_type") == visual_type
        ]
        for visual_type in ("icon_group", "visual_chunk")
    }
    image_ablation_by_visual_type = {
        visual_type: aggregate_visual_ablation(rows)
        for visual_type, rows in image_rows_by_visual_type.items()
        if rows
    }
    image_refusal_by_type = {
        negative_type: {
            "count": len(rows),
            **refusal_metrics(rows),
            "answer_refuse_recall": refusal_metrics(rows, "answer_refuse_detected").get("recall"),
        }
        for negative_type in unique(
            row.get("negative_type") or row.get("variant")
            for row in image_true_refusal_rows
            if row.get("negative_type") or row.get("variant")
        )
        if (
            rows := [
                row
                for row in image_true_refusal_rows
                if (row.get("negative_type") or row.get("variant")) == negative_type
            ]
        )
    }
    image_refusal = refusal_metrics(image_decision_rows)
    image_answer_refusal = refusal_metrics(
        image_decision_rows,
        "answer_refuse_detected",
    )
    return {
        "text": {
            "count": len(text_rows),
            "routes": aggregate_routes(text_rows),
            "routes_by_difficulty": difficulty_routes,
            "refusal": refusal_metrics(text_rows),
            "answers": aggregate_answer_scores(text_rows),
            "errors": sum(bool(row.get("generation_error")) for row in text_rows),
            "fatal_errors": sum(bool(row.get("fatal_error")) for row in text_rows),
            "cross_encoder_errors": sum(bool(row.get("cross_encoder_error")) for row in text_rows),
            "latency": latency_metrics(text_rows),
            "cross_encoder_latency": latency_metrics(text_rows, "cross_encoder_elapsed_seconds"),
        },
        "image": {
            "count": len(image_rows),
            "answerable_count": sum(bool(row.get("visual_regression_evaluable", not row.get("should_refuse"))) for row in image_rows),
            "icon_visual_count": sum(
                not row.get("should_refuse") and row.get("visual_gold_type") == "icon_group"
                for row in image_rows
            ),
            "operation_visual_count": sum(
                not row.get("should_refuse") and row.get("visual_gold_type") == "visual_chunk"
                for row in image_rows
            ),
            "strong_real_count": sum(
                row.get("dataset_compatibility") == "real_icon_strong_positive"
                for row in image_rows
            ),
            "true_refusal_count": sum(
                row.get("dataset_compatibility") == "true_general_refusal"
                for row in image_rows
            ),
            "legacy_non_icon_unscored_count": sum(
                row.get("dataset_compatibility") == "legacy_non_icon_unscored"
                for row in image_rows
            ),
            "refusal_evaluable_count": len(image_decision_rows),
            "routes": aggregate_routes(image_rows),
            "answerable_ablation": image_answerable_ablation,
            "ablation_by_dataset": image_ablation_by_dataset,
            "ablation_by_visual_type": image_ablation_by_visual_type,
            "routes_by_dataset": image_routes_by_dataset,
            "visual_cutoffs": aggregate_visual_cutoffs(image_rows),
            "vlm_verification": aggregate_vlm_verification(image_rows),
            "diagnostic_routes": aggregate_route_collection(image_rows, "diagnostic_routes"),
            "text_support": aggregate_single_route(image_rows, "text_support_route"),
            "refusal": image_refusal,
            "false_refusal_count": image_refusal.get("fp", 0),
            "false_accept_count": image_refusal.get("fn", 0),
            "refusal_by_type": image_refusal_by_type,
            "answer_refusal_lexical": image_answer_refusal,
            "answers": aggregate_answer_scores(image_rows),
            "llm_judge": aggregate_llm_judge(image_rows),
            "by_variant": variant_metrics(image_rows, top_k),
            "generation_errors": sum(bool(row.get("generation_error")) for row in image_rows),
            "vision_errors": sum(bool(row.get("vision_error")) for row in image_rows),
            "fatal_errors": sum(bool(row.get("fatal_error")) for row in image_rows),
            "latency": latency_metrics(image_rows),
        },
    }


def percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def markdown_report(result: dict[str, Any], top_k: int) -> str:
    summary = result["summary"]
    final_route = result.get("run", {}).get("text_final_route") or "cross_encoder"
    is_v2_text = "v2" in Path(str(result.get("run", {}).get("text_dataset") or "")).stem.lower()
    lines = [
        "# 车主手册多模态 RAG 评测结果",
        "",
        f"- 运行时间：{result['run']['finished_at']}",
        f"- Top-k：{top_k}",
        f"- 文本样本：{summary['text']['count']}；图片样本：{summary['image']['count']}",
        "- 回答质量、忠实度和执行度为确定性字符二元组代理指标；检索与拒答指标为严格自动指标。",
    ]
    if summary["text"]["count"]:
        lines.append(f"- 文本最终回答证据路线：`{final_route}` Top-{top_k}。")
    if summary["image"]["count"]:
        lines.append(
            "- 图片向量版本："
            f"`{result.get('run', {}).get('image_vector_variant', 'native')}`；"
            "送入VLM的融合路线："
            f"`{result.get('run', {}).get('visual_fusion_profile', 'legacy_hard_gate')}`。"
        )
    rerank_model = result.get("run", {}).get("text_rerank_model")
    if rerank_model and summary["text"]["count"]:
        lines.append(f"- 文本 Cross-Encoder：`{rerank_model}`；与 RRF 使用相同的 Top50 候选集合。")
    if is_v2_text and summary["text"]["count"]:
        lines.append("- v2 的 Recall 按必答事实计算：同一事实由任一有效 Chunk 命中即可，不重复惩罚等价证据块。")
    if summary["text"]["count"]:
        lines.extend([
            "",
            "## 文本检索消融",
            "",
            f"| 路线 | Evidence Hit@{top_k} | Precision@{top_k} | Evidence Recall@{top_k} | F1@{top_k} | Page Hit@{top_k} | MRR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, metrics in summary["text"]["routes"].items():
            lines.append(
                f"| {name} | {percent(metrics.get(f'hit_at_{top_k}'))} | "
                f"{percent(metrics.get(f'precision_at_{top_k}'))} | {percent(metrics.get(f'recall_at_{top_k}'))} | "
                f"{percent(metrics.get(f'f1_at_{top_k}'))} | {percent(metrics.get(f'page_hit_at_{top_k}'))} | "
                f"{metrics.get('reciprocal_rank', 0):.4f} |"
            )

    if any(summary["text"].get("routes_by_difficulty", {}).values()):
        lines.extend([
            "",
            "## 文本检索难度分组",
            "",
            f"| 难度 | 路线 | Evidence Hit@{top_k} | Precision@{top_k} | Evidence Recall@{top_k} | F1@{top_k} | Page Hit@{top_k} |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for difficulty in ("easy", "medium", "hard"):
            for name, metrics in summary["text"]["routes_by_difficulty"].get(difficulty, {}).items():
                lines.append(
                    f"| {difficulty} | {name} | {percent(metrics.get(f'hit_at_{top_k}'))} | "
                    f"{percent(metrics.get(f'precision_at_{top_k}'))} | {percent(metrics.get(f'recall_at_{top_k}'))} | "
                    f"{percent(metrics.get(f'f1_at_{top_k}'))} | {percent(metrics.get(f'page_hit_at_{top_k}'))} |"
                )

    if summary["image"]["count"]:
        route_specs = [
            ("pure_visual_image_embedding", "纯图片向量", "独立消融路线"),
            ("multimodal_embedding", "纯多模态向量", "独立消融路线"),
            ("opencv", "OpenCV 单路", "诊断路线，不单独放行"),
            ("rrf_image_1_multimodal_1", "RRF 图片1.0/多模态1.0", "两路等权融合"),
            ("rrf_image_0_6_multimodal_1", "RRF 图片0.6/多模态1.0", "多模态加权融合"),
            ("rrf_image_1_multimodal_0_6", "RRF 图片1.0/多模态0.6", "图片加权融合"),
            ("rrf_image_1_multimodal_1_opencv_0_2", "等权RRF + OpenCV0.2", "OpenCV增量消融"),
            ("rrf_image_0_6_multimodal_1_opencv_0_2", "多模态加权RRF + OpenCV0.2", "OpenCV增量消融"),
            ("rrf_image_1_multimodal_0_6_opencv_0_2", "图片加权RRF + OpenCV0.2", "OpenCV增量消融"),
            ("legacy_hard_gate", "旧图片分数硬门控", "回退基线"),
            ("full_chain_pre_vlm", "本次选定融合，VLM前", "生产候选检查点"),
            ("full_chain_visual", "本次选定融合，VLM后", "最终单一视觉锚点"),
        ]
        route_meanings = {
            "pure_visual_image_embedding": "图片向量Top20，经路线内去重。",
            "multimodal_embedding": "多模态向量Top20，仅保留视觉Chunk，经路线内去重。",
            "opencv": "OpenCV视觉相似度Top20；只作单路诊断。",
            "rrf_image_1_multimodal_1": "图片与多模态候选并集，按1.0/1.0做RRF。",
            "rrf_image_0_6_multimodal_1": "图片与多模态候选并集，按0.6/1.0做RRF。",
            "rrf_image_1_multimodal_0_6": "图片与多模态候选并集，按1.0/0.6做RRF。",
            "rrf_image_1_multimodal_1_opencv_0_2": "等权两路RRF后，仅给已入池的OpenCV同对象增加0.2权重。",
            "rrf_image_0_6_multimodal_1_opencv_0_2": "多模态加权RRF后增加OpenCV0.2辅助分。",
            "rrf_image_1_multimodal_0_6_opencv_0_2": "图片加权RRF后增加OpenCV0.2辅助分。",
            "legacy_hard_gate": "旧逻辑：只有图片向量分数不低于50的对象可以进入。",
            "full_chain_pre_vlm": "由--visual-fusion-profile选中的候选路线输出候选池；VLM只读取--vlm-top-k指定的前K条。",
            "full_chain_visual": "VLM从实际送入的前K条候选中选择一个视觉锚点，或拒绝全部候选。",
        }
        lines.extend([
            "",
            "## 视觉检索消融实验",
            "",
            "### 实验路线",
            "",
            "| 路线 | 性质 | 含义 |",
            "|---|---|---|",
        ])
        for name, label, role in route_specs:
            lines.append(f"| {label} | {role} | {route_meanings[name]} |")
        lines.extend([
            "",
            "> VLM二验后只剩一个最终视觉锚点，不是候选列表。因此该行的Hit@1、Hit@3、Hit@5必然相同，应理解为“最终锚点准确率”；VLM二验前的Top1/Top3/Top5才是候选召回指标。",
        ])

        visual_type_specs = [
            (
                "icon_group",
                "图标组召回",
                "图标按 `icon_group_id` 评分；同组代表图来自其他页也算正确，因此不评 Page Hit。",
                False,
            ),
            (
                "visual_chunk",
                "操作图精确召回",
                "操作图按精确 `visual_chunk_id` 评分；同时报告原图所在页是否命中。",
                True,
            ),
        ]
        ablation_by_visual_type = summary["image"].get("ablation_by_visual_type") or {}
        for visual_type, title, scope_note, show_page in visual_type_specs:
            ablation = ablation_by_visual_type.get(visual_type) or {}
            routes = ablation.get("routes") or {}
            cutoffs = ablation.get("visual_cutoffs") or {}
            lines.extend([
                "",
                f"### {title}（n={ablation.get('count', 0)}）",
                "",
                f"> {scope_note}",
                "",
                "| 路线 | Hit@1 | Hit@3 | Hit@5 | MRR |"
                + (" Page Hit@5 |" if show_page else ""),
                "|---|---:|---:|---:|---:|"
                + ("---:|" if show_page else ""),
            ])
            for name, label, _ in route_specs:
                at_1 = cutoffs.get(name, {}).get("at_1", {})
                at_3 = cutoffs.get(name, {}).get("at_3", {})
                at_5 = cutoffs.get(name, {}).get("at_5", {})
                route_metrics = routes.get(name, {})
                lines.append(
                    f"| {label} | {percent(at_1.get('hit_at_1'))} | "
                    f"{percent(at_3.get('hit_at_3'))} | {percent(at_5.get('hit_at_5'))} | "
                    f"{route_metrics.get('reciprocal_rank', 0):.4f} |"
                    + (
                        f" {percent(at_5.get('page_hit_at_5'))} |"
                        if show_page
                        else ""
                    )
                )

        combined_ablation = summary["image"].get("answerable_ablation") or {}
        combined_cutoffs = combined_ablation.get("visual_cutoffs") or {}
        lines.extend([
            "",
            f"### 可回答样本宏观汇总（n={combined_ablation.get('count', 0)}）",
            "",
            "> 该表仅用于观察总体趋势：50条图标按 `icon_group_id` 评分，20条操作图按精确 `visual_chunk_id` 评分，混合了两种 Gold 评分方式，不能替代上面的两张分组消融表。",
            "",
            "| 路线 | Hit@1 | Hit@3 | Hit@5 | MRR |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, label, _ in route_specs:
            at_1 = combined_cutoffs.get(name, {}).get("at_1", {})
            at_3 = combined_cutoffs.get(name, {}).get("at_3", {})
            at_5 = combined_cutoffs.get(name, {}).get("at_5", {})
            route_metrics = (combined_ablation.get("routes") or {}).get(name, {})
            lines.append(
                f"| {label} | {percent(at_1.get('hit_at_1'))} | "
                f"{percent(at_3.get('hit_at_3'))} | {percent(at_5.get('hit_at_5'))} | "
                f"{route_metrics.get('reciprocal_rank', 0):.4f} |"
            )

        vlm = summary["image"].get("vlm_verification") or {}
        vlm_cutoff = int(vlm.get("candidate_cutoff") or result.get("run", {}).get("vlm_top_k") or 0)
        lines.extend([
            "",
            "### VLM二验独立指标",
            "",
            (
                "> 本轮未运行VLM；下表只在二验实际执行时才有意义。"
                if vlm.get("status") == "not_run"
                else f"> 条件选择准确率只统计Gold已经进入候选Top{vlm_cutoff}的样本。"
            ),
            "",
            f"| 候选Top{vlm_cutoff}覆盖率 | 条件选择准确率 | 端到端锚点准确率 | Gold在候选中但VLM拒绝 | Gold在候选中但VLM选错 |",
            "|---:|---:|---:|---:|---:|",
            f"| {percent(vlm.get('candidate_coverage'))} | "
            f"{percent(vlm.get('conditional_selection_accuracy'))} | "
            f"{percent(vlm.get('end_to_end_anchor_accuracy'))} | "
            f"{vlm.get('false_rejection_with_gold_present_count', 0)} | "
            f"{vlm.get('wrong_selection_with_gold_present_count', 0)} |",
        ])

        dataset_labels = {
            "icon_visual_regression": "原有图标",
            "real_icon_strong_positive": "实拍强正图标",
            "general_visual_answerable": "操作示意图",
        }
        routes_by_dataset = summary["image"].get("routes_by_dataset") or {}
        text_support_k = 4
        lines.extend([
            "",
            "## 视觉锚点后的统一正文召回",
            "",
            "> 本节评估的是VLM确认视觉锚点之后的正文BM25/Dense→RRF→Cross-Encoder，不属于上面的视觉检索消融。",
            "",
            f"| 样本分组 | 数量 | Evidence Hit@{text_support_k} | Evidence Recall@{text_support_k} | Page Hit@{text_support_k} | MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for compatibility, label in dataset_labels.items():
            group = routes_by_dataset.get(compatibility) or {}
            metrics = group.get("text_support") or {}
            lines.append(
                f"| {label} | {group.get('count', 0)} | {percent(metrics.get(f'hit_at_{text_support_k}'))} | "
                f"{percent(metrics.get(f'recall_at_{text_support_k}'))} | "
                f"{percent(metrics.get(f'page_hit_at_{text_support_k}'))} | "
                f"{metrics.get('reciprocal_rank', 0):.4f} |"
            )
        text_support = summary["image"].get("text_support") or {}
        lines.append(
            f"| 全部可回答样本 | {summary['image']['answerable_count']} | "
            f"{percent(text_support.get(f'hit_at_{text_support_k}'))} | "
            f"{percent(text_support.get(f'recall_at_{text_support_k}'))} | "
            f"{percent(text_support.get(f'page_hit_at_{text_support_k}'))} | "
            f"{text_support.get('reciprocal_rank', 0):.4f} |"
        )

    if result.get("run", {}).get("skip_generation"):
        lines.extend(["", "## 拒答与回答质量", ""])
        if (
            summary["image"]["count"]
            and (summary["image"].get("vlm_verification") or {}).get("status") == "available"
        ):
            refusal = summary["image"]["refusal"]
            lines.extend([
                "本轮未生成最终答案；以下只报告VLM视觉门控的拒答表现。",
                "",
                "| 拒答P | 拒答R | 拒答F1 | 拒答Accuracy | 误拒答 | 误接收 |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| {percent(refusal.get('precision'))} | {percent(refusal.get('recall'))} | "
                f"{percent(refusal.get('f1'))} | {percent(refusal.get('accuracy'))} | "
                f"{refusal.get('fp', 0)} | {refusal.get('fn', 0)} |",
            ])
            refusal_labels = {
                "other_model_icon": "其他车型图标",
                "unrelated_image": "无关图片",
                "blurry_unrecognizable": "严重模糊",
                "ambiguous_full_dashboard": "多目标完整仪表盘",
            }
            refusal_by_type = summary["image"].get("refusal_by_type") or {}
            if refusal_by_type:
                lines.extend([
                    "",
                    "| 拒答类型 | 数量 | VLM门控拒答召回 |",
                    "|---|---:|---:|",
                ])
                for negative_type, metrics in refusal_by_type.items():
                    lines.append(
                        f"| {refusal_labels.get(negative_type, negative_type)} | "
                        f"{metrics.get('count', 0)} | {percent(metrics.get('recall'))} |"
                    )
        else:
            lines.append("本次为纯检索消融，未运行VLM和答案生成，因此不报告拒答指标。")
    else:
        lines.extend(["", "## 拒答与回答质量", "", "| 链路 | 拒答P | 拒答R | 拒答F1 | 拒答Accuracy | 误拒答 | 误接收 | Claim覆盖 | 忠实度代理 | 执行度 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for kind in ("text", "image"):
            if not summary[kind]["count"]:
                continue
            refusal = summary[kind]["refusal"]
            if refusal.get("status") == "not_available":
                continue
            answers = summary[kind]["answers"]
            lines.append(
                f"| {kind} | {percent(refusal['precision'])} | {percent(refusal['recall'])} | {percent(refusal['f1'])} | "
                f"{percent(refusal['accuracy'])} | {refusal.get('fp', 0)} | {refusal.get('fn', 0)} | "
                f"{percent(answers.get('claim_coverage'))} | "
                f"{percent(answers.get('groundedness_proxy'))} | {percent(answers.get('execution_score'))} |"
            )

        if summary["image"]["count"] and summary["image"]["answer_refusal_lexical"].get("status") != "not_available":
            image_lexical = summary["image"]["answer_refusal_lexical"]
            lines.append(
                f"| image(final lexical) | {percent(image_lexical['precision'])} | {percent(image_lexical['recall'])} | "
                f"{percent(image_lexical['f1'])} | {percent(image_lexical['accuracy'])} | "
                f"{image_lexical.get('fp', 0)} | {image_lexical.get('fn', 0)} | - | - | - |"
            )
        refusal_by_type = summary.get("image", {}).get("refusal_by_type") or {}
        if refusal_by_type:
            refusal_labels = {
                "other_model_icon": "其他车型图标",
                "unrelated_image": "无关图片",
                "blurry_unrecognizable": "严重模糊",
                "ambiguous_full_dashboard": "多目标完整仪表盘",
            }
            lines.extend([
                "",
                "### 真实拒答分类型结果",
                "",
                "| 拒答类型 | 数量 | 链路拒答召回 | 最终答案拒答召回 |",
                "|---|---:|---:|---:|",
            ])
            for negative_type, metrics in refusal_by_type.items():
                lines.append(
                    f"| {refusal_labels.get(negative_type, negative_type)} | {metrics.get('count', 0)} | "
                    f"{percent(metrics.get('recall'))} | {percent(metrics.get('answer_refuse_recall'))} |"
                )
        if summary["image"]["count"] and summary["image"]["refusal"].get("status") == "not_available":
            lines.extend([
                "",
                "图片全局拒答指标暂停：当前20条旧负样本全部是手册内普通视觉对象，缺少汽车无关图、模糊图、当前手册不存在的图标和其他车型图标。",
            ])
    if summary["image"]["count"]:
        lines.extend([
            "",
            "## 可回答样本变体分组（VLM二验后）",
            "",
            "| 变体 | 数量 | 最终视觉锚点准确率 |",
            "|---|---:|---:|",
        ])
        for variant, metrics in summary["image"]["by_variant"].items():
            lines.append(f"| {variant} | {metrics['count']} | {percent(metrics[f'full_chain_visual_hit_at_{top_k}'])} |")

    lines.extend(["", "## 运行异常", ""])
    if summary["text"]["count"]:
        lines.append(
            f"- 文本生成异常：{summary['text']['errors']}；Cross-Encoder异常："
            f"{summary['text']['cross_encoder_errors']}；Fatal error：{summary['text']['fatal_errors']}"
        )
    if summary["image"]["count"]:
        lines.append(
            f"- 图片生成异常：{summary['image']['generation_errors']}；VLM 初判异常：{summary['image']['vision_errors']}；"
            f"Fatal error：{summary['image']['fatal_errors']}"
        )

    lines.extend(["", "## 端到端耗时", "", "| 链路 | Mean | P50 | P95 | Max |", "|---|---:|---:|---:|---:|"])
    if summary["text"]["count"]:
        lines.append(
            f"| text | {summary['text']['latency']['mean_seconds']:.2f}s | {summary['text']['latency']['p50_seconds']:.2f}s | "
            f"{summary['text']['latency']['p95_seconds']:.2f}s | {summary['text']['latency']['max_seconds']:.2f}s |"
        )
        lines.append(
            f"| text Cross-Encoder only | {summary['text']['cross_encoder_latency']['mean_seconds']:.2f}s | "
            f"{summary['text']['cross_encoder_latency']['p50_seconds']:.2f}s | "
            f"{summary['text']['cross_encoder_latency']['p95_seconds']:.2f}s | "
            f"{summary['text']['cross_encoder_latency']['max_seconds']:.2f}s |"
        )
    if summary["image"]["count"]:
        lines.append(
            f"| image | {summary['image']['latency']['mean_seconds']:.2f}s | {summary['image']['latency']['p50_seconds']:.2f}s | "
            f"{summary['image']['latency']['p95_seconds']:.2f}s | {summary['image']['latency']['max_seconds']:.2f}s |"
        )
    lines.extend(["", "> 注意：回答质量代理指标适合持续回归比较，不应替代人工复核或独立 LLM-as-a-Judge。", ""])
    image_judge = summary.get("image", {}).get("llm_judge") or {}
    if image_judge:
        lines.extend([
            "## 图片回答 LLM-as-a-Judge",
            "",
            f"- 裁判：{image_judge.get('judge_model')}；正样本：{image_judge.get('count')}；分档：0.0～1.0，每 0.2 一档。",
            "",
            "| Answer Correctness | Groundedness | Answer Execution |",
            "|---:|---:|---:|",
            f"| {percent(image_judge.get('answer_correctness'))} | {percent(image_judge.get('groundedness'))} | {percent(image_judge.get('answer_execution'))} |",
            "",
        ])
    return "\n".join(lines)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, text_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> None:
    write_json_atomic(path, {"text_rows": text_rows, "image_rows": image_rows})


def _record_map(path: Path) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in load_jsonl(path)}


def _backfill_route_page_metrics(row: dict[str, Any], record: dict[str, Any], top_k: int) -> None:
    for route_name, route in row.get("routes", {}).items():
        if row.get("kind") == "text":
            gold_pages = record.get("reference_pages") or []
        elif route_name == "full_chain_text_support":
            gold_pages = unique(str(item.get("page")) for item in record.get("gold_evidence") or [])
        else:
            gold_pages = [record.get("source_pdf_page")] if record.get("source_pdf_page") else []
        if not gold_pages:
            continue
        page_metrics = retrieval_at_k(
            unique(page_from_chunk_id(chunk_id) for chunk_id in route.get("top_chunk_ids") or []),
            [str(page) for page in gold_pages],
            top_k,
        )
        route.setdefault("metrics", {}).update({f"page_{name}": value for name, value in page_metrics.items()})


def rebuild_existing_report(args: argparse.Namespace) -> None:
    result = json.loads(args.report_from.read_text(encoding="utf-8"))
    top_k = int(result.get("run", {}).get("top_k") or args.top_k)
    text_records = _record_map(args.text_dataset)
    image_records = _record_map(args.image_dataset)
    db = SessionLocal()
    try:
        evidence_ids = unique(
            chunk_id
            for row in result.get("text_rows", []) + result.get("image_rows", [])
            for chunk_id in row.get("evidence_chunk_ids") or []
        )
        chunks = db.query(ManualChunk).filter(ManualChunk.chunk_id.in_(evidence_ids)).all() if evidence_ids else []
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        for row in result.get("text_rows", []):
            record = text_records[row["id"]]
            row["predicted_refuse"] = detect_refusal(row.get("answer") or "")
            row["answer_refuse_detected"] = row["predicted_refuse"]
            evidence = [_chunk_as_evidence(chunk_map[chunk_id]) for chunk_id in row.get("evidence_chunk_ids") or [] if chunk_id in chunk_map]
            row["answer_scores"] = score_answer(record, row.get("answer") or "", evidence, row["predicted_refuse"])
            _backfill_route_page_metrics(row, record, top_k)
        for row in result.get("image_rows", []):
            record = image_records[row["id"]]
            row["source_pdf_page"] = record.get("source_pdf_page")
            legacy_non_icon = record.get("negative_type") == "manual_non_icon_figure"
            expected_groups = list(
                record.get("expected_icon_group_ids")
                or (
                    [record.get("expected_icon_group_id")]
                    if record.get("expected_icon_group_id")
                    else []
                )
            )
            visual_gold = (
                record.get("expected_visual_chunk_ids")
                or record.get("relevant_visual_chunk_ids")
                or []
            )
            row["expected_icon_group_id"] = (
                record.get("expected_icon_group_id")
                or (expected_groups[0] if expected_groups else None)
            )
            row["expected_icon_group_ids"] = expected_groups
            row["visual_gold_type"] = record.get("visual_gold_type") or (
                "icon_group"
                if expected_groups
                else "visual_chunk"
                if visual_gold
                else "none"
            )
            row["expected_visual_title"] = record.get("expected_visual_title")
            row["expected_visual_modality"] = record.get("expected_visual_modality")
            row["expected_visual_chunk_ids"] = list(
                record.get("expected_visual_chunk_ids")
                or record.get("relevant_visual_chunk_ids")
                or []
            )
            row["visual_regression_evaluable"] = bool(
                record.get("visual_regression_evaluable", bool(visual_gold))
            )
            row["refusal_evaluable"] = bool(record.get("refusal_evaluable", False))
            row["dataset_compatibility"] = record.get("dataset_compatibility") or (
                "legacy_non_icon_unscored" if legacy_non_icon else "icon_visual_regression"
            )
            row["negative_type"] = record.get("negative_type")
            row["expected_refusal_reason"] = (
                record.get("expected_refusal_reason") or record.get("refusal_reason")
            )
            visual_pages = (
                [record["source_pdf_page"]]
                if row["visual_gold_type"] == "visual_chunk" and record.get("source_pdf_page")
                else None
            )
            for route_collection in (
                row.get("routes", {}),
                row.get("diagnostic_routes", {}),
            ):
                for route in route_collection.values():
                    metrics = retrieval_at_k(
                        route.get("top_keys") or [],
                        visual_gold,
                        top_k,
                    )
                    if visual_pages:
                        page_metrics = retrieval_at_k(
                            route.get("top_pages") or [],
                            visual_pages,
                            top_k,
                        )
                        metrics.update(
                            {f"page_{name}": value for name, value in page_metrics.items()}
                        )
                    route["metrics"] = metrics
            row["guardrail_refuse"] = bool(row.get("guardrail_refuse", row.get("predicted_refuse")))
            row["predicted_refuse"] = row["guardrail_refuse"]
            row["answer_refuse_detected"] = detect_refusal(row.get("answer") or "")
            evidence = [_chunk_as_evidence(chunk_map[chunk_id]) for chunk_id in row.get("evidence_chunk_ids") or [] if chunk_id in chunk_map]
            row["answer_scores"] = (
                {}
                if legacy_non_icon
                else score_answer(
                    record,
                    row.get("answer") or "",
                    evidence,
                    row["guardrail_refuse"] or row["answer_refuse_detected"],
                )
            )
            _backfill_route_page_metrics(row, record, top_k)
    finally:
        db.close()

    result["summary"] = aggregate(result.get("text_rows", []), result.get("image_rows", []), top_k)
    result.setdefault("run", {})["report_rebuilt_at"] = datetime.now(ZoneInfo(os.getenv("TIMEZONE", "Asia/Shanghai"))).isoformat(timespec="seconds")
    write_json_atomic(args.report_from, result)
    latest_json = args.results_dir / f"{args.result_name}_latest.json"
    latest_markdown = args.results_dir / f"{args.result_name}_latest.md"
    write_json_atomic(latest_json, result)
    report = markdown_report(result, top_k)
    if not args.json_only:
        args.report_from.with_suffix(".md").write_text(report, encoding="utf-8")
        latest_markdown.write_text(report, encoding="utf-8")
    print(report)


def _chunk_as_evidence(chunk: ManualChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "page": chunk.page_number,
        "modality": chunk.modality,
        "section_title": chunk.section_title,
        "content_text": chunk.content_text,
        "object_name": chunk.object_name,
        "parent_chunk_id": chunk.parent_chunk_id,
        "is_leaf": chunk.is_leaf,
        "token_count": chunk.token_count,
        "icon_group_id": chunk.icon_group_id,
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.image_timeout_seconds < 0:
        raise ValueError("--image-timeout-seconds must be zero or positive")
    if args.vlm_top_k <= 0:
        raise ValueError("--vlm-top-k must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.result_name):
        raise ValueError("--result-name may contain only letters, numbers, underscores and hyphens")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.report_from:
        rebuild_existing_report(args)
        return
    checkpoint_path = args.results_dir / f"{args.result_name}_checkpoint.json"
    text_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        text_rows = checkpoint.get("text_rows") or []
        image_rows = checkpoint.get("image_rows") or []

    text_records = load_jsonl(args.text_dataset) if args.mode in {"all", "text"} else []
    image_records = load_jsonl(args.image_dataset) if args.mode in {"all", "image"} else []
    if args.image_id:
        requested_ids = set(args.image_id)
        image_records = [record for record in image_records if record.get("id") in requested_ids]
        missing_ids = requested_ids - {record.get("id") for record in image_records}
        if missing_ids:
            raise ValueError(f"Unknown --image-id values: {sorted(missing_ids)}")
    if args.limit_text:
        text_records = text_records[: args.limit_text]
    if args.limit_image:
        image_records = image_records[: args.limit_image]

    timezone = ZoneInfo(os.getenv("TIMEZONE", "Asia/Shanghai"))
    started_at = datetime.now(timezone)
    print(f"Starting evaluation: text={len(text_records)}, image={len(image_records)}", flush=True)
    completed_text = {row["id"] for row in text_rows}
    text_db = SessionLocal()
    try:
        for index, record in enumerate(text_records, start=1):
            if record["id"] in completed_text:
                continue
            try:
                row = evaluate_text_record(
                    text_db,
                    record,
                    args.top_k,
                    args.skip_generation,
                    args.text_final_route,
                )
            except Exception as error:
                text_db.rollback()
                row = {
                    "id": record["id"],
                    "kind": "text",
                    "category": record.get("category"),
                    "should_refuse": bool(record.get("should_refuse")),
                    "predicted_refuse": False,
                    "routes": {},
                    "answer": "",
                    "answer_scores": {},
                    "fatal_error": f"{type(error).__name__}: {error}",
                }
            text_rows.append(row)
            save_checkpoint(checkpoint_path, text_rows, image_rows)
            print(f"[text {index}/{len(text_records)}] {record['id']} {row.get('elapsed_seconds', 0):.1f}s", flush=True)
    finally:
        text_db.close()

    # The default image path mirrors production: first-pass understanding on,
    # candidate verification and OpenCV off, alpha-white vectors and equal RRF.
    # The legacy ``--skip-vlm`` still disables both VLM stages for old commands.
    first_pass_vlm = not args.skip_first_pass_vlm and not args.skip_vlm
    candidate_vlm = args.enable_candidate_vlm and not args.skip_vlm
    use_opencv = (
        args.enable_opencv_ablation
        or args.visual_fusion_profile == "legacy_hard_gate"
        or args.visual_fusion_profile.endswith("_opencv")
    )

    db = SessionLocal()
    try:
        completed_image = {row["id"] for row in image_rows}
        for index, record in enumerate(image_records, start=1):
            if record["id"] in completed_image:
                continue
            record_started = time.monotonic()
            try:
                with image_evaluation_timeout(args.image_timeout_seconds):
                    row = evaluate_image_record(
                        db,
                        record,
                        args.image_dataset,
                        args.top_k,
                        args.skip_generation,
                        first_pass_vlm,
                        candidate_vlm,
                        use_opencv,
                        args.image_vector_variant,
                        args.visual_fusion_profile,
                        args.skip_text_support,
                        args.vlm_top_k,
                    )
            except (ImageEvaluationTimeout, Exception) as error:
                db.rollback()
                row = {
                    "id": record["id"],
                    "kind": "image",
                    "category": record.get("category"),
                    "variant": record.get("variant") or "negative",
                    "should_refuse": bool(record.get("should_refuse")),
                    "predicted_refuse": False,
                    "routes": {},
                    "answer": "",
                    "answer_scores": {},
                    "fatal_error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": round(time.monotonic() - record_started, 3),
                }
            image_rows.append(row)
            save_checkpoint(checkpoint_path, text_rows, image_rows)
            print(f"[image {index}/{len(image_records)}] {record['id']} {row.get('elapsed_seconds', 0):.1f}s", flush=True)
    finally:
        db.close()

    finished_at = datetime.now(timezone)
    result = {
        "run": {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "mode": args.mode,
            "top_k": args.top_k,
            "skip_generation": args.skip_generation,
            "skip_vlm": args.skip_vlm,
            "first_pass_vlm": first_pass_vlm,
            "candidate_vlm": candidate_vlm,
            "opencv_ablation": use_opencv,
            "skip_text_support": args.skip_text_support,
            "image_vector_variant": args.image_vector_variant,
            "visual_fusion_profile": args.visual_fusion_profile,
            "vlm_top_k": args.vlm_top_k,
            "image_timeout_seconds": args.image_timeout_seconds,
            "text_final_route": args.text_final_route,
            "text_dataset": str(args.text_dataset),
            "image_dataset": str(args.image_dataset),
            "text_rerank_model": next(
                (row.get("cross_encoder_model") for row in text_rows if row.get("cross_encoder_model")),
                None,
            ),
            "answer_model": os.getenv("LLM_MODEL", "deepseek-r1") if not args.skip_generation else None,
        },
        "summary": aggregate(text_rows, image_rows, args.top_k),
        "text_rows": text_rows,
        "image_rows": image_rows,
    }
    timestamp = finished_at.strftime("%Y%m%d_%H%M%S")
    json_path = args.results_dir / f"{args.result_name}_{timestamp}.json"
    markdown_path = args.results_dir / f"{args.result_name}_{timestamp}.md"
    latest_json = args.results_dir / f"{args.result_name}_latest.json"
    latest_markdown = args.results_dir / f"{args.result_name}_latest.md"
    if not args.latest_only:
        write_json_atomic(json_path, result)
    write_json_atomic(latest_json, result)
    report = markdown_report(result, args.top_k)
    if not args.json_only:
        if not args.latest_only:
            markdown_path.write_text(report, encoding="utf-8")
        latest_markdown.write_text(report, encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(report, flush=True)
    print(
        json.dumps(
            {
                "json": str(latest_json if args.latest_only else json_path),
                "markdown": None if args.json_only else str(latest_markdown if args.latest_only else markdown_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; rerun with --resume to continue from checkpoint.", file=sys.stderr)
        raise
