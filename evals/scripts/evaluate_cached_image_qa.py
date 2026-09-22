"""Run image-QA downstream stages from a saved visual-retrieval result.

This deliberately does not call image retrieval or VLM vision.  It reuses the
saved visual anchor and saved post-anchor query, then runs body-text retrieval,
answer generation and an independent LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from evaluate_owner_manual import (
    _build_image_answer_question,
    _build_image_text_query,
    _build_insufficient_text_answer,
    _build_visual_anchor,
    detect_refusal,
    resolve_image_path,
    score_answer,
    unique,
    write_json_atomic,
)
from router.qa_rt import _should_refuse_image_answer
from service.answer_generation_service import complete_answer
from service.retrieval_service import retrieve_visual_anchor_text
from service.vision_understanding_service import understand_vehicle_image
from utils.database import SessionLocal


SCORE_GRID = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--id-prefix", default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--anchor-query",
        action="store_true",
        help="Append cached visual object_name/section_title to the saved query.",
    )
    parser.add_argument(
        "--object-name-query",
        action="store_true",
        help="Build text query only from the cached object_name and the user's intent; do not use VLM fields or section_title.",
    )
    parser.add_argument(
        "--ignore-cached-guardrail",
        action="store_true",
        help="Treat the cached fusion Top1 as the visual anchor and discard stale VLM/guardrail state.",
    )
    parser.add_argument(
        "--rerun-first-pass-vlm",
        action="store_true",
        help="Rerun only first-pass VLM understanding; visual retrieval stays cached.",
    )
    return parser.parse_args()


def minimal_vision(row: dict[str, Any]) -> dict[str, Any]:
    """Use only first-pass fields persisted by the visual evaluator."""

    if row.get("vision"):
        return dict(row["vision"])
    return {
        "recognized_object": row.get("recognized_object") or "unknown",
        "visual_description": "",
        "ocr_text": "",
        "candidate_names": [],
        "visual_keywords": [],
        "normalized_query": row.get("text_query") or "",
        "risk_level": row.get("visual_match", {}).get("risk_level") or "unknown",
    }


def visual_anchor_from_row(row: dict[str, Any]) -> dict[str, Any]:
    anchor = dict(row.get("visual_match") or {})
    if not anchor:
        candidates = row.get("route_candidates", {}).get("full_chain_visual") or []
        anchor = dict(candidates[0]) if candidates else {}
    return _build_visual_anchor(anchor)


def fetch_text_support(
    row: dict[str, Any],
    anchor_query: bool = False,
    object_name_query: bool = False,
) -> dict[str, Any]:
    if row.get("should_refuse"):
        return {"text_evidence": [], "text_query": row.get("text_query") or "", "error": None}
    db = SessionLocal()
    try:
        anchor = visual_anchor_from_row(row)
        vision = row.get("vision") or {}
        if object_name_query:
            query = " ".join(
                part
                for part in (
                    clean(anchor.get("object_name")),
                    "含义 亮起 怎么办 处理方法",
                )
                if part
            )
        elif vision:
            query = _build_image_text_query(
                vision,
                row.get("question"),
                anchor,
            )
        else:
            query_parts = [clean(row.get("text_query")) or clean(row.get("question"))]
            if anchor_query:
                query_parts.extend([
                    clean(anchor.get("object_name")),
                    clean(anchor.get("section_title")),
                ])
            query = " ".join(dict.fromkeys(part for part in query_parts if part))
        evidence = retrieve_visual_anchor_text(
            db,
            row.get("manual_id") or "demo_manual",
            query,
            anchor,
            top_k=4,
        )
        return {"text_evidence": evidence[:4], "text_query": query, "error": None}
    except Exception as error:
        db.rollback()
        return {"text_evidence": [], "text_query": row.get("text_query") or "", "error": f"{type(error).__name__}: {error}"}
    finally:
        db.close()


def generate_answer(row: dict[str, Any]) -> dict[str, Any]:
    visual_match = row.get("visual_match") or visual_anchor_from_row(row)
    vision = minimal_vision(row)
    question = row.get("question") or "这张图片是什么，应该如何理解或使用？"
    evidence = list(row.get("evidence") or [])
    if row.get("guardrail_refuse") or row.get("should_refuse"):
        answer = _build_insufficient_text_answer(visual_match)
    elif not evidence:
        answer = _build_insufficient_text_answer(visual_match)
    else:
        answer_question = _build_image_answer_question(
            vision,
            question,
            row.get("text_query") or question,
            visual_match,
        )
        answer = complete_answer(answer_question, evidence)
    answer_refuse = detect_refusal(answer)
    final_refuse = bool(row.get("guardrail_refuse")) or answer_refuse
    scores = score_answer(row, answer, evidence, final_refuse) if not row.get("should_refuse") else {}
    row["answer"] = answer
    row["answer_refuse_detected"] = answer_refuse
    row["predicted_refuse"] = final_refuse
    row["evidence_status"] = (
        "expected_refusal"
        if row.get("should_refuse") and answer_refuse
        else (
            "manual_visual_not_found"
            if row.get("guardrail_refuse")
            else ("insufficient_text_evidence" if not evidence else "verified")
        )
    )
    row["answer_scores"] = scores
    row["evidence_chunk_ids"] = [item.get("chunk_id") for item in evidence]
    return row


def evidence_for_judge(row: dict[str, Any]) -> str:
    blocks = []
    for index, item in enumerate(row.get("evidence") or [], start=1):
        blocks.append(
            f"[证据{index}] page={item.get('page')} section={clean(item.get('section_title'))}\n"
            f"{clean(item.get('content_text'))}"
        )
    return "\n\n".join(blocks) or "<没有返回证据>"


def judge_prompt(record: dict[str, Any], row: dict[str, Any]) -> str:
    claims = record.get("reference_claims") or [record.get("reference_answer") or ""]
    return f"""你是独立的车主手册图片问答评测员。只输出 JSON，不要 Markdown。
问题：{clean(record.get('question'))}
参考答案：{clean(record.get('reference_answer'))}
原子必答点：{json.dumps(claims, ensure_ascii=False)}
系统回答：{clean(row.get('answer')) or '<空回答>'}
最终证据：
{evidence_for_judge(row)}

请评分（只能取 0、0.2、0.4、0.6、0.8、1.0）：
- answer_correctness：回答是否正确并覆盖必答点。
- groundedness：回答中的事实是否都被最终证据支持。
- answer_execution：是否直接回答问题、没有无依据编造，并给出可执行的手册建议。
输出：{{"answer_correctness":0.0,"groundedness":0.0,"answer_execution":0.0,"correctness_reason":"一句话","groundedness_reason":"一句话","execution_reason":"一句话"}}"""


def parse_json(text: str) -> dict[str, Any]:
    value = clean(text)
    match = re.search(r"\{.*\}", value, re.S)
    if match:
        value = match.group(0)
    return json.loads(value)


def judge_one(record: dict[str, Any], row: dict[str, Any], model: str) -> dict[str, Any]:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY/LLM_API_KEY is not configured")
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt(record, row)}],
        temperature=0,
        stream=False,
    )
    raw = parse_json(completion.choices[0].message.content or "")
    result = {"id": row["id"], "judge_model": model}
    for field in ("answer_correctness", "groundedness", "answer_execution"):
        score = float(raw.get(field))
        if score not in SCORE_GRID:
            raise ValueError(f"{row['id']}.{field} invalid score={score}")
        result[field] = score
    for field in ("correctness_reason", "groundedness_reason", "execution_reason"):
        result[field] = clean(raw.get(field))
    return result


def main() -> None:
    args = parse_args()
    cached = json.loads(args.cached_result.read_text(encoding="utf-8"))
    records = {row["id"]: row for row in load_jsonl(args.dataset)}
    rows = [dict(row) for row in cached.get("image_rows") or []]
    rows = [row for row in rows if row.get("id") in records]
    if args.id_prefix:
        rows = [row for row in rows if str(row.get("id") or "").startswith(args.id_prefix)]
    for row in rows:
        record = records[row["id"]]
        # The saved visual evaluator intentionally omitted question/text-gold
        # fields; copy them only for this downstream answer/judge run.
        for field in (
            "manual_id",
            "question",
            "reference_answer",
            "reference_claims",
            "should_refuse",
            "refusal_evaluable",
            "expected_refusal_reason",
        ):
            row[field] = record.get(field)
        if args.object_name_query:
            # The no-VLM arm must not leak cached first-pass descriptions or
            # OCR into either retrieval or the final answer prompt.
            row.pop("vision", None)
        if args.ignore_cached_guardrail:
            # This downstream-only experiment reuses the already selected
            # fusion Top1.  Historical VLM decisions in the source cache must
            # not be allowed to reject that anchor or force an answer refusal.
            row["guardrail_refuse"] = False
            row["predicted_refuse"] = False
            row["text_support_skipped"] = False
            visual_match = dict(row.get("visual_match") or {})
            visual_match["matched"] = bool(visual_match)
            visual_match["retrieval_anchor"] = bool(visual_match)
            for field in ("vlm_verified", "vlm_confidence", "vlm_reason"):
                visual_match.pop(field, None)
            row["visual_match"] = visual_match
    print(f"cached visual rows={len(rows)}; no visual retrieval calls")

    if args.rerun_first_pass_vlm:
        def run_vision(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            record = records[row["id"]]
            image_path = resolve_image_path(record["image_path"], args.dataset)
            return row["id"], understand_vehicle_image(
                image_path,
                record.get("question") or "这张图片是什么，应该如何理解或使用？",
            )

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(run_vision, row): row["id"] for row in rows}
            vision_map: dict[str, dict[str, Any]] = {}
            for index, future in enumerate(as_completed(futures), start=1):
                case_id, vision = future.result()
                vision_map[case_id] = vision
                print(f"[vision {index}/{len(rows)}] {case_id}", flush=True)
        for row in rows:
            row["vision"] = vision_map[row["id"]]
        print("first-pass VLM refreshed; visual retrieval remains cached", flush=True)

    # Text retrieval is the only database stage.  It is intentionally done
    # before generation so the answer calls can run in parallel safely.
    for index, row in enumerate(rows, start=1):
        support = fetch_text_support(
            row,
            anchor_query=args.anchor_query,
            object_name_query=args.object_name_query,
        )
        row["text_query"] = support["text_query"]
        text_evidence = support["text_evidence"]
        visual_anchor = visual_anchor_from_row(row)
        row["evidence"] = ([{**visual_anchor, "evidence_role": "visual"}] if visual_anchor else []) + [
            {**item, "evidence_role": "text"} for item in text_evidence
        ]
        row["text_support_route"] = {
            "top_chunk_ids": [item.get("chunk_id") for item in text_evidence],
            "metrics": {},
        }
        row["generation_error"] = support.get("error")
        print(f"[text {index}/{len(rows)}] {row['id']} evidence={len(text_evidence)}", flush=True)

    if args.retrieval_only:
        summary = {
            "source_cached_result": str(args.cached_result),
            "visual_retrieval_rerun": False,
            "vlm_rerun": False,
            "text_support_rerun": True,
            "answer_generation_rerun": False,
            "judge_rerun": False,
            "anchor_query": args.anchor_query,
            "object_name_query": args.object_name_query,
            "ignore_cached_guardrail": args.ignore_cached_guardrail,
            "first_pass_vlm_rerun": args.rerun_first_pass_vlm,
            "count": len(rows),
            "answerable_count": sum(not row.get("should_refuse") for row in rows),
            "text_evidence_nonempty": sum(
                bool(row.get("text_support_route", {}).get("top_chunk_ids"))
                for row in rows
                if not row.get("should_refuse")
            ),
        }
        result = {"run": summary, "summary": {"image": summary}, "image_rows": rows}
        write_json_atomic(args.output, result)
        args.output.with_suffix(".md").write_text(
            "# Cached VLM 图片问答：正文召回阶段\n\n"
            "本文件只完成正文召回，未调用答案模型或 judge。\n\n"
            f"- 样本：{len(rows)}\n- 可回答：{summary['answerable_count']}\n"
            f"- 正文证据非空：{summary['text_evidence_nonempty']}\n",
            encoding="utf-8",
        )
        return

    answerable = [row for row in rows if not row.get("should_refuse")]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(generate_answer, row): row["id"] for row in answerable}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            print(f"[answer {index}/{len(answerable)}] {row['id']}", flush=True)
    for row in rows:
        if row.get("should_refuse"):
            generate_answer(row)

    judge_model = args.judge_model or os.getenv("EVAL_JUDGE_MODEL") or "qwen-plus"
    judgments: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(judge_one, records[row["id"]], row, judge_model): row["id"]
            for row in answerable
        }
        for index, future in enumerate(as_completed(futures), start=1):
            judgment = future.result()
            judgments.append(judgment)
            print(f"[judge {index}/{len(answerable)}] {judgment['id']}", flush=True)
    judgments.sort(key=lambda item: item["id"])
    args.judge_output.parent.mkdir(parents=True, exist_ok=True)
    args.judge_output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in judgments),
        encoding="utf-8",
    )
    judge_map = {item["id"]: item for item in judgments}
    summary = {
        "source_cached_result": str(args.cached_result),
        "visual_retrieval_rerun": False,
        "vlm_rerun": False,
        "text_support_rerun": True,
        "answer_generation_rerun": True,
        "judge_model": judge_model,
        "count": len(rows),
        "answerable_count": len(answerable),
        "refusal_count": sum(bool(row.get("should_refuse")) for row in rows),
        "refusal_correct": sum(
            bool(row.get("should_refuse")) and bool(row.get("predicted_refuse"))
            for row in rows
        ),
        "text_evidence_nonempty": sum(bool(row.get("text_support_route", {}).get("top_chunk_ids")) for row in answerable),
        "answer_correctness": sum(item["answer_correctness"] for item in judgments) / len(judgments) if judgments else 0.0,
        "groundedness": sum(item["groundedness"] for item in judgments) / len(judgments) if judgments else 0.0,
        "answer_execution": sum(item["answer_execution"] for item in judgments) / len(judgments) if judgments else 0.0,
    }
    result = {
        "run": {
            "mode": "cached_visual_downstream_only",
            "source_cached_result": str(args.cached_result),
            "visual_retrieval_rerun": False,
            "vlm_rerun": False,
            "anchor_query": args.anchor_query,
            "object_name_query": args.object_name_query,
            "ignore_cached_guardrail": args.ignore_cached_guardrail,
            "first_pass_vlm_rerun": args.rerun_first_pass_vlm,
            "answer_model": os.getenv("LLM_MODEL", "deepseek-r1"),
            "judge_model": judge_model,
            "text_support_top_k": 4,
        },
        "summary": {"image": {"llm_judge": summary}},
        "image_rows": rows,
    }
    write_json_atomic(args.output, result)
    args.output.with_suffix(".md").write_text(
        "# Cached VLM 图片问答下游测试\n\n"
        f"- 视觉召回：复用 `{args.cached_result}`，未重新召回。\n"
        f"- VLM：未重新调用，复用已保存的视觉锚点和 text query。\n"
        f"- 样本：{len(rows)}；可回答：{len(answerable)}；正文证据非空：{summary['text_evidence_nonempty']}。\n"
        f"- 答案模型：`{result['run']['answer_model']}`；独立 judge：`{judge_model}`。\n\n"
        "| 指标 | 结果 |\n|---|---:|\n"
        f"| Answer Correctness | {summary['answer_correctness'] * 100:.2f}% |\n"
        f"| Groundedness | {summary['groundedness'] * 100:.2f}% |\n"
        f"| Answer Execution | {summary['answer_execution'] * 100:.2f}% |\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
