"""Judge image-QA groundedness: original answers vs reflection-loop answers.

This is an ADD-ON evaluation script.  It does NOT modify the serving pipeline.

For each answerable image sample it reuses a saved original answer + evidence
and compares, with the SAME independent LLM judge:

  - baseline: groundedness of the ORIGINAL answer
  - agent:    groundedness of the answer after a bounded
              critic -> regenerate / refuse reflection loop

Design notes
------------
* The critic and the judge default to ``qwen-plus`` (DASHSCOPE) because the
  GPT quota is exhausted.  The regeneration generator reuses the saved
  ``answer_model`` (deepseek) so critic != generator.
* The reflection loop never re-retrieves and never checks gold: it only
  checks whether each claim in the draft answer is entailed by the evidence
  it was given, then deletes / rewrites / refuses accordingly (bounded).
* The script is resumable via a per-sample JSONL next to the markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, "/app")

from openai import OpenAI

from evaluate_owner_manual import (
    _build_image_answer_question,
    _build_insufficient_text_answer,
    _build_visual_anchor,
)
from evaluate_cached_image_qa import (
    clean,
    judge_one,
    load_jsonl,
    minimal_vision,
    visual_anchor_from_row,
)
from service.answer_generation_service import build_grounded_prompt


# --------------------------------------------------------------------------
#  Small helpers
# --------------------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    value = clean(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.S)
    if fenced:
        value = fenced.group(1)
    else:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    return json.loads(value)


def evidence_blocks(evidence: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence or [], start=1):
        blocks.append(
            f"[证据{index}] page={item.get('page')} "
            f"section={clean(item.get('section_title'))} "
            f"role={item.get('evidence_role') or item.get('modality')}\n"
            f"{clean(item.get('content_text'))}"
        )
    return "\n\n".join(blocks) or "<没有返回证据>"


# --------------------------------------------------------------------------
#  Critic (reflection loop) and regeneration
# --------------------------------------------------------------------------

def critic_prompt(question: str, answer: str, evidence: list[dict[str, Any]]) -> str:
    return f"""你是车主手册图片问答答案的忠实度检查器。你只能依据给定证据判断，禁止使用手册之外的知识。
用户问题：{clean(question)}
待检查答案：{clean(answer)}
候选证据：
{evidence_blocks(evidence)}

任务：
1. 把答案拆成原子断言（每条只含一个事实，不要合并多个事实）。
2. 逐条判断该断言是否被给定证据蕴含：
   - supported：证据直接支持
   - unsupported：证据中找不到对应支撑
   - contradicted：证据明确与之矛盾
   每条必须给出 evidence_ref（如 "[证据2]"）或 null。
3. 汇总：
   - passes：是否全部 supported
   - refusal_required：核心结论是否无证据支撑、应当拒答而不是继续猜
   - feedback：给重写建议，列出需要删除或修改的断言

只输出 JSON，不要 Markdown：
{{
  "claims": [
    {{"claim": "…", "verdict": "supported", "evidence_ref": "[证据2]"}}
  ],
  "overall": {{"passes": true, "refusal_required": false, "feedback": "…"}}
}}"""


def critic(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    client: OpenAI,
    model: str,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": critic_prompt(question, answer, evidence)}],
        temperature=0,
        stream=False,
    )
    raw = extract_json(completion.choices[0].message.content or "")
    claims = raw.get("claims") or []
    overall = raw.get("overall") or {}
    return {
        "claims": [
            {
                "claim": clean(item.get("claim")),
                "verdict": str(item.get("verdict")),
                "evidence_ref": clean(item.get("evidence_ref")),
                "note": clean(item.get("note")),
            }
            for item in claims
        ],
        "overall": {
            "passes": bool(overall.get("passes")),
            "refusal_required": bool(overall.get("refusal_required")),
            "feedback": clean(overall.get("feedback")),
        },
    }


def feedback_instruction(feedback: str) -> str:
    return (
        "上一版草稿经忠实度检查发现以下问题，请重写：\n"
        f"- {clean(feedback)}\n"
        "要求：只保留证据支持的结论；证据没有支持的，明确写“手册中无法确认”；"
        "若核心结论证据不足，直接拒答而不是猜测。"
    )


def generate_with_feedback(
    answer_question: str,
    evidence: list[dict[str, Any]],
    feedback: str,
    client: OpenAI,
    model: str,
) -> str:
    base = build_grounded_prompt(answer_question, evidence)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": base + "\n\n" + feedback_instruction(feedback),
            }
        ],
        temperature=0.2,
        stream=False,
    )
    return clean(completion.choices[0].message.content)


def image_answer_question(row: dict[str, Any]) -> str:
    vision = row.get("vision") or minimal_vision(row)
    anchor = row.get("visual_match") or visual_anchor_from_row(row)
    return _build_image_answer_question(
        vision,
        row.get("question") or "这张图片是什么，应该如何理解或使用？",
        row.get("text_query") or row.get("question") or "",
        anchor,
    )


def refusal_answer(row: dict[str, Any]) -> str:
    anchor = row.get("visual_match") or visual_anchor_from_row(row)
    try:
        return _build_insufficient_text_answer(anchor)
    except Exception:
        return "当前手册中未检索到足以支持该结论的证据，无法可靠确认。"


def reflect(
    row: dict[str, Any],
    critic_client: OpenAI,
    critic_model: str,
    gen_client: OpenAI,
    gen_model: str,
    max_rounds: int = 2,
) -> tuple[str, dict[str, Any]]:
    question = row.get("question") or ""
    answer = clean(row.get("answer"))
    evidence = row.get("evidence") or []
    answer_question = image_answer_question(row)
    trace: dict[str, Any] = {
        "regenerated": False,
        "refused": False,
        "rounds": 0,
        "claims": 0,
        "unsupported": 0,
        "contradicted": 0,
        "critique_feedback": "",
    }
    final = answer
    for round_index in range(max_rounds):
        verdict = critic(question, final, evidence, critic_client, critic_model)
        claims = verdict["claims"]
        trace["claims"] = len(claims)
        trace["unsupported"] += sum(1 for item in claims if item["verdict"] == "unsupported")
        trace["contradicted"] += sum(1 for item in claims if item["verdict"] == "contradicted")
        if verdict["overall"]["passes"] or not claims:
            break
        trace["critique_feedback"] = verdict["overall"]["feedback"]
        if verdict["overall"]["refusal_required"]:
            trace["refused"] = True
            trace["rounds"] = round_index + 1
            final = refusal_answer(row)
            break
        final = generate_with_feedback(
            answer_question,
            evidence,
            verdict["overall"]["feedback"],
            gen_client,
            gen_model,
        )
        trace["regenerated"] = True
        trace["rounds"] = round_index + 1
    return final, trace


# --------------------------------------------------------------------------
#  One sample: baseline judge + reflection + agent judge
# --------------------------------------------------------------------------

def process_row(
    row: dict[str, Any],
    record: dict[str, Any],
    args: argparse.Namespace,
    critic_client: OpenAI,
    gen_client: OpenAI,
) -> dict[str, Any]:
    baseline_row = dict(row)
    baseline_row["answer"] = clean(row.get("answer"))
    baseline = judge_one(record, baseline_row, args.judge_model)

    final, trace = reflect(
        row,
        critic_client,
        args.critic_model,
        gen_client,
        args.gen_model,
        args.max_rounds,
    )
    agent_row = dict(row)
    agent_row["answer"] = final
    agent = judge_one(record, agent_row, args.judge_model)

    return {
        "id": row["id"],
        "variant": row.get("variant") or row.get("category") or "unknown",
        "question": row.get("question"),
        "baseline_answer": baseline_row["answer"],
        "agent_answer": final,
        "changed": baseline_row["answer"] != final,
        "baseline_judge": baseline,
        "agent_judge": agent,
        "trace": trace,
    }


# --------------------------------------------------------------------------
#  Report
# --------------------------------------------------------------------------

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(values: list[float]) -> str:
    return f"{mean(values) * 100:.2f}%"


def signed_pct(values: list[float]) -> str:
    return f"{mean(values) * 100:+.2f}%"


def build_report(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    source_cached_result: Path,
) -> str:
    baseline_g = [item["baseline_judge"]["groundedness"] for item in results]
    agent_g = [item["agent_judge"]["groundedness"] for item in results]
    baseline_c = [item["baseline_judge"]["answer_correctness"] for item in results]
    agent_c = [item["agent_judge"]["answer_correctness"] for item in results]
    changed = sum(1 for item in results if item["changed"])
    regenerated = sum(1 for item in results if item["trace"]["regenerated"])
    refused = sum(1 for item in results if item["trace"]["refused"])
    critiqued = sum(1 for item in results if item["trace"]["unsupported"] or item["trace"]["contradicted"])

    variants = sorted({item["variant"] for item in results})
    lines: list[str] = []
    lines.append("# 图片问答答案忠实度：原始 vs 反射循环（Groundedness A/B）\n")
    lines.append(f"- 数据来源：`{source_cached_result}`")
    lines.append(f"- 可回答样本：{len(results)}")
    lines.append(f"- 裁判模型：`{args.judge_model}`（qwen，DASHSCOPE；GPT 额度已用完，非 GPT-5.6 官方口径）")
    lines.append(f"- Critic（反射检查器）模型：`{args.critic_model}`")
    lines.append(f"- 重生成模型：`{args.gen_model}`（与原答案生成模型一致，critic ≠ 生成器）")
    lines.append(f"- 最大反思轮数：{args.max_rounds}")
    lines.append("")
    lines.append("## 总览\n")
    lines.append("| 指标 | 原始答案 | 反射循环后 | 差值 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Groundedness | {pct(baseline_g)} | {pct(agent_g)} | {signed_pct([a - b for a, b in zip(agent_g, baseline_g)])} |")
    lines.append(f"| Answer Correctness | {pct(baseline_c)} | {pct(agent_c)} | {signed_pct([a - b for a, b in zip(agent_c, baseline_c)])} |")
    lines.append("")
    lines.append("## 反射循环统计\n")
    lines.append("| 统计项 | 值 |")
    lines.append("|---|---:|")
    lines.append(f"| 答案被改动 | {changed} / {len(results)} |")
    lines.append(f"| 触发重生成 | {regenerated} / {len(results)} |")
    lines.append(f"| 触发拒答（核心结论无支撑） | {refused} / {len(results)} |")
    lines.append(f"| 被指出 unsupported/contradicted 断言 | {critiqued} / {len(results)} |")
    lines.append(f"| 平均拆出断言数 | {mean([item['trace']['claims'] for item in results]):.2f} |")
    lines.append(f"| 平均 unsupported 数 | {mean([item['trace']['unsupported'] for item in results]):.2f} |")
    lines.append(f"| 平均 contradicted 数 | {mean([item['trace']['contradicted'] for item in results]):.2f} |")
    lines.append(f"| 平均反思轮数 | {mean([item['trace']['rounds'] for item in results]):.2f} |")
    lines.append("")
    lines.append("## 分变体\n")
    lines.append("| 变体 | 样本 | 原始 Groundedness | 反射后 Groundedness | 差值 |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant in variants:
        items = [item for item in results if item["variant"] == variant]
        g0 = [item["baseline_judge"]["groundedness"] for item in items]
        g1 = [item["agent_judge"]["groundedness"] for item in items]
        lines.append(
            f"| {variant} | {len(items)} | {pct(g0)} | {pct(g1)} | "
            f"{signed_pct([a - b for a, b in zip(g1, g0)])} |"
        )
    lines.append("")
    lines.append("## 逐条结果\n")
    lines.append("| ID | 变体 | 原始G | 反射G | Δ | 改动 | 重生成 | 拒答 | 断言 |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|---:|")
    for item in sorted(results, key=lambda r: r["id"]):
        g0 = item["baseline_judge"]["groundedness"]
        g1 = item["agent_judge"]["groundedness"]
        lines.append(
            f"| {item['id']} | {item['variant']} | {g0:.1f} | {g1:.1f} | "
            f"{g1 - g0:+.1f} | {'是' if item['changed'] else ''} | "
            f"{'是' if item['trace']['regenerated'] else ''} | "
            f"{'是' if item['trace']['refused'] else ''} | "
            f"{item['trace']['claims']} |"
        )
    lines.append("")
    lines.append("## 说明与诚实边界\n")
    lines.append("- 反射循环只做答案 vs 证据的相对检查，不重检索、不查金标；它压制无证据断言，不提升答案正确率。")
    lines.append(f"- 裁判为 `{args.judge_model}`，非项目文本侧使用的 GPT-5.6 独立裁判；本数字用于同口径 A/B 对比，不能与 GPT-5.6 官方分数直接比较。")
    lines.append("- 样本为可回答图片问答（should_refuse=False）；拒答样本不参与 Groundedness 评分（与现有图片评测口径一致）。")
    return "\n".join(lines)


# --------------------------------------------------------------------------
#  Entry point
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL") or "qwen-plus")
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--gen-model", default=None)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached = json.loads(args.cached_result.read_text(encoding="utf-8"))
    records = {item["id"]: item for item in load_jsonl(args.dataset)}

    gen_model = args.gen_model or cached.get("run", {}).get("answer_model") or os.getenv("LLM_MODEL", "deepseek-r1")
    args.gen_model = gen_model
    args.critic_model = args.critic_model or args.judge_model

    rows = [dict(item) for item in cached.get("image_rows") or []]
    rows = [item for item in rows if item.get("id") in records]
    rows = [item for item in rows if not item.get("should_refuse")]
    rows = [item for item in rows if clean(item.get("answer")) and (item.get("evidence"))]
    rows.sort(key=lambda item: str(item.get("id")))
    if args.limit:
        rows = rows[: args.limit]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "judge_model": args.judge_model,
                "critic_model": args.critic_model,
                "gen_model": gen_model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Critic / judge run on DASHSCOPE (qwen-plus); the generator runs on the
    # deepseek endpoint that produced the original answers (deepseek-v4-flash).
    dash_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    dash_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("LLM_BASE_URL")
    if not dash_key:
        raise RuntimeError("DASHSCOPE_API_KEY/LLM_API_KEY is not configured")
    llm_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    llm_url = os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
    critic_client = OpenAI(api_key=dash_key, base_url=dash_url)
    gen_client = OpenAI(api_key=llm_key, base_url=llm_url)

    samples_path = args.output.with_suffix(".samples.jsonl")
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    done = {item["id"]: item for item in load_jsonl(samples_path)} if samples_path.exists() else {}
    pending = [item for item in rows if item["id"] not in done]
    print(f"completed={len(done)} pending={len(pending)}", flush=True)

    results: dict[str, dict[str, Any]] = dict(done)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(process_row, row, records[row["id"]], args, critic_client, gen_client): row["id"]
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result["id"]] = result
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{len(done) + index}/{len(rows)}] {result['id']} "
                f"G={result['baseline_judge']['groundedness']:.1f}->{result['agent_judge']['groundedness']:.1f} "
                f"regen={result['trace']['regenerated']} refused={result['trace']['refused']}",
                flush=True,
            )

    ordered = [results[row["id"]] for row in rows if row["id"] in results]
    args.output.write_text(
        build_report(ordered, args, args.cached_result),
        encoding="utf-8",
    )
    print(f"report written: {args.output}", flush=True)


if __name__ == "__main__":
    main()
