"""Build summary and case-by-case audit for cached downstream image QA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def clean(value) -> str:
    return " ".join(str(value or "").split())


def md(value, limit=1200) -> str:
    value = clean(value).replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def pct(value) -> str:
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    judgments = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.judgments.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = result.get("image_rows") or []
    answerable = [row for row in rows if not row.get("should_refuse")]
    scored = [judgments[row["id"]] for row in answerable if row["id"] in judgments]
    first_pass_vlm_rerun = bool(result.get("run", {}).get("first_pass_vlm_rerun"))
    anchor_query = bool(result.get("run", {}).get("anchor_query"))
    object_name_query = bool(result.get("run", {}).get("object_name_query"))
    ignore_cached_guardrail = bool(result.get("run", {}).get("ignore_cached_guardrail"))
    summary = result.get("summary", {}).get("image", {})
    if "llm_judge" in summary:
        summary = summary["llm_judge"]
    mismatch = []
    for row in answerable:
        record = records[row["id"]]
        expected = clean(record.get("expected_icon") or record.get("expected_visual_title"))
        object_name = clean((row.get("visual_match") or {}).get("object_name"))
        query = clean(row.get("text_query"))
        if expected and expected not in query:
            mismatch.append((row["id"], expected, object_name, query))

    def group_metrics(prefix: str) -> tuple[int, float, float, float, int]:
        selected = [row for row in answerable if row["id"].startswith(prefix)]
        selected_judgments = [judgments[row["id"]] for row in selected if row["id"] in judgments]
        text_hits = 0
        for row in selected:
            record = records[row["id"]]
            gold = set(record.get("relevant_text_chunk_ids") or [])
            for unit in record.get("text_evidence_units") or []:
                if isinstance(unit, dict):
                    gold.update(unit.get("acceptable_chunk_ids") or [])
            returned = {item.get("chunk_id") for item in row.get("evidence") or []}
            text_hits += bool(gold & returned)
        if not selected_judgments:
            return len(selected), 0.0, 0.0, 0.0, text_hits
        return (
            len(selected),
            mean(item["answer_correctness"] for item in selected_judgments),
            mean(item["groundedness"] for item in selected_judgments),
            mean(item["answer_execution"] for item in selected_judgments),
            text_hits,
        )

    grouped = [
        ("普通图标", *group_metrics("image_pos_")),
        ("操作图", *group_metrics("image_operation_")),
        ("Strong 实拍", *group_metrics("image_strong_")),
    ]

    summary_md = [
        "# Cached VLM 图片问答下游测试汇总",
        "",
        "本轮复用已保存的视觉召回和视觉锚点，没有重新执行图片向量、OpenCV、RRF 或 VLM 二验。",
        (
            "第一次 VLM 已重新执行；随后使用当前 `_build_image_text_query()` 构造 query，并重新执行正文召回、答案生成和 Judge。"
            if first_pass_vlm_rerun
            else (
                "第一次 VLM 未执行；正文 query 仅由视觉锚点 `object_name + 含义/亮起/怎么办/处理方法` 构成。"
                if object_name_query
                else (
                    "第一次 VLM 未执行；query 仅由用户问题、缓存视觉锚点的 `object_name` 和 `section_title` 构成，不包含 VLM 描述、OCR、normalized_query、候选名或视觉关键词。"
                    if anchor_query
                    else "第一次 VLM 未重新执行；本轮复用了缓存 query。"
                )
            )
        ),
        (
            "- 已清除来源缓存中的旧 VLM/guardrail 状态，缓存融合 Top1 直接作为本轮视觉锚点。"
            if ignore_cached_guardrail
            else "- 保留来源缓存中的视觉状态。"
        ),
        "",
        f"- 视觉结果来源：`{result['run']['source_cached_result']}`",
        f"- 总样本：{len(rows)}；可回答：{len(answerable)}；正文证据非空：{summary.get('text_evidence_nonempty', 0)}",
        f"- 答案模型：`{result['run'].get('answer_model')}`；独立 Judge：`{result['run'].get('judge_model')}`",
        f"- `expected_icon` 未出现在最终 text query 的可回答样本：{len(mismatch)}/{len(answerable)}（这是对象名/初诊 query 污染的直接信号，不等同于最终答案必错）。",
        "",
        "## LLM Judge",
        "",
        "| 指标 | 平均分 |",
        "|---|---:|",
        f"| Answer Correctness | {pct(mean(item['answer_correctness'] for item in scored) if scored else 0)} |",
        f"| Groundedness | {pct(mean(item['groundedness'] for item in scored) if scored else 0)} |",
        f"| Answer Execution | {pct(mean(item['answer_execution'] for item in scored) if scored else 0)} |",
        "",
        "## 分类结果",
        "",
        "| 样本 | 数量 | 正文 Gold Hit | Answer Correctness | Groundedness | Answer Execution |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, count, correctness, groundedness, execution, text_hits in grouped:
        summary_md.append(
            f"| {label} | {count} | {text_hits}/{count} | {pct(correctness)} | {pct(groundedness)} | {pct(execution)} |"
        )
    summary_md.extend([
        "",
        "## 对象名污染 query 的样本",
        "",
        "| ID | 数据集标准名 | 视觉结果 object_name | 实际 text query |",
        "|---|---|---|---|",
    ])
    for case_id, expected, object_name, query in mismatch:
        summary_md.append(f"| `{case_id}` | {md(expected, 80)} | {md(object_name, 80)} | {md(query, 180)} |")
    if not mismatch:
        summary_md.append("| - | 无 | - | - |")
    summary_md.append("")
    args.result.with_suffix(".md").write_text("\n".join(summary_md), encoding="utf-8")

    audit = [
        "# Cached VLM 图片问答逐条 Audit",
        "",
        "每条记录均复用缓存视觉锚点；以下展示正文检索、答案和独立 Judge 的实际输入输出。",
    ]
    for index, row in enumerate(rows, start=1):
        record = records[row["id"]]
        judgment = judgments.get(row["id"], {})
        visual = row.get("visual_match") or {}
        evidence = row.get("evidence") or []
        evidence_refs = ", ".join(
            f"p{item.get('page')}:{item.get('chunk_id')}" for item in evidence[1:]
        ) or "无"
        audit.extend([
            "",
            f"## {index}. `{row['id']}`",
            "",
            f"- 问题：{md(record.get('question'))}",
            f"- 数据集标准名：{md(record.get('expected_icon') or record.get('expected_visual_title'))}",
            f"- 缓存视觉 object_name：{md(visual.get('object_name'))}",
            f"- text query：{md(row.get('text_query'))}",
            f"- 正文证据：{evidence_refs}",
            f"- 答案：{md(row.get('answer'), 2200)}",
        ])
        if judgment:
            audit.extend([
                f"- Judge：correctness={judgment.get('answer_correctness')}；groundedness={judgment.get('groundedness')}；execution={judgment.get('answer_execution')}",
                f"- Judge 理由：{md(judgment.get('correctness_reason'))}",
            ])
        else:
            audit.append("- Judge：拒答样本未评分")
    audit_path = args.result.with_name(args.result.stem + "_audit.md")
    audit_path.write_text("\n".join(audit), encoding="utf-8")
    print(json.dumps({"summary": str(args.result.with_suffix('.md')), "audit": str(audit_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
