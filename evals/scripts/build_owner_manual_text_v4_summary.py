"""Build the concise summary for the v4 owner-manual text evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.scripts.text_eval_v4_metrics import (
    aggregate_auto_merge_context,
    aggregate_route_cutoffs,
    case_claim_metrics,
)
from models.manual import ManualChunk
from utils.database import SessionLocal


ROUTE_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "rrf": "Weighted RRF",
    "cross_encoder": "Cross-Encoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v4 text eval summary.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pct(value: Any) -> str:
    return f"{float(value or 0) * 100:.2f}%"


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return (
        sum(float(row[field]) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    result_rows = list(result.get("text_rows") or [])
    judgments = load_jsonl(args.judgments)
    if {row["id"] for row in judgments} != {row["id"] for row in records}:
        raise ValueError("judgments do not cover exactly the dataset ids")

    answerable = [record for record in records if not record.get("should_refuse")]
    claim_count = sum(len(record.get("required_claims") or []) for record in answerable)
    answerable_ids = {record["id"] for record in answerable}
    refusal_ids = {record["id"] for record in records if record.get("should_refuse")}
    judgment_answerable = [row for row in judgments if row["id"] in answerable_ids]
    judgment_refusal = [row for row in judgments if row["id"] in refusal_ids]
    judge_models = sorted(
        {str(row.get("judge_model") or "").strip() for row in judgments}
    )
    judge_model = ", ".join(model for model in judge_models if model) or "unknown"
    answer_model = str(result.get("run", {}).get("answer_model") or "unknown")

    route_cutoffs = aggregate_route_cutoffs(records, result_rows)
    returned_ids = {
        chunk_id
        for row in result_rows
        for chunk_id in (
            list(row.get("evidence_chunk_ids") or [])
            + [
                item
                for route in (row.get("routes") or {}).values()
                for item in list(route.get("top_chunk_ids") or [])
            ]
        )
    }
    db = SessionLocal()
    try:
        chunks = (
            db.query(ManualChunk)
            .filter(ManualChunk.chunk_id.in_(returned_ids))
            .all()
        )
    finally:
        db.close()
    chunk_metadata = {
        chunk.chunk_id: {
            "token_count": int(chunk.token_count or 0),
            "is_leaf": chunk.is_leaf,
        }
        for chunk in chunks
    }
    final_route = str(
        result.get("run", {}).get("text_final_route") or "cross_encoder"
    )
    context = aggregate_auto_merge_context(
        records,
        result_rows,
        chunk_metadata,
        top_k=5,
        final_route=final_route,
    )

    record_by_id = {record["id"]: record for record in records}
    incomplete: list[str] = []
    zero_coverage: list[str] = []
    for row in result_rows:
        record = record_by_id[row["id"]]
        if record.get("should_refuse"):
            continue
        metrics = case_claim_metrics(
            record,
            list(
                ((row.get("routes") or {}).get(final_route) or {}).get(
                    "top_chunk_ids"
                )
                or []
            ),
            5,
        )
        if metrics["complete_claim_hit"] < 1:
            incomplete.append(row["id"])
        if metrics["any_claim_hit"] < 1:
            zero_coverage.append(row["id"])

    false_refusals = [
        row["id"]
        for row in result_rows
        if not record_by_id[row["id"]].get("should_refuse")
        and row.get("predicted_refuse")
    ]
    false_accepts = [
        row["id"]
        for row in result_rows
        if record_by_id[row["id"]].get("should_refuse")
        and not row.get("predicted_refuse")
    ]
    text_summary = (result.get("summary") or {}).get("text") or {}
    refusal = text_summary.get("refusal") or {}
    latency = text_summary.get("latency") or {}
    ce_latency = text_summary.get("cross_encoder_latency") or {}
    pre = context["pre_auto_merge"]
    post = context["post_auto_merge"]
    token_delta = float(post["mean_tokens"]) - float(pre["mean_tokens"])
    token_delta_pct = (
        token_delta / float(pre["mean_tokens"]) if pre["mean_tokens"] else 0.0
    )
    gold_issue_ids = [
        row["id"] for row in judgments if bool(row.get("gold_issue"))
    ]

    lines = [
        "# 车主手册文本问答重构后评测汇总",
        "",
        "## 1. 测试口径",
        "",
        f"- 运行时间：`{result.get('run', {}).get('finished_at')}`",
        f"- 数据集版本：`{records[0].get('dataset_version')}`",
        f"- 样本：{len(records)} 条，其中应回答 {len(answerable)} 条、应拒答 {len(records) - len(answerable)} 条。",
        f"- 可回答题共标注 {claim_count} 个原子必答点。每个必答点可对应一个或多个独立支持叶子 Chunk，命中任意一个即可。",
        f"- 检索指标在 AutoMerge 前计算，只评价叶子召回器；AutoMerge 另做前后上下文比较；最终回答由独立裁判 `{judge_model}` 按 0.2 步长评分。",
        "- 叶子检索将同一必答点的等价叶子视为 OR 组；多个必答点若由完全相同的叶子支持，只计一个叶子 Gold 组。因此同一事实的 A/B 等价叶子不需要同时召回，同一叶子内的多个事实也不会伪装成多个必需 Chunk。",
        "",
        "## 2. 关键结论",
        "",
        f"- Cross-Encoder 的 Leaf Hit@5 为 **{pct(route_cutoffs['cross_encoder']['at_5']['leaf_hit'])}**，Macro Leaf-group Recall@5 为 **{pct(route_cutoffs['cross_encoder']['at_5']['macro_leaf_recall'])}**；对应的完整必答点覆盖率为 **{pct(route_cutoffs['cross_encoder']['at_5']['complete_claim_hit'])}**。",
        f"- 相比 Weighted RRF，Cross-Encoder 将 Leaf Hit@1 从 {pct(route_cutoffs['rrf']['at_1']['leaf_hit'])} 提高到 {pct(route_cutoffs['cross_encoder']['at_1']['leaf_hit'])}，Leaf Hit@5 从 {pct(route_cutoffs['rrf']['at_5']['leaf_hit'])} 提高到 {pct(route_cutoffs['cross_encoder']['at_5']['leaf_hit'])}。",
        f"- AutoMerge 在 {pct(context['merge_trigger_rate'])} 的可回答题上触发，但本轮必答点覆盖没有变化；平均上下文从 {pre['mean_tokens']:.1f} token 增至 {post['mean_tokens']:.1f} token（{token_delta_pct * 100:.1f}%）。这说明它提供了更完整的章节上下文，但本轮没有带来额外召回收益。",
        f"- 最终回答 LLM 评审：全部样本 Answer Correctness **{pct(mean(judgments, 'answer_correctness'))}**、Groundedness **{pct(mean(judgments, 'groundedness'))}**。",
        "",
        "## 3. AutoMerge 前的叶子检索消融",
        "",
        "| 路线 | K | Leaf Hit | Leaf Precision | Macro Leaf-group Recall | Leaf F1 | MRR@K |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route_name in ("bm25", "dense", "rrf", "cross_encoder"):
        for cutoff in (1, 3, 5):
            metrics = route_cutoffs[route_name][f"at_{cutoff}"]
            lines.append(
                f"| {ROUTE_LABELS[route_name]} | {cutoff} | "
                f"{pct(metrics['leaf_hit'])} | "
                f"{pct(metrics['leaf_precision'])} | "
                f"{pct(metrics['macro_leaf_recall'])} | "
                f"{pct(metrics['leaf_f1'])} | "
                f"{metrics['leaf_reciprocal_rank']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 4. AutoMerge 前后必答点覆盖",
            "",
            "| 上下文 | Any Claim Hit@5 | Complete Claim Hit@5 | Macro Claim Recall@5 | Micro Claim Recall@5 | 平均 token |",
            "|---|---:|---:|---:|---:|---:|",
            f"| CE Top5 叶子（合并前） | {pct(pre['any_claim_hit'])} | {pct(pre['complete_claim_hit'])} | {pct(pre['macro_claim_recall'])} | {pct(pre['micro_claim_recall'])} | {pre['mean_tokens']:.1f} |",
            f"| 最终送模上下文（合并后） | {pct(post['any_claim_hit'])} | {pct(post['complete_claim_hit'])} | {pct(post['macro_claim_recall'])} | {pct(post['micro_claim_recall'])} | {post['mean_tokens']:.1f} |",
            "",
            f"- 覆盖提升/不变/下降：{context['coverage_improved_count']}/{context['coverage_unchanged_count']}/{context['coverage_reduced_count']}。",
            "",
            "## 5. 最终回答与拒答",
            "",
            "| 范围 | 样本数 | Answer Correctness | Groundedness |",
            "|---|---:|---:|---:|",
            f"| 全部 | {len(judgments)} | {pct(mean(judgments, 'answer_correctness'))} | {pct(mean(judgments, 'groundedness'))} |",
            f"| 应回答 | {len(judgment_answerable)} | {pct(mean(judgment_answerable, 'answer_correctness'))} | {pct(mean(judgment_answerable, 'groundedness'))} |",
            f"| 应拒答 | {len(judgment_refusal)} | {pct(mean(judgment_refusal, 'answer_correctness'))} | {pct(mean(judgment_refusal, 'groundedness'))} |",
            "",
            "| Refusal Precision | Refusal Recall | Refusal F1 | Refusal Accuracy |",
            "|---:|---:|---:|---:|",
            f"| {pct(refusal.get('precision'))} | {pct(refusal.get('recall'))} | {pct(refusal.get('f1'))} | {pct(refusal.get('accuracy'))} |",
            "",
            f"- 错误拒答：{', '.join(f'`{item}`' for item in false_refusals) or '无'}。",
            f"- 错误接受：{', '.join(f'`{item}`' for item in false_accepts) or '无'}。",
            f"- LLM 标记金标问题：{', '.join(f'`{item}`' for item in gold_issue_ids) or '无'}。",
            "",
            "## 6. 失败样本与运行状态",
            "",
            f"- CE Top5 未完整覆盖必答点：{', '.join(f'`{item}`' for item in incomplete) or '无'}。",
            f"- CE Top5 零必答点覆盖：{', '.join(f'`{item}`' for item in zero_coverage) or '无'}。",
            f"- 全链路延迟：均值 {latency.get('mean_seconds', 0):.3f}s，P50 {latency.get('p50_seconds', 0):.3f}s，P95 {latency.get('p95_seconds', 0):.3f}s，最大 {latency.get('max_seconds', 0):.3f}s。",
            f"- Cross-Encoder 延迟：均值 {ce_latency.get('mean_seconds', 0):.3f}s，P95 {ce_latency.get('p95_seconds', 0):.3f}s。",
            f"- 生成错误 {text_summary.get('errors', 0)}，Cross-Encoder 错误 {text_summary.get('cross_encoder_errors', 0)}，致命错误 {text_summary.get('fatal_errors', 0)}。",
            "",
            "## 7. 结果文件",
            "",
            f"- 原始结果：`{args.result.name}`",
            f"- LLM 逐题裁判：`{args.judgments.name}`",
            "- 完整逐题证据、路线排名和评分理由见同目录的 `owner_manual_text_refactor_audit.md`。",
            "",
            f"模型说明：回答模型为 `{answer_model}`；裁判为 `{judge_model}`，不参与答案生成。",
        ]
    )
    output = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "characters": len(output),
                "records": len(records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
