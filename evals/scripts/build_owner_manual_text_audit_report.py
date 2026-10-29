"""Build a human-auditable, per-case report from a completed text evaluation.

The report intentionally exposes raw inputs and outputs instead of recomputing
or defending the aggregate metrics.  For every test case it lists the ground
truth annotation, route rankings, exact chunks returned by the system, the text
actually visible to answer generation, and the generated answer.

Run inside the application container::

    python -m evals.scripts.build_owner_manual_text_audit_report \
      --dataset /app/evals/datasets/owner_manual_text_eval_v4.jsonl \
      --result /app/evals/results/text/owner_manual_text_claim_v4_latest.json \
      --output /tmp/owner_manual_text_audit.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.manual import ManualChunk
from utils.database import SessionLocal
from evals.scripts.text_eval_v4_metrics import (
    aggregate_auto_merge_context,
    aggregate_route_cutoffs,
    case_claim_metrics,
    case_leaf_group_metrics,
)


ROUTE_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "rrf": "RRF",
    "cross_encoder": "Cross-Encoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the owner-manual text evaluation audit report.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--judgments",
        type=Path,
        help="Optional JSONL with per-case Answer Correctness and Groundedness judgments.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_text(value: str | None) -> str:
    return str(value or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def answer_prompt_body(chunk: ManualChunk) -> tuple[str, str]:
    """Mirror answer_generation_service.build_grounded_prompt exactly."""
    if clean_text(chunk.content_text):
        return "content_text", clean_text(chunk.content_text)
    return "empty", ""


def percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def mean_score(case_ids: list[str], judgments: dict[str, dict[str, Any]], field: str) -> float:
    values = [float(judgments[case_id][field]) for case_id in case_ids]
    return sum(values) / len(values) if values else 0.0


def validate_judgments(
    records: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> None:
    expected_ids = {record["id"] for record in records}
    missing = sorted(expected_ids - set(judgments))
    extra = sorted(set(judgments) - expected_ids)
    if missing or extra:
        raise ValueError(f"judgment id mismatch: missing={missing[:5]}, extra={extra[:5]}")
    allowed_scores = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
    for case_id, judgment in judgments.items():
        for field in ("answer_correctness", "groundedness"):
            score = float(judgment[field])
            if score not in allowed_scores:
                raise ValueError(f"{case_id}.{field}={score} is not on the 0.2 score grid")
        if not clean_text(judgment.get("correctness_reason")):
            raise ValueError(f"{case_id} is missing correctness_reason")
        if not clean_text(judgment.get("groundedness_reason")):
            raise ValueError(f"{case_id} is missing groundedness_reason")


def format_route_metrics(
    route: dict[str, Any],
    record: dict[str, Any],
    top_k: int,
) -> str:
    leaf_metrics = case_leaf_group_metrics(
        record,
        list(route.get("top_chunk_ids") or []),
        top_k,
    )
    claim_metrics = case_claim_metrics(
        record,
        list(route.get("top_chunk_ids") or []),
        top_k,
    )
    if not leaf_metrics:
        return "无 gold，因此该路线未计算检索分数"
    return (
        f"Leaf Hit@{top_k}={percent(leaf_metrics.get('hit'))}；"
        f"Leaf Precision@{top_k}={percent(leaf_metrics.get('precision'))}；"
        f"Leaf-group Recall@{top_k}={percent(leaf_metrics.get('recall'))}；"
        f"Leaf F1@{top_k}={percent(leaf_metrics.get('f1'))}；"
        f"Claim Recall@{top_k}={percent(claim_metrics.get('macro_claim_recall'))}；"
        f"MRR@{top_k}={float(leaf_metrics.get('reciprocal_rank') or 0):.4f}"
    )


def strict_unit_hits(record: dict[str, Any], chunk_id: str) -> list[str]:
    return [
        unit["id"]
        for unit in record.get("evidence_units") or []
        if chunk_id
        in {
            *(unit.get("acceptable_chunk_ids") or []),
            *(unit.get("acceptable_parent_chunk_ids") or []),
        }
    ]


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def append_aggregate_summary(
    lines: list[str],
    records: list[dict[str, Any]],
    result: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
    top_k: int,
    route_labels: dict[str, str],
    chunk_map: dict[str, ManualChunk],
) -> None:
    text_summary = (result.get("summary") or {}).get("text") or {}
    refusal = text_summary.get("refusal") or {}
    answers = text_summary.get("answers") or {}
    result_rows = list(result.get("text_rows") or [])
    route_cutoffs = aggregate_route_cutoffs(records, result_rows)
    chunk_metadata = {
        chunk_id: {
            "token_count": int(chunk.token_count or 0),
            "is_leaf": chunk.is_leaf,
        }
        for chunk_id, chunk in chunk_map.items()
    }
    final_route = str(
        result.get("run", {}).get("text_final_route") or "cross_encoder"
    )
    context = aggregate_auto_merge_context(
        records,
        result_rows,
        chunk_metadata,
        top_k=top_k,
        final_route=final_route,
    )
    all_ids = [record["id"] for record in records]
    answerable_ids = [record["id"] for record in records if not record.get("should_refuse")]
    refusal_ids = [record["id"] for record in records if record.get("should_refuse")]
    gold_issue_ids = [
        case_id for case_id in all_ids if bool(judgments[case_id].get("gold_issue"))
    ]

    lines.extend(
        [
            "## 汇总结果",
            "",
            "### LLM 语义裁判",
            "",
            "| 范围 | 样本数 | Answer Correctness | Groundedness |",
            "|---|---:|---:|---:|",
            f"| 全部 | {len(all_ids)} | {percent(mean_score(all_ids, judgments, 'answer_correctness'))} | {percent(mean_score(all_ids, judgments, 'groundedness'))} |",
            f"| 应回答 | {len(answerable_ids)} | {percent(mean_score(answerable_ids, judgments, 'answer_correctness'))} | {percent(mean_score(answerable_ids, judgments, 'groundedness'))} |",
            f"| 应拒答 | {len(refusal_ids)} | {percent(mean_score(refusal_ids, judgments, 'answer_correctness'))} | {percent(mean_score(refusal_ids, judgments, 'groundedness'))} |",
            "",
            f"- 发现 {len(gold_issue_ids)} 条参考标注问题：{', '.join(f'`{case_id}`' for case_id in gold_issue_ids) or '无'}。裁判按PDF事实评分，同时在逐题结果中保留问题说明。",
            "- Groundedness 只检查回答中的事实是否被本次最终证据支撑，不检查答案是否完整；完整性由 Answer Correctness 负责。",
            "",
            "### AutoMerge 前的叶子检索指标（v4 必答点标注）",
            "",
            "| 路线 | K | Leaf Hit | Leaf Precision | Macro Leaf-group Recall | Leaf F1 | MRR@K |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for route_name in route_labels:
        for cutoff in (1, 3, 5):
            metrics = (
                route_cutoffs.get(route_name, {}).get(f"at_{cutoff}") or {}
            )
            lines.append(
                f"| {route_labels[route_name]} | {cutoff} | "
                f"{percent(metrics.get('leaf_hit'))} | "
                f"{percent(metrics.get('leaf_precision'))} | "
                f"{percent(metrics.get('macro_leaf_recall'))} | "
                f"{percent(metrics.get('leaf_f1'))} | "
                f"{float(metrics.get('leaf_reciprocal_rank') or 0):.4f} |"
            )
    pre = context["pre_auto_merge"]
    post = context["post_auto_merge"]
    lines.extend(
        [
            "",
            "### AutoMerge 前后上下文",
            "",
            "| 上下文 | Any Claim Hit@5 | Complete Claim Hit@5 | Macro Claim Recall@5 | Micro Claim Recall@5 | 平均 token |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {route_labels.get(final_route, final_route)} Top5（合并前） | "
            f"{percent(pre.get('any_claim_hit'))} | {percent(pre.get('complete_claim_hit'))} | "
            f"{percent(pre.get('macro_claim_recall'))} | {percent(pre.get('micro_claim_recall'))} | "
            f"{float(pre.get('mean_tokens') or 0):.1f} |",
            f"| 最终送模上下文（AutoMerge 后） | "
            f"{percent(post.get('any_claim_hit'))} | {percent(post.get('complete_claim_hit'))} | "
            f"{percent(post.get('macro_claim_recall'))} | {percent(post.get('micro_claim_recall'))} | "
            f"{float(post.get('mean_tokens') or 0):.1f} |",
            "",
            f"- AutoMerge 触发率：{percent(context.get('merge_trigger_rate'))}；"
            f"必答点覆盖提升/不变/下降：{context.get('coverage_improved_count')}/"
            f"{context.get('coverage_unchanged_count')}/{context.get('coverage_reduced_count')}。",
            "",
            "### 拒答与执行度（自动指标）",
            "",
            "| Refusal Precision | Refusal Recall | Refusal F1 | Refusal Accuracy | Answer Execution |",
            "|---:|---:|---:|---:|---:|",
            f"| {percent(refusal.get('precision'))} | {percent(refusal.get('recall'))} | "
            f"{percent(refusal.get('f1'))} | {percent(refusal.get('accuracy'))} | "
            f"{percent(answers.get('execution_score'))} |",
            "",
        ]
    )


def build_report(
    records: list[dict[str, Any]],
    result: dict[str, Any],
    chunk_map: dict[str, ManualChunk],
    top_k: int,
    judgments: dict[str, dict[str, Any]] | None = None,
) -> str:
    rows = {row["id"]: row for row in result.get("text_rows") or []}
    final_route = str(result.get("run", {}).get("text_final_route") or "cross_encoder")
    route_labels = {
        name: label
        for name, label in ROUTE_LABELS.items()
        if name != "cross_encoder" or final_route == "cross_encoder"
    }
    final_route_label = ROUTE_LABELS.get(final_route, final_route)
    lines = [
        f"# 车主手册文本评测逐题审计报告：{final_route_label}",
        "",
        "本报告用于人工复核，不对当前指标口径作正确性背书。每条测试均展示原始标注、检索路线的真实 Top-5、最终送入回答模型的 Chunk，以及系统最终回答。",
        "",
        "## 报告说明",
        "",
        f"- 测试数据：`{result.get('run', {}).get('text_dataset')}`",
        f"- 结果时间：{result.get('run', {}).get('finished_at')}；Top-k={top_k}",
        f"- 最终回答证据路线：**{final_route_label} Top-{top_k}**",
        f"- 测试条数：{len(records)}；可回答：{sum(not record.get('should_refuse') for record in records)}；应拒答：{sum(bool(record.get('should_refuse')) for record in records)}",
        "- `严格 gold` 表示该叶子 Chunk 是某个必答点的独立支持证据；同一必答点有多个 OR 叶子时，命中任意一个即可。",
        "- `chunk_page` 是数据库中的 Chunk 起始页，`evidence_pages` 是答案实际出现的 PDF 物理页。",
        "- `送模正文` 使用 `content_text`；`section_title` 单独作为完整书签路径内联到提示词。",
        "- 同一 Chunk 若被多条路线返回，只在该题的 Chunk 详情区展示一次；路线排名表保留其真实名次。",
        "",
    ]
    if judgments:
        judge_model = clean_text(next(iter(judgments.values())).get("judge_model")) or "未标注"
        lines.extend(
            [
                "## 裁判口径",
                "",
                f"- 裁判模型：**{judge_model}**。",
                "- Answer Correctness：1.0=正确完整；0.8=基本正确，仅小遗漏；0.6=主结论正确但有重要遗漏；0.4=部分正确且遗漏较大；0.2=只有少量有效内容；0.0=错误、未回答或对可回答题错误拒答。",
                "- Groundedness：1.0=全部事实有证据；0.8=仅少量轻微扩展；0.6=主要事实有证据但存在明显证据外内容；0.4=有据与无据内容相当；0.2=只有少量内容有据；0.0=基本无证据或与证据冲突。",
                "- 两项均限制为 `0.0, 0.2, 0.4, 0.6, 0.8, 1.0` 六档。",
                "",
            ]
        )
        append_aggregate_summary(
            lines,
            records,
            result,
            judgments,
            top_k,
            route_labels,
            chunk_map,
        )
    lines.extend(["## 总目录", ""])
    for index, record in enumerate(records, start=1):
        lines.append(f"- [{index:02d}. {record['id']} - {record['question']}](#case-{index:02d})")

    for index, record in enumerate(records, start=1):
        row = rows.get(record["id"])
        if row is None:
            raise KeyError(f"result row missing: {record['id']}")
        routes = row.get("routes") or {}
        route_top_ids = {
            route_name: (routes.get(route_name, {}).get("top_chunk_ids") or [])[:top_k]
            for route_name in route_labels
        }
        final_ids = (row.get("evidence_chunk_ids") or [])[:top_k]
        detail_ids = unique(
            final_ids
            + [chunk_id for route_name in route_labels for chunk_id in route_top_ids[route_name]]
        )

        lines.extend(
            [
                "",
                f"<a id=\"case-{index:02d}\"></a>",
                f"## {index:02d}. {record['id']}",
                "",
                f"- 难度：`{record.get('difficulty')}`；类别：`{record.get('category')}`；风险：`{record.get('risk_level')}`",
                f"- 预期：{'拒答' if record.get('should_refuse') else '应回答'}；系统判断：{'拒答' if row.get('predicted_refuse') else '回答'}",
                f"- 参考页：{', '.join(str(page) for page in record.get('reference_pages') or []) or '无'}",
                "",
                "### 问题",
                "",
                record["question"],
                "",
                "### 参考答案",
                "",
                record.get("reference_answer") or "无",
                "",
                "### 当前标准证据",
                "",
            ]
        )
        required_claims = record.get("required_claims") or []
        evidence_units = record.get("evidence_units") or []
        if not required_claims and not evidence_units:
            lines.append("该题预期拒答，没有标准答案 Chunk。")
        elif required_claims:
            for unit_index, claim in enumerate(required_claims, start=1):
                supports = claim.get("supporting_leaf_chunks") or []
                parent_ids = [
                    parent.get("chunk_id")
                    for parent in claim.get("supporting_auto_merge_parents") or []
                    if parent.get("chunk_id")
                ]
                lines.extend(
                    [
                        f"#### 必答事实 {unit_index}: `{claim.get('claim_id')}`",
                        "",
                        f"- 必答点：{clean_text(claim.get('claim_text'))}",
                        f"- 叶子 OR 证据数：{len(supports)}；任意一个即可命中该必答点。",
                    ]
                )
                for support in supports:
                    lines.extend(
                        [
                            f"- 叶子 Chunk：`{support.get('chunk_id')}`；"
                            f"chunk 起始页：{support.get('chunk_page')}；"
                            f"PDF 证据页：{support.get('evidence_pages')}",
                            f"- PDF 证据原文：{clean_text(support.get('evidence_text'))}",
                        ]
                    )
                lines.extend(
                    [
                        "- AutoMerge 后可接受父块："
                        + (
                            ", ".join(f"`{chunk_id}`" for chunk_id in parent_ids)
                            if parent_ids
                            else "无"
                        ),
                        "",
                    ]
                )
        else:
            for unit_index, unit in enumerate(evidence_units, start=1):
                ids = unit.get("acceptable_chunk_ids") or []
                lines.extend(
                    [
                        f"#### 必答事实 {unit_index}: `{unit.get('id')}`",
                        "",
                        f"- 页码：{unit.get('page')}；当前标注可接受 Chunk 数：{len(ids)}",
                        f"- 可接受 Chunk ID：{', '.join(f'`{chunk_id}`' for chunk_id in ids)}",
                        "- PDF 证据原文：",
                        "",
                        "````text",
                        clean_text(unit.get("evidence_text")),
                        "````",
                        "",
                    ]
                )

        lines.extend(["### 检索路线的真实 Top-5", ""])
        for route_name, route_label in route_labels.items():
            route = routes.get(route_name) or {}
            ranked = route_top_ids[route_name]
            lines.append(f"#### {route_label}")
            lines.append("")
            lines.append(
                f"- 按 v4 标注重算：{format_route_metrics(route, record, top_k)}"
            )
            if ranked:
                for rank, chunk_id in enumerate(ranked, start=1):
                    chunk = chunk_map.get(chunk_id)
                    page = chunk.page_number if chunk else "?"
                    modality = chunk.modality if chunk else "missing"
                    gold = "是" if chunk_id in set(record.get("relevant_chunk_ids") or []) else "否"
                    unit_hits = strict_unit_hits(record, chunk_id)
                    lines.append(
                        f"- Rank {rank}：第 {page} 页；`{modality}`；`{chunk_id}`；严格 gold={gold}；"
                        f"命中事实={','.join(unit_hits) if unit_hits else '无'}"
                    )
            else:
                lines.append("- 无返回结果")
            lines.append("")

        lines.extend(
            [
                "### 最终送入回答模型",
                "",
                f"- Chunk 顺序：{', '.join(f'`{chunk_id}`' for chunk_id in final_ids) or '无'}",
                f"- 生成耗时：{float(row.get('elapsed_seconds') or 0):.3f}s；生成异常：{row.get('generation_error') or '无'}",
                "",
                "### 实际返回 Chunk 详情",
                "",
            ]
        )

        for chunk_index, chunk_id in enumerate(detail_ids, start=1):
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                lines.extend([f"#### Chunk {chunk_index}: `{chunk_id}`", "", "数据库中未找到该 Chunk。", ""])
                continue
            source, body = answer_prompt_body(chunk)
            raw_content = clean_text(chunk.content_text)
            route_positions = []
            for route_name, route_label in route_labels.items():
                if chunk_id in route_top_ids[route_name]:
                    route_positions.append(f"{route_label}#{route_top_ids[route_name].index(chunk_id) + 1}")
            if chunk_id in final_ids:
                route_positions.append(f"最终证据#{final_ids.index(chunk_id) + 1}")
            unit_hits = strict_unit_hits(record, chunk_id)
            lines.extend(
                [
                    f"#### Chunk {chunk_index}: `{chunk_id}`",
                    "",
                    f"- 出现位置：{', '.join(route_positions)}",
                    f"- 页码：{chunk.page_number}；模态：`{chunk.modality}`；章节标题：{clean_text(chunk.section_title) or '无'}",
                    f"- 父节点：{chunk.parent_chunk_id or '无'}；叶子：{chunk.is_leaf}；token：{chunk.token_count}",
                    f"- 严格 gold：{'是' if chunk_id in set(record.get('relevant_chunk_ids') or []) else '否'}；命中必答事实：{', '.join(unit_hits) if unit_hits else '无'}",
                    f"- 原始 `content_text` 字符数：{len(raw_content)}",
                    "",
                    "````text",
                    raw_content or "<空>",
                    "````",
                    "",
                    f"- 实际送模正文：使用上面的 `{source}`，字符数 {len(body)}。",
                    "",
                ]
            )

        scores = row.get("answer_scores") or {}
        lines.extend(
            [
                "### 系统最终回答",
                "",
                clean_text(row.get("answer")) or "<空回答>",
                "",
            ]
        )
        if judgments:
            judgment = judgments[record["id"]]
            lines.extend(
                [
                    "### LLM 裁判结果",
                    "",
                    f"- **Answer Correctness：{float(judgment['answer_correctness']):.1f} / 1.0**",
                    f"- 评分理由：{clean_text(judgment.get('correctness_reason'))}",
                    f"- **Groundedness：{float(judgment['groundedness']):.1f} / 1.0**",
                    f"- 评分理由：{clean_text(judgment.get('groundedness_reason'))}",
                    f"- 金标问题：{'是' if judgment.get('gold_issue') else '否'}",
                ]
            )
            if judgment.get("gold_issue"):
                lines.append(f"- 金标问题说明：{clean_text(judgment.get('gold_issue_reason'))}")
            lines.append("")
        lines.extend(
            [
                "### 自动代理指标（保留）",
                "",
                f"- Claim 覆盖：{percent(scores.get('claim_coverage'))}；Claim 通过率：{percent(scores.get('claim_pass_rate'))}",
                f"- 忠实度代理：{percent(scores.get('groundedness_proxy'))}；执行度：{percent(scores.get('execution_score'))}；综合代理：{percent(scores.get('answer_quality_proxy'))}",
                "",
                "---",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    records = load_jsonl(args.dataset)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    judgments: dict[str, dict[str, Any]] | None = None
    if args.judgments:
        judgment_rows = load_jsonl(args.judgments)
        judgments = {row["id"]: row for row in judgment_rows}
        if len(judgments) != len(judgment_rows):
            raise ValueError("judgment JSONL contains duplicate ids")
        validate_judgments(records, judgments)
    text_rows = result.get("text_rows") or []
    all_chunk_ids = unique(
        [
            chunk_id
            for row in text_rows
            for route in (row.get("routes") or {}).values()
            for chunk_id in (route.get("top_chunk_ids") or [])[: args.top_k]
        ]
        + [chunk_id for row in text_rows for chunk_id in (row.get("evidence_chunk_ids") or [])[: args.top_k]]
    )
    db = SessionLocal()
    try:
        chunks = db.query(ManualChunk).filter(ManualChunk.chunk_id.in_(all_chunk_ids)).all()
    finally:
        db.close()
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    missing = sorted(set(all_chunk_ids) - set(chunk_map))
    if missing:
        raise RuntimeError(f"{len(missing)} returned chunks are missing from PostgreSQL: {missing[:5]}")
    report = build_report(records, result, chunk_map, args.top_k, judgments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "returned_unique_chunks": len(all_chunk_ids),
                "judgments": len(judgments or {}),
                "characters": len(report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
