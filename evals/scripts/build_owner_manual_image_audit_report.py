"""Build a case-by-case audit for the latest image-QA evaluation.

The optional judgment JSONL contains one row for every answerable case with
0.2-step scores for answer correctness, groundedness and answer execution.
When judgments are supplied, this command also enriches the result JSON and
regenerates its compact Markdown summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_owner_manual import markdown_report, write_json_atomic


SCORE_FIELDS = ("answer_correctness", "groundedness", "answer_execution")
CURRENT_ROUTE_NAMES = (
    "pure_visual_image_embedding",
    "multimodal_embedding",
    "rrf_image_1_multimodal_1",
    "rrf_image_1_multimodal_0_6",
    "legacy_hard_gate",
    "full_chain_pre_vlm",
    "full_chain_visual",
)
LEGACY_ROUTE_NAMES = (
    "pure_visual_image_embedding",
    "multimodal_embedding",
    "three_way_equal_rrf",
    "three_way_weighted_rrf",
    "full_chain_pre_vlm",
    "full_chain_visual",
)
ROUTE_LABELS = {
    "pure_visual_image_embedding": "白底图片向量",
    "multimodal_embedding": "多模态向量",
    "rrf_image_1_multimodal_1": "等权双路 RRF",
    "rrf_image_1_multimodal_0_6": "图片加权双路 RRF",
    "legacy_hard_gate": "旧图片分数硬门控",
    "full_chain_pre_vlm": "实验融合 VLM 前",
    "full_chain_visual": "实验 Top5 VLM 后",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a detailed image-QA evaluation audit.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def compact(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def validate_judgments(
    rows: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> None:
    expected = {row["id"] for row in rows if not row.get("should_refuse")}
    if set(judgments) != expected:
        missing = sorted(expected - set(judgments))
        extra = sorted(set(judgments) - expected)
        raise ValueError(f"Judgment IDs do not match answerable cases; missing={missing}, extra={extra}")
    for case_id, judgment in judgments.items():
        for field in SCORE_FIELDS:
            score = float(judgment[field])
            if score < 0 or score > 1 or abs(score * 5 - round(score * 5)) > 1e-9:
                raise ValueError(f"{case_id}.{field} must be one of 0.0, 0.2, ..., 1.0")


def attach_judgments(result: dict[str, Any], judgments: dict[str, dict[str, Any]]) -> None:
    answerable: list[dict[str, Any]] = []
    for row in result.get("image_rows", []):
        judgment = judgments.get(row["id"])
        if judgment:
            row["llm_judge"] = judgment
            answerable.append(judgment)
    result["summary"]["image"]["llm_judge"] = {
        "judge_model": next((row.get("judge_model") for row in answerable if row.get("judge_model")), "current Codex model"),
        "scale": "0.0 to 1.0 in 0.2 increments",
        "count": len(answerable),
        **{
            field: sum(float(row[field]) for row in answerable) / len(answerable)
            for field in SCORE_FIELDS
        },
    }


def route_cell(candidate: dict[str, Any]) -> str:
    name = candidate.get("object_name") or candidate.get("section_title") or candidate.get("chunk_id")
    return f"{compact(name, 36)} (p{candidate.get('page') or '-'}, `{candidate.get('chunk_id')}`)"


def visual_gold_keys(row: dict[str, Any]) -> set[str]:
    return set(row.get("expected_visual_chunk_ids") or [])


def route_top1_hit(row: dict[str, Any], route: str) -> bool:
    keys = row.get("routes", {}).get(route, {}).get("top_keys") or []
    return bool(keys) and keys[0] in visual_gold_keys(row)


def build_report(result: dict[str, Any], records: dict[str, dict[str, Any]]) -> str:
    image_summary = result["summary"]["image"]
    refusal = image_summary["refusal"]
    rows = list(result.get("image_rows", []))
    answerable_rows = [row for row in rows if not row.get("should_refuse")]
    visual_only = bool(result.get("run", {}).get("skip_text_support"))
    production_top1_misses = [
        row["id"]
        for row in answerable_rows
        if not route_top1_hit(row, "full_chain_pre_vlm")
    ]
    vlm_final_misses = [
        row["id"]
        for row in answerable_rows
        if not row.get("routes", {}).get("full_chain_visual", {}).get("metrics", {}).get("hit_at_5")
    ]
    vlm_regressions = [
        row["id"]
        for row in answerable_rows
        if route_top1_hit(row, "full_chain_pre_vlm")
        and not row.get("routes", {}).get("full_chain_visual", {}).get("metrics", {}).get("hit_at_5")
    ]
    vlm_corrections = [
        row["id"]
        for row in answerable_rows
        if not route_top1_hit(row, "full_chain_pre_vlm")
        and row.get("routes", {}).get("full_chain_visual", {}).get("metrics", {}).get("hit_at_5")
    ]
    text_misses = [] if visual_only else [
        row["id"]
        for row in answerable_rows
        if not row.get("text_support_route", {}).get("metrics", {}).get(
            f"hit_at_{int(row.get('text_support_top_k') or 4)}"
        )
    ]
    refusal_misses = [
        row["id"]
        for row in rows
        if row.get("should_refuse")
        and row.get("refusal_evaluable")
        and not row.get("predicted_refuse")
    ]
    false_refusals = [
        row["id"]
        for row in rows
        if not row.get("should_refuse")
        and row.get("predicted_refuse")
    ]
    lines = [
        "# 图片问答逐条审计报告",
        "",
        f"- 运行时间：{result['run']['finished_at']}",
        (
            f"- 样本：{image_summary['answerable_count']} 条可回答（原有图标 "
            f"{image_summary.get('icon_visual_count', 0) - image_summary.get('strong_real_count', 0)}、"
            f"操作图 {image_summary.get('operation_visual_count', 0)}、实拍强正 "
            f"{image_summary.get('strong_real_count', 0)}）+ 真实拒答 "
            f"{image_summary.get('true_refusal_count', 0)} 条。"
        ),
        f"- 评测配置：图片向量=`{result.get('run', {}).get('image_vector_variant')}`；融合=`{result.get('run', {}).get('visual_fusion_profile')}`；VLM候选Top{result.get('run', {}).get('vlm_top_k')}。",
    ]
    if visual_only:
        lines.append("- 本轮只评视觉召回与VLM实验门控，未运行正文召回和最终答案生成。")
    lines.append("- 生产决策：使用白底图片向量 + 多模态向量等权 RRF 的 Top1；VLM 二验默认关闭。")
    lines.append("- 拒答口径：20 条拒答样本中只有 7 条通过 Gold 审计并纳入通用视觉拒答评分。")
    if refusal.get("status") == "not_available":
        lines.append("- 通用拒答指标：暂停；当前数据集没有合格的开放域拒答样本。")
    else:
        lines.append(
            f"- Top5 VLM 实验拒答 Precision / Recall / F1 / Accuracy：{percent(refusal['precision'])} / {percent(refusal['recall'])} / {percent(refusal['f1'])} / {percent(refusal['accuracy'])}"
        )
    judge = image_summary.get("llm_judge")
    if judge:
        lines.extend([
            f"- 回答裁判：{judge['judge_model']}；分档：0.0、0.2、0.4、0.6、0.8、1.0",
            f"- Answer Correctness / Groundedness / Answer Execution：{percent(judge['answer_correctness'])} / {percent(judge['groundedness'])} / {percent(judge['answer_execution'])}",
        ])

    lines.extend([
        "",
        "## 失败索引",
        "",
        f"- 生产融合 Top1 未命中（{len(production_top1_misses)}）：{', '.join(f'`{case_id}`' for case_id in production_top1_misses) or '无'}",
        f"- 实验 Top5 VLM 最终锚点未命中（{len(vlm_final_misses)}）：{', '.join(f'`{case_id}`' for case_id in vlm_final_misses) or '无'}",
        f"- VLM 将正确 Top1 改坏或误拒（{len(vlm_regressions)}）：{', '.join(f'`{case_id}`' for case_id in vlm_regressions) or '无'}",
        f"- VLM 修正错误 Top1（{len(vlm_corrections)}）：{', '.join(f'`{case_id}`' for case_id in vlm_corrections) or '无'}",
        f"- 实验 VLM 对可回答样本误拒答（{len(false_refusals)}）：{', '.join(f'`{case_id}`' for case_id in false_refusals) or '无'}",
        f"- 7 条有效负样本中误接收（{len(refusal_misses)}）：{', '.join(f'`{case_id}`' for case_id in refusal_misses) or '无'}",
    ])
    if not visual_only:
        lines.append(
            f"- 正文证据未命中（{len(text_misses)}）：{', '.join(f'`{case_id}`' for case_id in text_misses) or '无'}"
        )

    for index, row in enumerate(rows, start=1):
        record = records[row["id"]]
        visual_match = row.get("visual_match") or {}
        recognized_object = row.get("recognized_object") or visual_match.get("object_name")
        visual_hit = (
            row.get("routes", {})
            .get("full_chain_visual", {})
            .get("metrics", {})
            .get("hit_at_5")
        )
        text_hit = (
            row.get("text_support_route", {})
            .get("metrics", {})
            .get(f"hit_at_{int(row.get('text_support_top_k') or 4)}")
        )
        expected = (
            f"应拒答：{record.get('expected_refusal_reason') or record.get('negative_type') or '证据不足'}"
            if row.get("should_refuse")
            else record.get("expected_visual_title") or record.get("expected_icon")
        )
        lines.extend([
            "",
            f"## {index}. {row['id']}",
            "",
            f"- 图片：`{record.get('image_path')}`",
            f"- 问题：{record.get('question')}",
            f"- 金标：{expected}",
            f"- 参考答案：{record.get('reference_answer') or '不适用'}",
            f"- 初诊：`{row.get('image_type')}`；is_target_icon=`{row.get('is_target_icon')}`；置信度={row.get('classification_confidence')}",
            f"- 初诊理由：{compact(row.get('classification_reason'), 600)}",
            f"- 实验VLM决策：{'拒绝全部候选' if row.get('predicted_refuse') else '接受候选'}；识别结果：{recognized_object or '无'}",
        ])
        if not row.get("should_refuse"):
            lines.extend([
                (
                    f"- 评测判定：生产融合Top1={'命中' if route_top1_hit(row, 'full_chain_pre_vlm') else '未命中'}；"
                    f"实验Top5 VLM最终锚点={'命中' if visual_hit else '未命中'}"
                ),
                "",
                "### 视觉候选",
                "",
                "| 路线 | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 |",
                "|---|---|---|---|---|---|",
            ])
            snapshots = row.get("route_candidates") or {}
            route_names = (
                CURRENT_ROUTE_NAMES
                if "rrf_image_1_multimodal_1" in snapshots
                else LEGACY_ROUTE_NAMES
            )
            for route in route_names:
                candidates = list(snapshots.get(route) or [])[:5]
                cells = [route_cell(item) for item in candidates]
                cells.extend(["-"] * (5 - len(cells)))
                lines.append(f"| {ROUTE_LABELS.get(route, route)} | " + " | ".join(cells) + " |")

            verification = row.get("verification") or {}
            lines.extend([
                "",
                "### VLM 实验二验",
                "",
                f"- selected_index：`{verification.get('selected_index')}`",
                f"- reject_all：`{verification.get('reject_all')}`",
                f"- confidence：`{verification.get('confidence')}`",
                f"- 理由：{compact(verification.get('reason'), 800)}",
            ])
            if not visual_only:
                lines.extend([
                    "",
                    "### 最终正文证据",
                    "",
                    "| # | 页码 | Chunk ID | 模态 | 内容摘要 |",
                    "|---:|---:|---|---|---|",
                ])
                for evidence_index, item in enumerate(row.get("evidence") or [], start=1):
                    content = item.get("content_text")
                    lines.append(
                        f"| {evidence_index} | {item.get('page') or '-'} | `{item.get('chunk_id')}` | "
                        f"{item.get('modality')} | {compact(content)} |"
                    )
                lines.extend(["", "### 系统回答", "", row.get("answer") or "（空）"])
                scores = row.get("answer_scores") or {}
                if scores:
                    lines.extend([
                        "",
                        "### 自动代理指标",
                        "",
                        f"- Claim 覆盖：{percent(scores.get('claim_coverage'))}",
                        f"- Claim 通过率：{percent(scores.get('claim_pass_rate'))}",
                        f"- 忠实度代理：{percent(scores.get('groundedness_proxy'))}",
                        f"- 回答执行度：{percent(scores.get('execution_score'))}",
                        f"- 回答质量代理：{percent(scores.get('answer_quality_proxy'))}",
                    ])
            judgment = row.get("llm_judge")
            if judgment:
                lines.extend([
                    "",
                    "### 回答裁判",
                    "",
                    f"- Answer Correctness：{judgment['answer_correctness']:.1f}；{judgment.get('correctness_reason', '')}",
                    f"- Groundedness：{judgment['groundedness']:.1f}；{judgment.get('groundedness_reason', '')}",
                    f"- Answer Execution：{judgment['answer_execution']:.1f}；{judgment.get('execution_reason', '')}",
                ])
        else:
            verification = row.get("verification") or {}
            lines.extend([
                "",
                "### VLM 实验输出（数据集标记为拒答）",
                "",
                f"- 是否纳入拒答评分：`{bool(row.get('refusal_evaluable'))}`",
                f"- 实际决策：{'拒绝全部候选' if row.get('predicted_refuse') else '接受候选'}",
                f"- selected_index：`{verification.get('selected_index')}`",
                f"- confidence：`{verification.get('confidence')}`",
                f"- 理由：{compact(verification.get('reason'), 800)}",
            ])
        if row.get("fatal_error") or row.get("vision_error") or row.get("generation_error"):
            lines.extend([
                "",
                f"- 异常：fatal={row.get('fatal_error')}；vision={row.get('vision_error')}；generation={row.get('generation_error')}",
            ])
    report = "\n".join(lines)
    return "\n".join(line.rstrip() for line in report.splitlines()) + "\n"


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    records = {record["id"]: record for record in load_jsonl(args.dataset)}
    if args.judgments:
        judgments = {row["id"]: row for row in load_jsonl(args.judgments)}
        validate_judgments(result.get("image_rows", []), judgments)
        attach_judgments(result, judgments)
        write_json_atomic(args.result, result)
        args.result.with_suffix(".md").write_text(
            markdown_report(result, int(result["run"].get("top_k") or 5)),
            encoding="utf-8",
        )
    args.output.write_text(build_report(result, records), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
