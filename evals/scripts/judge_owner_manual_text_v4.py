"""Judge final text answers against the v4 atomic-claim annotations.

The script is resumable: every completed judgment is appended to JSONL, and a
later run skips ids already present in that file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


SCORE_GRID = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation for the v4 text dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--allow-same-model",
        action="store_true",
        help="Explicitly allow the answer model to judge its own answers.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def format_evidence(row: dict[str, Any]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(row.get("evidence") or [], start=1):
        blocks.append(
            f"[证据{index}] 第{item.get('page')}页；"
            f"章节={clean_text(item.get('section_title'))}\n"
            f"{clean_text(item.get('content_text'))}"
        )
    return "\n\n".join(blocks) or "<没有返回证据>"


def build_prompt(record: dict[str, Any], row: dict[str, Any]) -> str:
    claims = [
        {
            "claim_id": item.get("claim_id"),
            "claim_text": item.get("claim_text"),
        }
        for item in record.get("required_claims") or []
    ]
    absence_check = record.get("manual_absence_check") or {}
    return f"""你是严格的 RAG 问答评测员。请只输出 JSON，不要输出 Markdown。

评分对象：
- 问题：{record.get('question')}
- 题目类型：{"应拒答" if record.get("should_refuse") else "应回答"}
- 参考答案：{clean_text(record.get('reference_answer')) or "<无>"}
- 原子必答点：{json.dumps(claims, ensure_ascii=False)}
- 应拒答题的全手册核验说明：{json.dumps(absence_check, ensure_ascii=False)}
- 系统回答：{clean_text(row.get('answer')) or "<空回答>"}

系统最终获得的证据：
{format_evidence(row)}

请分别评分：
1. answer_correctness：检查系统回答是否正确并覆盖原子必答点。
   - 应回答题：1.0=所有必答点正确覆盖；0.8=基本完整，仅轻微遗漏；
     0.6=主结论正确但有重要遗漏；0.4=部分正确、遗漏较大；
     0.2=只有少量有效内容；0.0=错误、未回答或错误拒答。
   - 应拒答题：只有明确说明当前手册无法确认、没有编造确定答案时才可得高分。
2. groundedness：只检查系统回答中的事实是否被“系统最终获得的证据”支持。
   1.0=全部有据；0.8=仅轻微无害扩展；0.6=主要内容有据但有明显扩展；
   0.4=有据和无据相当；0.2=仅少量有据；0.0=基本无据或与证据冲突。
3. gold_issue：只有参考答案、必答点或应拒答标签与题面明显矛盾时才为 true。

两个分数只能取 0、0.2、0.4、0.6、0.8、1。
输出格式：
{{
  "answer_correctness": 0.0,
  "groundedness": 0.0,
  "correctness_reason": "一句具体理由",
  "groundedness_reason": "一句具体理由",
  "gold_issue": false,
  "gold_issue_reason": ""
}}"""


def extract_json(text: str) -> dict[str, Any]:
    value = clean_text(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.S)
    if fenced:
        value = fenced.group(1)
    else:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    return json.loads(value)


def normalize_judgment(
    case_id: str, model: str, raw: dict[str, Any]
) -> dict[str, Any]:
    answer_correctness = float(raw.get("answer_correctness"))
    groundedness = float(raw.get("groundedness"))
    if answer_correctness not in SCORE_GRID or groundedness not in SCORE_GRID:
        raise ValueError(
            f"{case_id}: scores must use the 0.2 grid, got "
            f"{answer_correctness}/{groundedness}"
        )
    correctness_reason = clean_text(raw.get("correctness_reason"))
    groundedness_reason = clean_text(raw.get("groundedness_reason"))
    if not correctness_reason or not groundedness_reason:
        raise ValueError(f"{case_id}: judge reasons must not be empty")
    return {
        "id": case_id,
        "judge_model": model,
        "answer_correctness": answer_correctness,
        "groundedness": groundedness,
        "correctness_reason": correctness_reason,
        "groundedness_reason": groundedness_reason,
        "gold_issue": bool(raw.get("gold_issue")),
        "gold_issue_reason": clean_text(raw.get("gold_issue_reason")),
    }


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in result.get("text_rows") or []}
    if set(rows) != {record["id"] for record in records}:
        raise ValueError("dataset ids and result ids do not match")

    api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
    model = args.model or os.getenv("EVAL_JUDGE_MODEL")
    if not model:
        raise RuntimeError(
            "An independent judge is required: pass --model or configure "
            "EVAL_JUDGE_MODEL. The script no longer falls back to LLM_MODEL."
        )
    answer_model = str(result.get("run", {}).get("answer_model") or "")
    if model == answer_model and not args.allow_same_model:
        raise RuntimeError(
            f"judge model {model!r} equals answer model {answer_model!r}; "
            "use an independent model or explicitly pass --allow-same-model"
        )
    if not api_key:
        raise RuntimeError("LLM_API_KEY or DASHSCOPE_API_KEY is required")
    client = OpenAI(api_key=api_key, base_url=base_url)

    existing_rows = load_jsonl(args.output)
    completed = {row["id"]: row for row in existing_rows}
    if len(completed) != len(existing_rows):
        raise ValueError("output contains duplicate judgment ids")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pending = [record for record in records if record["id"] not in completed]
    print(
        json.dumps(
            {
                "model": model,
                "answer_model": answer_model,
                "total": len(records),
                "completed": len(completed),
                "pending": len(pending),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with args.output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(pending, start=1):
            case_id = record["id"]
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": build_prompt(record, rows[case_id]),
                    }
                ],
                temperature=0,
                stream=False,
            )
            raw = extract_json(completion.choices[0].message.content or "")
            judgment = normalize_judgment(case_id, model, raw)
            handle.write(json.dumps(judgment, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{len(completed) + index}/{len(records)}] "
                f"{case_id}: correctness={judgment['answer_correctness']:.1f}, "
                f"groundedness={judgment['groundedness']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
