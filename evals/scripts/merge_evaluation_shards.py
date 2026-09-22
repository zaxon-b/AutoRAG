"""Merge disjoint evaluator result shards and recompute one official report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_owner_manual import aggregate, markdown_report, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _unique_rows(results: list[dict], key: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for result in results:
        for row in result.get(key) or []:
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"A {key} row is missing id")
            if row_id in rows:
                raise ValueError(f"Duplicate {key} id across shards: {row_id}")
            rows[row_id] = row
    return list(rows.values())


def main() -> None:
    args = parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    if not results:
        raise ValueError("No result shards supplied")
    top_k_values = {int(result.get("run", {}).get("top_k") or 0) for result in results}
    if len(top_k_values) != 1:
        raise ValueError(f"Shard top-k mismatch: {sorted(top_k_values)}")
    top_k = top_k_values.pop()
    text_rows = _unique_rows(results, "text_rows")
    image_rows = _unique_rows(results, "image_rows")
    run = dict(results[0].get("run") or {})
    run.update(
        {
            "started_at": min(result["run"]["started_at"] for result in results),
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "duration_seconds": round(
                sum(float(result.get("run", {}).get("duration_seconds") or 0) for result in results),
                3,
            ),
            "merged_shards": [str(path) for path in args.input],
        }
    )
    merged = {
        "run": run,
        "summary": aggregate(text_rows, image_rows, top_k),
        "text_rows": text_rows,
        "image_rows": image_rows,
    }
    write_json_atomic(args.output, merged)
    args.output.with_suffix(".md").write_text(markdown_report(merged, top_k), encoding="utf-8")
    print(
        f"merged text={len(text_rows)} image={len(image_rows)} "
        f"json={args.output} markdown={args.output.with_suffix('.md')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
