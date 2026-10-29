"""Convert icon-group visual gold to the unique visual chunk IDs.

After an icon group owns exactly one chunk, icon and operation-figure hits are
scored with the same rule: the returned visual chunk must be the audited one.
This script upgrades a working dataset copy by filling
``expected_visual_chunk_ids`` for icon rows.

Records whose group cannot be mapped unambiguously are reported and left as-is;
the script never guesses a chunk from a similar name. Historical snapshots
under ``final_eval/`` are not touched.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from service.hierarchical_chunk_service import stable_chunk_id


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPOSITORY_ROOT / "evals" / "datasets" / "owner_manual_image_eval_native_v3.jsonl"
DEFAULT_TARGET = REPOSITORY_ROOT / "evals" / "datasets" / "owner_manual_image_eval_native_v4.jsonl"
DEFAULT_REPORT = REPOSITORY_ROOT / "evals" / "datasets" / "owner_manual_image_eval_native_v4_gold_conversion.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert icon-group gold to unique visual chunk IDs.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--group-chunk-map",
        type=Path,
        help="JSON mapping produced by migrate_icon_index.py; falls back to the stable chunk ID rule.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_mapping(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in parsed.items()}


def expected_groups(record: dict[str, Any]) -> list[str]:
    groups = record.get("expected_icon_group_ids") or []
    if not groups and record.get("expected_icon_group_id"):
        groups = [record["expected_icon_group_id"]]
    return [str(group) for group in groups if group]


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.group_chunk_map)
    records = load_jsonl(args.source)

    converted = 0
    already_explicit = 0
    refusals = 0
    unresolved: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for record in records:
        if record.get("should_refuse"):
            refusals += 1
            continue
        explicit = [str(value) for value in record.get("expected_visual_chunk_ids") or []]
        source_chunks = [str(value) for value in record.get("relevant_visual_chunk_ids") or []]
        groups = expected_groups(record)
        if explicit:
            already_explicit += 1
        elif groups:
            chunk_ids = [
                mapping.get(group)
                or stable_chunk_id(record["manual_id"], "icon_group_chunk", group)
                for group in groups
            ]
            chunk_ids = list(dict.fromkeys(chunk_ids))
            if len(chunk_ids) > 1:
                ambiguous.append({"id": record["id"], "groups": groups, "chunks": chunk_ids})
                continue
            record["expected_visual_chunk_ids"] = chunk_ids
            converted += 1
        elif source_chunks:
            record["expected_visual_chunk_ids"] = source_chunks
            converted += 1
        else:
            unresolved.append({"id": record["id"], "reason": "no group or chunk gold"})
            continue

        if (
            groups
            and source_chunks
            and set(source_chunks) != set(record["expected_visual_chunk_ids"])
        ):
            ambiguous.append(
                {
                    "id": record["id"],
                    "groups": groups,
                    "chunks": record["expected_visual_chunk_ids"],
                    "relevant_visual_chunk_ids": source_chunks,
                }
            )

    counts = Counter(row.get("visual_gold_type") for row in records)
    report_lines = [
        "# Visual gold conversion to unique chunk IDs",
        "",
        f"- source: `{args.source}`",
        f"- target: `{args.target}`",
        f"- records: {len(records)}",
        f"- already explicit: {already_explicit}",
        f"- newly converted: {converted}",
        f"- refusal rows without visual gold: {refusals}",
        f"- unresolved: {len(unresolved)}",
        f"- ambiguous: {len(ambiguous)}",
        f"- visual_gold_type counts: {dict(counts)}",
        "",
    ]
    if unresolved:
        report_lines.extend(
            ["## Unresolved (not converted, needs manual review)", ""]
            + [f"- `{item['id']}`: {item['reason']}" for item in unresolved]
            + [""]
        )
    if ambiguous:
        report_lines.extend(
            ["## Ambiguous (left untouched)", ""]
            + [f"- `{item}`" for item in ambiguous]
            + [""]
        )
    report = "\n".join(report_lines)
    print(report)

    if args.dry_run:
        return
    args.target.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
