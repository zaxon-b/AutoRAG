from __future__ import annotations

import json
import time
import unittest
from collections import Counter
from pathlib import Path

from evaluate_owner_manual import (
    aggregate_visual_cutoffs,
    detect_refusal,
    ImageEvaluationTimeout,
    image_evaluation_timeout,
    refusal_metrics,
    route_result,
)


DATASET = Path("/app/evals/datasets/owner_manual_image_eval_native_v4.jsonl")


class RefactorDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_dataset_composition(self) -> None:
        self.assertEqual(len(self.rows), 90)
        self.assertEqual(
            Counter(row["dataset_compatibility"] for row in self.rows),
            {
                "icon_visual_regression": 40,
                "general_visual_answerable": 20,
                "real_icon_strong_positive": 9,
                "real_icon_visual_positive_text_absent_refusal": 1,
                "true_general_refusal": 20,
            },
        )

    def test_every_answerable_case_has_visual_text_and_answer_gold(self) -> None:
        answerable = [row for row in self.rows if not row.get("should_refuse")]
        self.assertEqual(len(answerable), 69)
        for row in answerable:
            self.assertTrue(row.get("relevant_visual_chunk_ids"), row["id"])
            self.assertTrue(row.get("relevant_text_chunk_ids"), row["id"])
            self.assertTrue(row.get("text_evidence_units"), row["id"])
            self.assertTrue(row.get("reference_answer"), row["id"])
            if row["expected_visual_modality"] == "icon":
                self.assertEqual(row.get("visual_gold_type"), "icon_group", row["id"])
                self.assertTrue(row.get("expected_icon_group_ids"), row["id"])
            else:
                self.assertEqual(row.get("visual_gold_type"), "visual_chunk", row["id"])
            self.assertTrue(row.get("source_occurrence_id"), row["id"])
            self.assertEqual(
                row.get("chunk_schema_compatibility"),
                "native_asset_occurrence_icon_group_chunk_v3",
                row["id"],
            )

    def test_only_audited_true_refusals_are_scored_and_none_has_fabricated_gold(self) -> None:
        refusals = [row for row in self.rows if row.get("should_refuse")]
        self.assertEqual(len(refusals), 21)
        scored_ids = {
            row["id"] for row in refusals if row.get("refusal_evaluable")
        }
        self.assertEqual(
            scored_ids,
            {
                "image_refusal_02",
                "image_refusal_04",
                "image_refusal_05",
                "image_refusal_06",
                "image_refusal_07",
                "image_refusal_08",
                "image_refusal_10",
                "image_strong_05",
            },
        )
        self.assertTrue(all(not row.get("relevant_visual_chunk_ids") for row in refusals))
        self.assertTrue(all(row.get("expected_refusal_reason") for row in refusals))
        self.assertEqual(
            Counter(row.get("negative_type") for row in refusals),
            {
                "other_model_icon": 5,
                "unrelated_image": 5,
                "blurry_unrecognizable": 5,
                "ambiguous_full_dashboard": 5,
                None: 1,
            },
        )


class GeneralVisualMetricTest(unittest.TestCase):
    def test_image_case_timeout_interrupts_stalled_work(self) -> None:
        with self.assertRaises(ImageEvaluationTimeout):
            with image_evaluation_timeout(0.01):
                time.sleep(0.1)

    def test_operation_figure_is_scored_by_exact_visual_chunk(self) -> None:
        result = route_result(
            [
                {
                    "chunk_id": "operation-1",
                    "page": 80,
                    "modality": "operation_figure",
                }
            ],
            ["operation-1"],
            5,
            key_for_chunk={"icon-1": "icon-group-1"},
            gold_pages=[80],
        )
        self.assertEqual(result["metrics"]["hit_at_5"], 1.0)

    def test_operation_figure_cutoffs_do_not_require_icon_group(self) -> None:
        metrics = aggregate_visual_cutoffs(
            [
                {
                    "should_refuse": False,
                    "expected_icon_group_id": None,
                    "expected_icon_group_ids": [],
                    "visual_gold_type": "visual_chunk",
                    "expected_visual_chunk_ids": ["operation-gold"],
                    "source_pdf_page": 80,
                    "routes": {
                        "pure_visual_image_embedding": {
                            "top_keys": ["wrong", "operation-gold"],
                        }
                    },
                    "route_candidates": {
                        "pure_visual_image_embedding": [
                            {"chunk_id": "wrong", "page": 81},
                            {"chunk_id": "operation-gold", "page": 80},
                        ]
                    },
                }
            ]
        )
        route = metrics["pure_visual_image_embedding"]
        self.assertEqual(route["at_1"]["hit_at_1"], 0.0)
        self.assertEqual(route["at_3"]["hit_at_3"], 1.0)

    def test_refusal_metrics_are_available_for_true_negatives(self) -> None:
        metrics = refusal_metrics(
            [
                {"should_refuse": True, "predicted_refuse": True},
                {"should_refuse": True, "predicted_refuse": False},
                {"should_refuse": False, "predicted_refuse": False},
            ]
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_fixed_visual_guardrail_answer_is_detected_as_refusal(self) -> None:
        self.assertTrue(detect_refusal("当前手册未找到可靠视觉证据，无法给出确定结论。"))


if __name__ == "__main__":
    unittest.main()
