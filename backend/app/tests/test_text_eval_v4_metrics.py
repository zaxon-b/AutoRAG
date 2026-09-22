from __future__ import annotations

import unittest

from evals.scripts.text_eval_v4_metrics import (
    case_claim_metrics,
    case_leaf_group_metrics,
    leaf_equivalence_groups,
)


def claim(claim_id: str, leaf_ids: list[str], parent_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "supporting_leaf_chunks": [
            {"chunk_id": chunk_id} for chunk_id in leaf_ids
        ],
        "supporting_auto_merge_parents": [{"chunk_id": parent_id}],
    }


class TextEvalV4MetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "required_claims": [
                claim("claim_1", ["leaf_a", "leaf_b"], "parent_1"),
                claim("claim_2", ["leaf_a", "leaf_b"], "parent_1"),
                claim("claim_3", ["leaf_c"], "parent_2"),
            ]
        }

    def test_identical_leaf_alternatives_collapse_to_one_gold_group(self) -> None:
        groups = leaf_equivalence_groups(self.record)
        self.assertEqual(
            {frozenset(group) for group in groups},
            {frozenset({"leaf_a", "leaf_b"}), frozenset({"leaf_c"})},
        )

    def test_equivalent_leaf_is_or_not_two_mandatory_gold_chunks(self) -> None:
        metrics = case_leaf_group_metrics(self.record, ["leaf_b"], 1)
        self.assertEqual(metrics["hit"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["complete_hit"], 0.0)

    def test_one_leaf_can_cover_multiple_claims(self) -> None:
        metrics = case_claim_metrics(self.record, ["leaf_a"], 1)
        self.assertEqual(metrics["recalled_claims"], 2.0)
        self.assertAlmostEqual(metrics["macro_claim_recall"], 2 / 3)

    def test_parent_is_only_accepted_after_auto_merge(self) -> None:
        before = case_claim_metrics(
            self.record,
            ["parent_1"],
            1,
            include_parents=False,
        )
        after = case_claim_metrics(
            self.record,
            ["parent_1"],
            1,
            include_parents=True,
        )
        self.assertEqual(before["recalled_claims"], 0.0)
        self.assertEqual(after["recalled_claims"], 2.0)


if __name__ == "__main__":
    unittest.main()
