"""Targeted checks for the 2026-10 alignment changes (D01/D02/D09/D10)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from service.es_index_service import _should_index
from service.hierarchical_chunk_service import _split_blocks_by_tokens, build_hierarchical_text_chunks
from service.icon_asset_grouping_service import (
    _accepted_partitions,
    _review_component,
    _select_prototypes,
)


def _unit(unit_id: str, *, path: str = "unit.png") -> dict:
    return {
        "unit_id": unit_id,
        "asset_id": unit_id.split(":")[0],
        "orientation": "r0",
        "image_path": path,
        "phash": "0000000000000000",
        "colour": {"hue": 10.0, "saturation": 100.0},
        "vector": [1.0, 0.0, 0.0],
        "quality_score": 1.0,
    }


class IconPartitionTest(unittest.TestCase):
    def test_partition_is_used_directly_without_pairwise_confirmation(self) -> None:
        units = {unit_id: _unit(unit_id) for unit_id in ("a:r0", "b:r0", "c:r0")}
        verdict = [{"indices": [1, 2], "confidence": 0.95}]
        with (
            patch.dict("os.environ", {"VLM_API_KEY": "test-key"}),
            patch(
                "service.icon_asset_grouping_service._vlm_partition",
                return_value=verdict,
            ) as partition,
            patch(
                "service.icon_asset_grouping_service._vlm_confirm_same_identity",
                create=True,
            ) as pair_confirm,
        ):
            groups = _review_component(["a:r0", "b:r0", "c:r0"], units, set())
        partition.assert_called_once()
        pair_confirm.assert_not_called()
        self.assertIn(["a:r0", "b:r0"], groups)
        self.assertIn(["c:r0"], groups)

    def test_unassigned_and_low_confidence_items_stay_singletons(self) -> None:
        batch = ["a:r0", "b:r0", "c:r0"]
        verdict = [
            {"indices": [1, 2], "confidence": 0.95},
            {"indices": [2, 3], "confidence": 0.50},
            {"indices": [7], "confidence": 0.99},
        ]
        groups, assigned = _accepted_partitions(verdict, batch)
        self.assertEqual(groups, [["a:r0", "b:r0"]])
        self.assertEqual(assigned, {"a:r0", "b:r0"})

    def test_duplicate_membership_removes_ambiguous_item(self) -> None:
        batch = ["a:r0", "b:r0", "c:r0"]
        verdict = [
            {"indices": [1, 2], "confidence": 0.95},
            {"indices": [2, 3], "confidence": 0.95},
        ]
        groups, _ = _accepted_partitions(verdict, batch)
        self.assertNotIn("b:r0", [unit for group in groups for unit in group])

    def test_one_representative_prototype_per_group(self) -> None:
        units = [_unit("a:r0"), _unit("b:r0"), _unit("c:r0")]
        prototypes = _select_prototypes("group", units)
        self.assertEqual(len(prototypes), 1)
        self.assertEqual(prototypes[0]["prototype_rank"], 1)

    def test_icon_chunk_is_indexed_without_prototype(self) -> None:
        self.assertTrue(_should_index({"modality": "icon", "chunk_id": "chunk-1"}))
        self.assertFalse(_should_index({"modality": "icon"}))


class TokenBudgetTest(unittest.TestCase):
    def test_oversized_first_item_is_skipped(self) -> None:
        from service.auto_merge_service import _fit_token_budget

        items = [
            {"chunk_id": "long", "content_text": "x" * 6001},
            {"chunk_id": "short", "content_text": "y" * 100},
        ]
        with patch(
            "service.hierarchical_chunk_service.count_tokens",
            side_effect=lambda text: len(text or ""),
        ):
            kept = _fit_token_budget(items, 5000)
        self.assertEqual([item["chunk_id"] for item in kept], ["short"])

    def test_items_are_accepted_up_to_but_not_over_the_limit(self) -> None:
        from service.auto_merge_service import _fit_token_budget

        items = [{"chunk_id": "a", "content_text": "x" * 5000}]
        with patch(
            "service.hierarchical_chunk_service.count_tokens",
            side_effect=lambda text: len(text or ""),
        ):
            self.assertEqual(len(_fit_token_budget(items, 5000)), 1)
            self.assertEqual(_fit_token_budget(items, 4999), [])


class SourcePagesTest(unittest.TestCase):
    def test_cross_page_leaf_records_every_source_page(self) -> None:
        blocks = [(10, "第一页的段落"), (11, "第二页的段落")]
        leaves = _split_blocks_by_tokens(blocks, 500)
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0][0], [10, 11])

    def test_parent_payload_keeps_all_pages(self) -> None:
        bookmarks = [
            {
                "node_id": "n1",
                "path": "车辆使用 > 驾驶",
                "title": "驾驶",
                "page": 10,
                "y": 0.0,
                "level": 1,
                "parent_id": None,
                "children_ids": ["n2"],
            },
            {
                "node_id": "n2",
                "path": "车辆使用 > 驾驶 > 模式",
                "title": "模式",
                "page": 11,
                "y": 0.0,
                "level": 2,
                "parent_id": "n1",
                "children_ids": [],
            },
        ]
        blocks = {
            10: [{"kind": "text", "text": "关于模式", "bbox": [0, 100, 400, 120], "page_width": 432, "page_height": 648}],
            11: [{"kind": "text", "text": "切换到运动模式", "bbox": [0, 100, 400, 120], "page_width": 432, "page_height": 648}],
        }
        chunks = build_hierarchical_text_chunks("m1", bookmarks, blocks)
        parent = next(chunk for chunk in chunks if chunk.get("is_leaf") is False)
        self.assertEqual(parent["source_pages"], [10, 11])
        for leaf in (chunk for chunk in chunks if chunk.get("is_leaf") is True):
            self.assertTrue(leaf["source_pages"])
            self.assertLessEqual(set(leaf["source_pages"]), {10, 11})


if __name__ == "__main__":
    unittest.main()
