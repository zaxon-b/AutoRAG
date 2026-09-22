from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from service.auto_merge_service import auto_merge_text_evidence
from service.bookmark_outline_service import _bookmark_y, _has_valid_destination
from service.hierarchical_chunk_service import build_hierarchical_text_chunks
from service.hierarchical_chunk_service import _looks_like_index_page


class BookmarkCoordinateTest(unittest.TestCase):
    def test_invalid_outline_placeholder_is_rejected(self) -> None:
        self.assertFalse(_has_valid_destination([2, "第一章", 0, {"kind": 4}]))

    def test_bottom_origin_destination_is_converted_to_layout_coordinates(self) -> None:
        point = SimpleNamespace(y=538.0)
        self.assertAlmostEqual(_bookmark_y([3, "标题", 1, {"to": point}], 612.0), 74.0)


class HierarchicalChunkTest(unittest.TestCase):
    def test_leaf_inlines_full_bookmark_path_and_points_to_parent(self) -> None:
        nodes = [
            {
                "node_id": "l1",
                "ordinal": 1,
                "level": 1,
                "title": "使用和驾驶",
                "path": "使用和驾驶",
                "parent_id": None,
                "children_ids": ["l2"],
                "page": 1,
                "y": 60.0,
                "end_page": 2,
                "end_y": 600.0,
            },
            {
                "node_id": "l2",
                "ordinal": 2,
                "level": 2,
                "title": "启动车辆",
                "path": "使用和驾驶 > 启动车辆",
                "parent_id": "l1",
                "children_ids": [],
                "page": 1,
                "y": 100.0,
                "end_page": 2,
                "end_y": 600.0,
            },
        ]
        blocks = {
            1: [
                {
                    "block_id": "heading",
                    "page": 1,
                    "kind": "text",
                    "bbox": [50, 100, 150, 120],
                    "text": "启动车辆",
                    "page_width": 430,
                    "page_height": 612,
                },
                {
                    "block_id": "body-a",
                    "page": 1,
                    "kind": "text",
                    "bbox": [50, 140, 380, 220],
                    "text": "方法A：携带钥匙踩下制动踏板。方法B：使用NFC钥匙。",
                    "page_width": 430,
                    "page_height": 612,
                },
            ]
        }

        chunks = build_hierarchical_text_chunks("manual", nodes, blocks, leaf_token_limit=200)
        leaf = next(item for item in chunks if item["is_leaf"])
        parent = next(item for item in chunks if not item["is_leaf"] and item["section_title"] == "使用和驾驶")
        self.assertEqual(leaf["section_title"], "使用和驾驶 > 启动车辆")
        self.assertNotIn("启动车辆\n启动车辆", leaf["content_text"])
        self.assertEqual(leaf["parent_chunk_id"], parent["chunk_id"])

    def test_contents_page_is_not_built_as_searchable_text(self):
        blocks = [
            {
                "kind": "text",
                "page": 1,
                "bbox": [50, 80, 500, 400],
                "page_width": 600,
                "page_height": 800,
                "text": "目录\n座椅安全带........14\n安全气囊........19\n儿童保护装置........27\n双模工作模式........32",
            }
        ]
        nodes = [
            {
                "node_id": "toc",
                "page": 1,
                "y": 70,
                "end_page": 2,
                "end_y": 0,
                "title": "目 录",
                "path": "目 录",
                "parent_id": None,
                "children_ids": [],
            }
        ]

        self.assertTrue(_looks_like_index_page(1, blocks, nodes))
        self.assertEqual(build_hierarchical_text_chunks("manual", nodes, {1: blocks}), [])


class AutoMergeTest(unittest.TestCase):
    def test_two_of_three_direct_children_merge_to_parent(self) -> None:
        leaf_1 = SimpleNamespace(
            chunk_id="leaf-1",
            parent_chunk_id="parent",
        )
        leaf_2 = SimpleNamespace(
            chunk_id="leaf-2",
            parent_chunk_id="parent",
        )
        parent = SimpleNamespace(
            chunk_id="parent",
            parent_chunk_id=None,
            page_number=1,
            modality="text",
            section_title="使用和驾驶 > 启动车辆",
            content_text="方法A。方法B。方法C。",
            object_name=None,
            is_leaf=False,
            token_count=30,
            risk_level=None,
            image_path=None,
            thumbnail_path=None,
            icon_group_id=None,
        )
        db = MagicMock()
        all_results = [
            [leaf_1, leaf_2],
            [parent],
            [("parent", "leaf-1"), ("parent", "leaf-2"), ("parent", "leaf-3")],
            [parent],
        ]
        db.query.return_value.filter.return_value.all.side_effect = all_results
        candidates = [
            {
                "chunk_id": "leaf-1",
                "modality": "text",
                "token_count": 10,
                "score": 0.9,
                "match_method": "bm25",
            },
            {
                "chunk_id": "leaf-2",
                "modality": "text",
                "token_count": 10,
                "score": 0.8,
                "match_method": "dense",
            },
            {
                "chunk_id": "unrelated-large-candidate",
                "modality": "text",
                "token_count": 100,
                "score": 0.7,
                "match_method": "dense",
            },
        ]
        # The raw candidate pool is over budget, but the matched children must
        # still merge first; final evidence trimming happens afterwards.
        result = auto_merge_text_evidence(db, "manual", candidates, max_total_tokens=50)
        self.assertEqual([item["chunk_id"] for item in result], ["parent"])
        self.assertIn("auto_merge", result[0]["match_method"])


if __name__ == "__main__":
    unittest.main()
