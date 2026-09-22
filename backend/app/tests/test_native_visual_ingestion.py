from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from service.es_index_service import _should_index
from service.icon_asset_grouping_service import _candidate_edges
from service.native_pdf_image_service import orientation_signature
from service.visual_asset_classification_service import (
    _representative_occurrence,
    classify_visual_assets,
)


def _unit(
    unit_id: str,
    *,
    phash: str,
    hue: float = 10.0,
    saturation: float = 100.0,
    vector: list[float] | None = None,
) -> dict:
    return {
        "unit_id": unit_id,
        "phash": phash,
        "colour": {"hue": hue, "saturation": saturation},
        "vector": vector,
    }


class NativeVisualIngestionTest(unittest.TestCase):
    def test_orientation_signature_keeps_direction_and_reflection(self) -> None:
        self.assertEqual(orientation_signature([1, 0, 0, 1, 0, 0]), "r0")
        self.assertEqual(orientation_signature([0, 1, -1, 0, 0, 0]), "r90")
        self.assertEqual(orientation_signature([-1, 0, 0, 1, 0, 0]), "r180_mirror")

    def test_colour_conflict_rejects_before_phash(self) -> None:
        left = _unit("a", phash="0000000000000000", hue=5)
        right = _unit("b", phash="0000000000000000", hue=80)
        self.assertEqual(_candidate_edges([left, right]), [])

    def test_close_phash_enters_vlm_candidate_zone(self) -> None:
        left = _unit("a", phash="0000000000000000")
        right = _unit("b", phash="0000000000000003")
        self.assertEqual(_candidate_edges([left, right]), [("a", "b")])

    def test_vector_can_propose_cross_fingerprint_candidate(self) -> None:
        left = _unit(
            "a",
            phash="0000000000000000",
            vector=[1.0, 0.0, 0.0],
        )
        right = _unit(
            "b",
            phash="ffffffffffffffff",
            vector=[0.99, 0.01, 0.0],
        )
        self.assertEqual(_candidate_edges([left, right]), [("a", "b")])

    def test_only_icon_prototypes_enter_visual_index(self) -> None:
        self.assertFalse(_should_index({"modality": "icon", "chunk_id": "group"}))
        self.assertTrue(
            _should_index(
                {
                    "modality": "icon",
                    "chunk_id": "group",
                    "prototype_id": "group_prototype_1",
                }
            )
        )
        self.assertTrue(_should_index({"modality": "text", "is_leaf": True}))
        self.assertFalse(_should_index({"modality": "text", "is_leaf": False}))

    def test_first_named_occurrence_supplies_vlm_context(self) -> None:
        occurrences = [
            {
                "page_number": 51,
                "bbox": [0, 0, 20, 20],
                "name_candidates": ["手册名称表中的名称"],
                "nearby_text": ["手册名称表中的名称"],
            },
            {
                "page_number": 200,
                "bbox": [0, 0, 100, 100],
                "name_candidates": ["点亮"],
                "nearby_text": ["点亮"],
            },
        ]
        self.assertEqual(
            _representative_occurrence(occurrences)["page_number"],
            51,
        )

    def test_cached_small_figure_is_individually_reviewed_as_possible_icon(self) -> None:
        asset = {
            "asset_id": "asset-1",
            "native_image_path": "/tmp/asset-1.png",
            "classification": "operation_figure",
            "classification_reason": "native_visual_v2_icon_semantics:batch",
            "_classification_cached": True,
        }
        occurrence = {
            "asset_id": "asset-1",
            "page_number": 49,
            "bbox": [239.0, 491.0, 268.0, 520.0],
            "page_width": 432.0,
            "page_height": 648.0,
            "orientation": "r0",
            "name_candidates": ["小灯指示灯"],
            "nearby_text": ["小灯指示灯"],
        }
        verdict = {
            "asset-1": {
                "classification": "icon",
                "confidence": 0.99,
                "object_name": None,
                "reason": "表格中有明确图标名称",
            }
        }
        with (
            patch.dict("os.environ", {"VLM_API_KEY": "test-key"}),
            patch(
                "service.visual_asset_classification_service.oriented_image_path",
                return_value=Path("/tmp/asset-1.png"),
            ),
            patch(
                "service.visual_asset_classification_service._classify_batch",
                return_value=verdict,
            ) as classify,
        ):
            classify_visual_assets([asset], [occurrence], variant_dir=Path("/tmp"))
        self.assertEqual(asset["classification"], "icon")
        self.assertIn("individual_icon_candidate_review", asset["classification_reason"])
        classify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
