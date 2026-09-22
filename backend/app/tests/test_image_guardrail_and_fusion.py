from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from router.qa_rt import (
    _apply_retrieval_anchor,
    _apply_vlm_verification,
    _is_strong_visual_match,
    _should_refuse_image_answer,
    _verified_visual_anchor,
)
from service.visual_candidate_service import (
    VISUAL_MODALITIES,
    deduplicate_visual_candidates,
    gated_group_fusion,
    union_group_fusion,
)
from service.visual_similarity_service import retrieve_visual_chunks_by_multimodal
from service.vision_rerank_service import (
    _ambiguous_operation_candidates,
    _image_data_url,
    _validate_verification_payload,
)


def candidate(
    key: str,
    method: str,
    rank: int,
    *,
    modality: str = "icon",
) -> dict:
    return {
        "chunk_id": f"{key}_{method}",
        "icon_group_id": key if modality == "icon" else None,
        "_candidate_group_key": f"group:{key}",
        "modality": modality,
        "section_title": key,
        "score": 90 - rank,
        "match_method": method,
    }


class VisualVerificationPolicyTest(unittest.TestCase):
    def test_tiny_vlm_input_is_upscaled_in_memory_without_creating_a_file(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.png"
            cv2.imwrite(str(path), np.zeros((8, 12, 3), dtype=np.uint8))
            files_before = set(Path(directory).iterdir())

            data_url = _image_data_url(path)

            files_after = set(Path(directory).iterdir())
            payload = base64.b64decode(data_url.split(",", 1)[1])
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertEqual(files_after, files_before)
            self.assertEqual(max(decoded.shape[:2]), 256)

    @patch("service.vision_rerank_service._perceptual_fingerprint", return_value="same")
    def test_identical_operation_image_with_different_manual_meanings_is_ambiguous(self, _fingerprint) -> None:
        prepared = [
            {
                "chunk_id": "ev",
                "modality": "operation_figure",
                "object_name": "纯电动工作模式示意图",
                "image_path": Path("ev.png"),
            },
            {
                "chunk_id": "hev",
                "modality": "operation_figure",
                "object_name": "双模动力工作模式示意图",
                "image_path": Path("hev.png"),
            },
        ]

        result = _ambiguous_operation_candidates(Path("query.png"), prepared)

        self.assertEqual([item["chunk_id"] for item in result], ["ev", "hev"])

    def test_unselected_raw_similarity_is_not_an_anchor(self) -> None:
        vision = {"image_type": "icon", "vision_error": None}
        raw_match = {
            "modality": "icon",
            "score": 99.9,
            "vlm_verified": None,
        }
        self.assertFalse(_is_strong_visual_match(raw_match, vision))
        self.assertTrue(_should_refuse_image_answer(raw_match, vision))

    def test_production_rrf_top1_is_an_anchor_without_vlm_reranking(self) -> None:
        anchor = _apply_retrieval_anchor(
            {"chunk_id": "icon-1", "modality": "icon", "match_method": "rrf_union"}
        )
        self.assertTrue(_is_strong_visual_match(anchor, {"vision_error": None}))
        self.assertEqual(anchor["retrieval_anchor"], True)

    def test_operation_figure_is_not_rejected_by_first_pass_type(self) -> None:
        vision = {
            "image_type": "operation_figure",
            "confidence": 0.31,
            "vision_error": None,
        }
        match = {
            "modality": "operation_figure",
            "vlm_verified": True,
            "vlm_confidence": 0.92,
        }
        self.assertTrue(_is_strong_visual_match(match, vision))
        self.assertFalse(_should_refuse_image_answer(match, vision))

    def test_page_snapshot_can_be_visual_anchor(self) -> None:
        items = [{"chunk_id": "page-1", "modality": "page_snapshot", "section_title": "空调页面"}]
        verified = _apply_vlm_verification(
            items,
            {
                "matched": True,
                "selected_chunk_id": "page-1",
                "selected_title": "空调操作页面",
                "confidence": 0.94,
                "reason": "页面布局与OCR一致",
                "method": "vlm_visual_verification",
            },
        )
        anchor = _verified_visual_anchor(verified)
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["modality"], "page_snapshot")

    def test_none_of_the_above_produces_no_anchor(self) -> None:
        items = [{"chunk_id": "figure-1", "modality": "operation_figure"}]
        verified = _apply_vlm_verification(
            items,
            {
                "matched": False,
                "selected_chunk_id": None,
                "confidence": 0.88,
                "reason": "没有相同对象",
                "method": "vlm_visual_verification",
            },
        )
        self.assertIsNone(_verified_visual_anchor(verified))

    def test_second_pass_rejects_none_of_the_above(self) -> None:
        verdict = _validate_verification_payload(
            {
                "selected_index": None,
                "reject_all": True,
                "confidence": 0.99,
            },
            [{"index": 1, "chunk_id": "figure-1"}],
        )
        self.assertFalse(verdict["matched"])
        self.assertIsNone(verdict["selected_chunk_id"])

    def test_second_pass_maps_valid_index_to_server_chunk_id(self) -> None:
        verdict = _validate_verification_payload(
            {
                "selected_index": 1,
                "reject_all": False,
                "confidence": 0.96,
                "matching_features": ["相同座椅轮廓", "相同编号箭头位置"],
                "contradictions": [],
            },
            [{"index": 1, "chunk_id": "figure-1"}],
        )
        self.assertTrue(verdict["matched"])
        self.assertEqual(verdict["selected_chunk_id"], "figure-1")

    def test_second_pass_rejects_invalid_index(self) -> None:
        verdict = _validate_verification_payload(
            {"selected_index": 9, "reject_all": False, "confidence": 0.99},
            [{"index": 1, "chunk_id": "figure-1"}],
        )
        self.assertFalse(verdict["matched"])
        self.assertIsNone(verdict["selected_chunk_id"])

    def test_second_pass_rejects_missing_reject_all_flag(self) -> None:
        verdict = _validate_verification_payload(
            {"selected_index": 1, "confidence": 0.99},
            [{"index": 1, "chunk_id": "figure-1"}],
        )
        self.assertFalse(verdict["matched"])
        self.assertIsNone(verdict["selected_chunk_id"])


class VisualRouteFilteringTest(unittest.TestCase):
    def test_multimodal_filter_is_applied_in_search_request(self) -> None:
        with (
            patch(
                "service.visual_similarity_service.embed_multimodal_image",
                return_value=[0.1, 0.2],
            ),
            patch(
                "service.visual_similarity_service.search_multimodal_chunks",
                return_value=[],
            ) as search,
        ):
            retrieve_visual_chunks_by_multimodal("demo_manual", Path("query.png"), top_k=20)

        search.assert_called_once_with(
            manual_id="demo_manual",
            query_vector=[0.1, 0.2],
            top_k=20,
            modality_filter=sorted(VISUAL_MODALITIES),
        )

    def test_text_chunk_cannot_enter_deduplicated_visual_candidates(self) -> None:
        raw = [
            {"chunk_id": "text-1", "modality": "text", "section_title": "正文"},
            {
                "chunk_id": "figure-1",
                "modality": "operation_figure",
                "section_title": "开启前舱盖",
            },
        ]
        with (
            patch(
                "service.visual_candidate_service.annotate_icon_groups",
                return_value=raw,
            ),
            patch("service.visual_candidate_service._db_bind_key", return_value="db"),
            patch("service.visual_candidate_service._manual_icon_groups", return_value={}),
        ):
            result = deduplicate_visual_candidates(MagicMock(), "demo_manual", raw)
        self.assertEqual([item["chunk_id"] for item in result], ["figure-1"])

    def test_same_pixels_in_different_operation_occurrences_are_not_collapsed(self) -> None:
        raw = [
            {
                "chunk_id": "ev-figure",
                "modality": "operation_figure",
                "section_title": "纯电动工作模式",
                "image_path": "/same/image.png",
            },
            {
                "chunk_id": "hev-figure",
                "modality": "operation_figure",
                "section_title": "双模动力工作模式",
                "image_path": "/same/image.png",
            },
        ]
        with (
            patch("service.visual_candidate_service.annotate_icon_groups", return_value=raw),
            patch("service.visual_candidate_service._db_bind_key", return_value="db"),
            patch("service.visual_candidate_service._manual_icon_groups", return_value={}),
            patch("service.visual_candidate_service._perceptual_fingerprint", return_value="same"),
        ):
            result = deduplicate_visual_candidates(MagicMock(), "demo_manual", raw)

        self.assertEqual([item["chunk_id"] for item in result], ["ev-figure", "hev-figure"])


class GatedGroupFusionTest(unittest.TestCase):
    def test_forced_low_similarity_nearest_neighbour_is_rejected(self) -> None:
        weak = candidate("weak", "image_embedding", 1)
        weak["score"] = 28.8
        self.assertEqual(gated_group_fusion([weak], [], [], top_k=8), [])

    def test_secondary_routes_cannot_introduce_a_group(self) -> None:
        image = [candidate("image-a", "image_embedding", 1)]
        opencv = [candidate("opencv-only", "opencv_visual_match", 1)]
        multimodal = [candidate("multimodal-only", "multimodal_embedding", 1)]
        fused = gated_group_fusion(image, opencv, multimodal, top_k=8)
        self.assertEqual([item["icon_group_id"] for item in fused], ["image-a"])

    def test_opencv_alone_cannot_create_a_candidate_pool(self) -> None:
        opencv = [candidate("opencv-only", "opencv_visual_match", 1)]
        self.assertEqual(gated_group_fusion([], opencv, [], top_k=8), [])

    def test_cross_route_agreement_boosts_an_image_group(self) -> None:
        image = [
            candidate("image-a", "image_embedding", 1),
            candidate("image-b", "image_embedding", 2),
        ]
        multimodal = [candidate("image-b", "multimodal_embedding", 1)]
        fused = gated_group_fusion(image, [], multimodal, top_k=8)
        self.assertEqual(fused[0]["icon_group_id"], "image-b")
        self.assertIn("multimodal_embedding", fused[0]["agreement_routes"])


class UnionGroupFusionTest(unittest.TestCase):
    def test_either_embedding_route_can_introduce_a_candidate(self) -> None:
        image = [candidate("image-only", "image_embedding", 1)]
        multimodal = [candidate("multimodal-only", "multimodal_embedding", 1)]

        fused = union_group_fusion(image, [], multimodal, top_k=10)

        self.assertEqual(
            {item["icon_group_id"] for item in fused},
            {"image-only", "multimodal-only"},
        )

    def test_opencv_cannot_introduce_a_candidate(self) -> None:
        opencv = [candidate("opencv-only", "opencv_visual_match", 1)]
        self.assertEqual(union_group_fusion([], opencv, [], top_k=10, opencv_weight=0.2), [])

    def test_cross_route_agreement_is_ranked_ahead_of_single_route(self) -> None:
        image = [
            candidate("image-only", "image_embedding", 1),
            candidate("agreed", "image_embedding", 2),
        ]
        multimodal = [candidate("agreed", "multimodal_embedding", 1)]

        fused = union_group_fusion(image, [], multimodal, top_k=10)

        self.assertEqual(fused[0]["icon_group_id"], "agreed")
        self.assertEqual(
            fused[0]["agreement_routes"],
            ["image_embedding", "multimodal_embedding"],
        )


if __name__ == "__main__":
    unittest.main()
