from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from router.qa_rt import (
    _no_visual_candidate_failure_message,
    _retrieve_merged_evidence,
    _retrieve_visual_routes,
    image_qa,
)


class FakeUpload:
    filename = "query.png"

    async def read(self) -> bytes:
        return b"image-bytes"


def vision(image_type: str = "operation_figure") -> dict:
    return {
        "recognized_object": "前舱盖开启步骤",
        "visual_description": "展示打开前舱盖的两步操作示意图",
        "ocr_text": "拉起手柄",
        "candidate_names": ["开启前舱盖"],
        "visual_keywords": ["前舱盖", "拉手"],
        "normalized_query": "开启前舱盖 拉手 操作步骤",
        "image_type": image_type,
        "confidence": 0.55,
        "risk_level": "unknown",
        "is_target_icon": False,
        "classification_confidence": 0.55,
        "classification_reason": "metadata only",
        "vision_error": None,
    }


def visual_candidate() -> dict:
    return {
        "chunk_id": "figure-1",
        "page": 362,
        "modality": "operation_figure",
        "section_title": "控制器的操作 > 前舱盖 > 开启前舱盖",
        "object_name": "前舱盖开启操作图",
        "content_text": "拉起前舱盖开启手柄，然后拨动锁扣。",
        "score": 92.0,
        "match_method": "image_embedding",
        "_candidate_group_key": "chunk:figure-1",
    }


class ParallelVisualRetrievalTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_production_visual_routes_execute(self) -> None:
        with (
            patch("router.qa_rt.retrieve_visual_chunks_by_image_embedding", return_value=[{"chunk_id": "i"}]) as image,
            patch("router.qa_rt.retrieve_visual_chunks_by_multimodal", return_value=[{"chunk_id": "m"}]) as multimodal,
        ):
            result = await _retrieve_visual_routes(
                MagicMock(),
                "demo_manual",
                Path("query.png"),
                top_k=20,
            )

        self.assertEqual(set(result), {"image_embedding", "multimodal_embedding", "opencv"})
        image.assert_called_once_with(
            "demo_manual",
            Path("query.png"),
            20,
            embedding_variant="alpha_white",
        )
        multimodal.assert_called_once_with("demo_manual", Path("query.png"), 20)
        self.assertEqual(result["opencv"], [])


class ImageQaOrchestrationTest(unittest.IsolatedAsyncioTestCase):
    def common_patches(self, text_items: list[dict] | None = None):
        candidate = visual_candidate()
        return (
            patch("router.qa_rt.save_query_image", return_value=Path("/tmp/query.png")),
            patch("router.qa_rt.understand_vehicle_image", return_value=vision()),
            patch(
                "router.qa_rt._retrieve_visual_routes",
                new=AsyncMock(
                    return_value={
                        "image_embedding": [candidate],
                        "multimodal_embedding": [candidate],
                        "opencv": [candidate],
                    }
                ),
            ),
            patch(
                "router.qa_rt.deduplicate_visual_candidates",
                side_effect=lambda db, manual_id, items, max_items: items,
            ),
            patch("router.qa_rt.union_group_fusion", return_value=[candidate]),
            patch(
                "router.qa_rt.retrieve_visual_anchor_text",
                return_value=text_items or [],
            ),
            patch("router.qa_rt.complete_answer", return_value="grounded answer"),
        )

    async def test_opt_in_vlm_rejection_never_calls_final_llm_or_text_retrieval(self) -> None:
        verification = {
            "matched": False,
            "selected_chunk_id": None,
            "confidence": 0.9,
            "reason": "没有相同对象",
            "method": "vlm_visual_verification",
        }
        patches = self.common_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
            "router.qa_rt.ENABLE_VLM_VISUAL_VERIFICATION", True
        ), patch(
            "router.qa_rt.verify_visual_candidates", return_value=verification
        ), patches[5] as text_retrieval, patches[6] as llm:
            response = await image_qa(
                manual_id="demo_manual",
                question="怎么打开？",
                image=FakeUpload(),
                db=MagicMock(),
            )

        text_retrieval.assert_not_called()
        llm.assert_not_called()
        self.assertEqual(response["evidence"], [])
        self.assertEqual(response["evidence_status"], "manual_visual_not_found")

    async def test_production_top1_anchor_runs_text_retrieval_without_vlm_reranking(self) -> None:
        text_items = [
            {
                "chunk_id": f"text-{index}",
                "page": 362 + index,
                "modality": "text",
                "content_text": f"正文 {index}",
            }
            for index in range(6)
        ]
        patches = self.common_patches(text_items)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patch(
            "router.qa_rt.verify_visual_candidates"
        ) as verifier, patches[5] as text_retrieval, patches[6] as llm:
            response = await image_qa(
                manual_id="demo_manual",
                question="怎么打开？",
                image=FakeUpload(),
                db=MagicMock(),
            )

        verifier.assert_not_called()
        text_retrieval.assert_called_once()
        llm.assert_called_once()
        self.assertEqual(len(response["evidence"]), 5)
        self.assertEqual(response["evidence"][0]["evidence_role"], "visual")
        self.assertTrue(all(item["evidence_role"] == "text" for item in response["evidence"][1:]))
        self.assertEqual(len(response["text_evidence"]), 4)
        self.assertEqual(response["visual_evidence"]["modality"], "operation_figure")
        self.assertTrue(response["visual_evidence"]["retrieval_anchor"])

    async def test_missing_body_evidence_does_not_call_final_llm(self) -> None:
        patches = self.common_patches([])
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as text_retrieval, patches[6] as llm:
            response = await image_qa(
                manual_id="demo_manual",
                question="怎么打开？",
                image=FakeUpload(),
                db=MagicMock(),
            )

        text_retrieval.assert_called_once()
        llm.assert_not_called()
        self.assertEqual(response["evidence_status"], "insufficient_text_evidence")
        self.assertEqual(len(response["evidence"]), 1)
        self.assertEqual(response["evidence"][0]["evidence_role"], "visual")


class FailureMessageTest(unittest.TestCase):
    def test_no_candidate_statuses_are_distinct(self) -> None:
        unrelated = _no_visual_candidate_failure_message(
            {"image_type": "unknown", "visual_description": "绿色田野中的紫色房屋"}
        )
        blurry = _no_visual_candidate_failure_message(
            {"image_type": "unknown", "visual_description": "图片模糊无法辨认"}
        )
        missing = _no_visual_candidate_failure_message(
            {"image_type": "component_photo", "visual_description": "车辆充电接口照片"}
        )
        self.assertEqual(unrelated[0], "image_unrelated")
        self.assertEqual(blurry[0], "image_too_blurry")
        self.assertEqual(missing[0], "manual_visual_not_found")


class EvidenceAssemblyTest(unittest.TestCase):
    def test_raw_multimodal_results_do_not_join_final_cross_modal_rrf(self) -> None:
        anchor = {
            **visual_candidate(),
            "vlm_verified": True,
            "vlm_confidence": 0.95,
        }
        raw_multimodal_text = [{"chunk_id": "raw-text", "modality": "text"}]
        body = [{"chunk_id": "body-1", "page": 362, "modality": "text"}]
        with (
            patch("router.qa_rt.retrieve_visual_anchor_text", return_value=body),
            patch("router.qa_rt.rrf_fuse") as rrf,
        ):
            evidence = _retrieve_merged_evidence(
                MagicMock(),
                "demo_manual",
                ["开启前舱盖"],
                [anchor],
                raw_multimodal_text,
            )
        rrf.assert_not_called()
        self.assertEqual([item["chunk_id"] for item in evidence], ["figure-1", "body-1"])


if __name__ == "__main__":
    unittest.main()
