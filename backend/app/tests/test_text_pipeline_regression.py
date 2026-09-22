from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from service.retrieval_service import retrieve_text_pipeline


def leaf(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "page": 1,
        "modality": "text",
        "section_title": "使用和驾驶 > 启动车辆",
        "content_text": text,
        "parent_chunk_id": "parent-1",
        "is_leaf": True,
        "token_count": 20,
    }


class TextPipelineRegressionTest(unittest.TestCase):
    def test_only_leaf_text_runs_bm25_dense_rrf_rerank_then_auto_merge(self) -> None:
        keyword = [leaf("k1", "关键词正文")]
        dense = [leaf("d1", "向量正文")]
        fused = [*keyword, *dense]
        reranked = [
            {**dense[0], "cross_encoder_score": 0.9},
            {**keyword[0], "cross_encoder_score": 0.8},
        ]
        with (
            patch("service.retrieval_service.embed_text", return_value=[0.1]),
            patch("service.retrieval_service.search_keyword_chunks", return_value=keyword) as bm25,
            patch("service.retrieval_service.search_text_vector_chunks", return_value=dense) as dense_search,
            patch("service.retrieval_service.rrf_fuse", return_value=fused) as rrf,
            patch("service.retrieval_service._same_page_text_candidates", return_value=[]),
            patch("service.retrieval_service.rerank_text_candidates", return_value=reranked) as rerank,
            patch("service.retrieval_service.auto_merge_text_evidence", return_value=reranked) as auto_merge,
        ):
            result = retrieve_text_pipeline(MagicMock(), "demo_manual", "启动车辆有几种方式", top_k=2)

        bm25.assert_called_once()
        dense_search.assert_called_once()
        rrf.assert_called_once()
        rerank.assert_called_once()
        auto_merge.assert_called_once()
        self.assertEqual([item["chunk_id"] for item in result["evidence"]], ["d1", "k1"])


if __name__ == "__main__":
    unittest.main()
