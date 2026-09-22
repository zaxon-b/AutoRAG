import os
from typing import Any

import dashscope


TEXT_RERANK_MODEL = os.getenv("TEXT_RERANK_MODEL", "qwen3-rerank")
TEXT_RERANK_MAX_CHARS = int(os.getenv("TEXT_RERANK_MAX_CHARS", "1200"))
TEXT_RERANK_INSTRUCTION = os.getenv(
    "TEXT_RERANK_INSTRUCTION",
    "Given a vehicle owner manual question, rank passages that directly answer the question.",
)


class TextRerankError(RuntimeError):
    pass


def _document_text(candidate: dict[str, Any]) -> str:
    title = str(candidate.get("section_title") or "").strip()
    content = str(candidate.get("content_text") or "").strip()
    document = "\n".join(part for part in (title, content) if part)
    return document[:TEXT_RERANK_MAX_CHARS]


def rerank_text_candidates(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    documents = [_document_text(candidate) for candidate in candidates]
    response = dashscope.TextReRank.call(
        model=TEXT_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=TEXT_RERANK_INSTRUCTION,
    )
    if response.status_code != 200:
        raise TextRerankError(
            f"{response.code or response.status_code}: {response.message or 'text rerank request failed'}"
        )

    results = (response.output or {}).get("results") or []
    if len(results) != len(candidates):
        raise TextRerankError(
            f"text reranker returned {len(results)} results for {len(candidates)} candidates"
        )

    reranked: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for rank, result in enumerate(results, start=1):
        index = int(result.get("index"))
        if index < 0 or index >= len(candidates) or index in seen_indexes:
            raise TextRerankError(f"text reranker returned invalid candidate index: {index}")
        seen_indexes.add(index)

        item = dict(candidates[index])
        score = float(result.get("relevance_score") or 0.0)
        item["cross_encoder_score"] = score
        item["cross_encoder_rank"] = rank
        item["score"] = score
        methods = [method for method in str(item.get("match_method") or "").split("+") if method]
        if "cross_encoder" not in methods:
            methods.append("cross_encoder")
        item["match_method"] = "+".join(methods)
        reranked.append(item)
    return reranked
