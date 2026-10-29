"""文本 embedding：调用 text-embedding-v3，缺失 Key 时使用确定性兜底向量。"""

import hashlib
import os
import random

from openai import OpenAI


# 必须与 ES 的 dense_vector mapping 维度一致，否则索引写入会失败。
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-v3")
TEXT_EMBEDDING_BATCH_SIZE = int(os.getenv("TEXT_EMBEDDING_BATCH_SIZE", "10"))


def deterministic_embedding(text: str, dimensions: int = EMBEDDING_DIM) -> list[float]:
    """按文本哈希生成固定伪向量。

    没有 API Key 时也要让检索链路可运行；同一文本始终得到同一向量，
    避免建库和查询每次使用不同的随机数。
    """

    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return [rng.uniform(-0.01, 0.01) for _ in range(dimensions)]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量生成文本向量；任何一条失败都回退到确定性向量。"""

    if not texts:
        return []
    if not os.getenv("DASHSCOPE_API_KEY"):
        return [deterministic_embedding(text) for text in texts]

    vectors = _generate_dashscope_embeddings(texts)
    result: list[list[float]] = []
    for text, vector in zip(texts, vectors or []):
        if vector:
            result.append(vector)
        else:
            result.append(deterministic_embedding(text))

    # API 返回数量少于输入时补齐，保证调用方拿到的是一一对应的列表。
    while len(result) < len(texts):
        result.append(deterministic_embedding(texts[len(result)]))
    return result


def embed_text(text: str) -> list[float]:
    """单条文本的便捷封装。"""

    return embed_texts([text])[0]


def _generate_dashscope_embeddings(texts: list[str]) -> list[list[float] | None]:
    """通过 OpenAI 兼容接口调用 DashScope embedding。

    按 TEXT_EMBEDDING_BATCH_SIZE 分批；某一批失败时用 None 占位，
    由 embed_texts 统一替换成兜底向量。
    """

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    )
    embeddings: list[list[float] | None] = []
    for start in range(0, len(texts), TEXT_EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + TEXT_EMBEDDING_BATCH_SIZE]
        try:
            response = client.embeddings.create(
                model=TEXT_EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIM,
                encoding_format="float",
            )
            embeddings.extend([item.embedding for item in response.data])
        except Exception:
            embeddings.extend([None for _ in batch])
    return embeddings
