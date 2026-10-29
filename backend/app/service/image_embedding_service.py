"""图片 embedding：用 DashScope 多模态接口生成图片向量。

图片向量只用于“图片搜图片”的视觉召回；它与 OpenCV 形状特征、
多模态文本向量是三条独立路线。
"""

import logging
import os
import time
import hashlib
from pathlib import Path


# 必须与 ES dense_vector mapping 一致。
IMAGE_EMBEDDING_DIM = int(os.getenv("IMAGE_EMBEDDING_DIM", "768"))
IMAGE_EMBEDDING_FIELD = "image_embedding_vector"
# 白底预处理后的影子向量；与原生向量存在同一条 ES 文档上。
IMAGE_EMBEDDING_V2_FIELD = "image_embedding_vector_v2"
IMAGE_EMBEDDING_PROFILE_NATIVE = "native"
IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE = "alpha_white"
DEFAULT_IMAGE_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"
IMAGE_EMBEDDING_BATCH_SIZE = max(1, min(20, int(os.getenv("IMAGE_EMBEDDING_BATCH_SIZE", "10"))))
IMAGE_EMBEDDING_MAX_RETRIES = max(0, int(os.getenv("IMAGE_EMBEDDING_MAX_RETRIES", "3")))

logger = logging.getLogger(__name__)


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def _api_key() -> str | None:
    return _clean_env(os.getenv("IMAGE_EMBEDDING_API_KEY")) or _clean_env(os.getenv("DASHSCOPE_API_KEY"))


def _model() -> str:
    return _clean_env(os.getenv("IMAGE_EMBEDDING_MODEL")) or DEFAULT_IMAGE_EMBEDDING_MODEL


def embed_image(
    image_path: str | Path | None,
    *,
    preprocessing: str = IMAGE_EMBEDDING_PROFILE_NATIVE,
) -> list[float] | None:
    """单张图片的向量，是 embed_images 的便捷封装。"""

    vectors = embed_images([image_path], preprocessing=preprocessing)
    return vectors[0] if vectors else None


def embed_images(
    image_paths: list[str | Path | None],
    *,
    preprocessing: str = IMAGE_EMBEDDING_PROFILE_NATIVE,
) -> list[list[float] | None]:
    """批量生成图片向量，返回值与输入顺序一一对应。

    路径不存在、预处理失败或某批 API 调用失败的位置返回 None，
    而不是抛出异常中断整个建库流程。
    """

    try:
        from dashscope import MultiModalEmbedding, MultiModalEmbeddingItemImage
    except Exception:
        return [None for _ in image_paths]

    if not _api_key():
        return [None for _ in image_paths]

    valid_positions: list[int] = []
    valid_items = []
    for index, image_path in enumerate(image_paths):
        if not image_path:
            continue
        path = Path(image_path)
        if not path.exists():
            continue
        embedding_path = prepare_image_embedding_input(path, preprocessing=preprocessing)
        if embedding_path is None or not embedding_path.exists():
            continue
        valid_positions.append(index)
        valid_items.append(MultiModalEmbeddingItemImage(image=f"file://{embedding_path}", factor=1.0))

    results: list[list[float] | None] = [None for _ in image_paths]
    if not valid_items:
        return results

    for start in range(0, len(valid_items), IMAGE_EMBEDDING_BATCH_SIZE):
        _embed_batch(
            MultiModalEmbedding,
            valid_positions[start : start + IMAGE_EMBEDDING_BATCH_SIZE],
            valid_items[start : start + IMAGE_EMBEDDING_BATCH_SIZE],
            results,
        )
    return results


def prepare_image_embedding_input(
    image_path: str | Path,
    *,
    preprocessing: str = IMAGE_EMBEDDING_PROFILE_NATIVE,
) -> Path | None:
    """返回真正送给 embedding 服务的文件。

    native 直接使用原图；alpha_white 只对带透明通道的图生成白底缓存 PNG。
    缓存文件只是 embedding 输入，不会变成第二个 chunk 或第二条可检索对象。
    """

    path = Path(image_path)
    if not path.exists():
        return None
    if preprocessing == IMAGE_EMBEDDING_PROFILE_NATIVE:
        return path
    if preprocessing != IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE:
        raise ValueError(f"Unsupported image embedding preprocessing profile: {preprocessing}")

    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        # 没有 alpha 通道的图本身就是白底，无需再合成。
        if image.ndim != 3 or image.shape[2] != 4:
            return path

        # 用文件大小和修改时间做缓存键，源图变化后自动重建。
        stat = path.stat()
        cache_key = hashlib.sha1(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:12]
        target_dir = path.parent / "embedding_inputs_v2"
        target = target_dir / f"{path.stem}_{cache_key}.png"
        if target.exists():
            return target

        alpha = image[:, :, 3:4].astype("float32") / 255.0
        colour = image[:, :, :3].astype("float32")
        composited = np.clip(colour * alpha + 255.0 * (1.0 - alpha), 0, 255).astype("uint8")
        target_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), composited):
            return None
        return target
    except Exception as error:
        logger.warning(
            "Unable to prepare image embedding input: path=%s profile=%s error=%s",
            path,
            preprocessing,
            error,
        )
        return None


def _embed_batch(client, positions: list[int], items: list, results: list[list[float] | None]) -> None:
    """调用一批图片 embedding，并把成功结果写回原始下标。

    408/409/425/429 和 5xx 视为可重试错误，采用指数退避；其他错误
    直接放弃这一批，避免无意义等待。
    """

    response = None
    for attempt in range(1, IMAGE_EMBEDDING_MAX_RETRIES + 2):
        try:
            response = client.call(model=_model(), input=items, api_key=_api_key())
        except Exception as error:
            logger.warning("Image embedding request failed: size=%s attempt=%s error=%s", len(items), attempt, error)
            if attempt <= IMAGE_EMBEDDING_MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            return
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code == 200:
            break
        retryable = status_code in {408, 409, 425, 429} or status_code >= 500
        if retryable and attempt <= IMAGE_EMBEDDING_MAX_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 8))
            continue
        logger.warning(
            "Image embedding API returned status=%s code=%s message=%s",
            status_code,
            getattr(response, "code", None),
            getattr(response, "message", None),
        )
        return

    output = getattr(response, "output", None)
    embeddings = output.get("embeddings") if isinstance(output, dict) else None
    if not embeddings:
        return
    for position, item in zip(positions, embeddings):
        vector = item.get("embedding") if isinstance(item, dict) else None
        # 维度不符说明模型配置和索引 mapping 不一致，宁可留空也不写入错误向量。
        if vector and len(vector) == IMAGE_EMBEDDING_DIM:
            results[position] = [float(value) for value in vector]
