"""多模态 embedding：把文本和图片映射到同一向量空间。

建库时为视觉 chunk 写入 ``multimodal_vector``；图片问答时把用户图片转成
query vector。没有 Key、SDK 缺失或模型调用失败时返回 None，调用方沿用
原有检索链路。
"""

import os
import logging
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


MULTIMODAL_EMBEDDING_DIM = int(os.getenv("MULTIMODAL_EMBEDDING_DIM", "1024"))
MULTIMODAL_EMBEDDING_FIELD = "multimodal_vector"
DEFAULT_MULTIMODAL_EMBEDDING_MODEL = "multimodal-embedding-v1"
MAX_MULTIMODAL_TEXT_BATCH_SIZE = 20
MULTIMODAL_TEXT_BATCH_SIZE = min(
    MAX_MULTIMODAL_TEXT_BATCH_SIZE,
    max(1, int(os.getenv("MULTIMODAL_EMBEDDING_TEXT_BATCH_SIZE", "20"))),
)
MULTIMODAL_TEXT_MAX_CHARS = max(100, int(os.getenv("MULTIMODAL_EMBEDDING_TEXT_MAX_CHARS", "500")))
MULTIMODAL_EMBEDDING_MAX_RETRIES = max(0, int(os.getenv("MULTIMODAL_EMBEDDING_MAX_RETRIES", "3")))
# The provider's image endpoint has a low request-rate quota.  One request at
# a time is the safe default; deployments with a higher quota can raise it.
MULTIMODAL_IMAGE_WORKERS = max(1, min(8, int(os.getenv("MULTIMODAL_EMBEDDING_IMAGE_WORKERS", "1"))))

logger = logging.getLogger(__name__)

MULTIMODAL_IMAGE_MODALITIES = {
    "icon",
    "operation_figure",
    "page_snapshot",
}


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def _api_key() -> str | None:
    return _clean_env(os.getenv("MULTIMODAL_EMBEDDING_API_KEY")) or _clean_env(os.getenv("DASHSCOPE_API_KEY"))


def _model() -> str:
    return _clean_env(os.getenv("MULTIMODAL_EMBEDDING_MODEL")) or DEFAULT_MULTIMODAL_EMBEDDING_MODEL


def embed_multimodal_image(image_path: str | Path | None) -> list[float] | None:
    vectors = embed_multimodal_items([{"image_path": image_path}])
    return vectors[0] if vectors else None


def embed_multimodal_chunks(chunks: list[dict[str, Any]]) -> list[list[float] | None]:
    items: list[dict[str, Any]] = []
    for chunk in chunks:
        image_path = chunk.get("image_path")
        modality = str(chunk.get("modality") or "")
        if image_path and modality in MULTIMODAL_IMAGE_MODALITIES:
            items.append(
                {
                    "image_path": _embedding_image_path(Path(str(image_path))),
                }
            )
            continue
        # Text leaves use the dedicated text embedding model. Parent text chunks
        # are PostgreSQL-only and visual chunks use their actual image.
        items.append({})
    return embed_multimodal_items(items)


def _embedding_image_path(path: Path) -> Path:
    """Create a cached provider-safe JPEG without changing evidence images."""

    max_bytes = 2_800_000
    try:
        if path.stat().st_size <= max_bytes:
            return path
    except OSError:
        return path
    target_dir = path.parent / "embedding_inputs"
    target = target_dir / f"{path.stem}.jpg"
    try:
        if target.exists() and target.stat().st_size <= max_bytes:
            return target
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return path
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype("float32") / 255.0
            image = (
                image[:, :, :3].astype("float32") * alpha
                + 255.0 * (1.0 - alpha)
            ).astype("uint8")
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > 1800:
            scale = 1800.0 / longest
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        for quality in (88, 80, 72, 64):
            if cv2.imwrite(
                str(target),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            ) and target.stat().st_size <= max_bytes:
                return target
        return target if target.exists() else path
    except Exception as error:
        logger.warning("Unable to prepare embedding image: path=%s error=%s", path, error)
        return path


def embed_multimodal_items(items: list[dict[str, Any]]) -> list[list[float] | None]:
    try:
        from dashscope import MultiModalEmbedding, MultiModalEmbeddingItemImage
    except Exception as error:
        logger.exception("Unable to import the DashScope multimodal embedding SDK: %s", error)
        return [None for _ in items]
    try:
        from dashscope import MultiModalEmbeddingItemText
    except Exception:
        MultiModalEmbeddingItemText = None

    api_key = _api_key()
    if not api_key:
        logger.error(
            "Multimodal embedding is disabled because neither "
            "MULTIMODAL_EMBEDDING_API_KEY nor DASHSCOPE_API_KEY is configured."
        )
        return [None for _ in items]

    text_entries: list[tuple[int, Any]] = []
    image_entries: list[tuple[int, Any]] = []
    for index, item in enumerate(items):
        image_path = item.get("image_path")
        if image_path:
            path = Path(image_path)
            if not path.exists():
                logger.error("Multimodal image input does not exist: position=%s path=%s", index, path)
                continue
            image_entries.append((index, MultiModalEmbeddingItemImage(image=f"file://{path}", factor=1.0)))
            continue

        text = " ".join(str(item.get("text") or "").split())
        if text:
            text = textwrap.shorten(text, width=MULTIMODAL_TEXT_MAX_CHARS, placeholder="...")
            if MultiModalEmbeddingItemText is not None:
                text_entries.append((index, MultiModalEmbeddingItemText(text=text, factor=1.0)))
            else:
                text_entries.append((index, {"text": text, "factor": 1.0}))

    results: list[list[float] | None] = [None for _ in items]
    if not text_entries and not image_entries:
        return results

    # multimodal-embedding-v1 accepts at most 20 text elements per request and
    # at most one image per request.  Keep the two modalities separate so that
    # a large manual cannot invalidate the entire ingestion request.
    for start in range(0, len(text_entries), MULTIMODAL_TEXT_BATCH_SIZE):
        _embed_batch(
            MultiModalEmbedding,
            api_key,
            "text",
            text_entries[start : start + MULTIMODAL_TEXT_BATCH_SIZE],
            results,
        )
    if image_entries:
        with ThreadPoolExecutor(max_workers=min(MULTIMODAL_IMAGE_WORKERS, len(image_entries))) as executor:
            futures = [
                executor.submit(_embed_batch, MultiModalEmbedding, api_key, "image", [entry], results)
                for entry in image_entries
            ]
            for future in futures:
                future.result()

    failed = sum(vector is None for vector in results)
    if failed:
        logger.warning(
            "Multimodal embedding completed with missing vectors: total=%s succeeded=%s failed=%s",
            len(results),
            len(results) - failed,
            failed,
        )
    return results


def _embed_batch(
    embedding_client: Any,
    api_key: str,
    kind: str,
    entries: list[tuple[int, Any]],
    results: list[list[float] | None],
) -> None:
    positions = [position for position, _ in entries]
    request_items = [item for _, item in entries]
    response = None
    for attempt in range(1, MULTIMODAL_EMBEDDING_MAX_RETRIES + 2):
        try:
            response = embedding_client.call(model=_model(), input=request_items, api_key=api_key)
        except Exception as error:
            logger.exception(
                "Multimodal embedding API raised an exception: kind=%s batch_size=%s attempt=%s/%s error=%s",
                kind,
                len(entries),
                attempt,
                MULTIMODAL_EMBEDDING_MAX_RETRIES + 1,
                error,
            )
            if attempt <= MULTIMODAL_EMBEDDING_MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            return

        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code == 200:
            break

        code = getattr(response, "code", None)
        message = getattr(response, "message", None)
        logger.error(
            "Multimodal embedding API request failed: kind=%s batch_size=%s attempt=%s/%s "
            "status_code=%s code=%s message=%s",
            kind,
            len(entries),
            attempt,
            MULTIMODAL_EMBEDDING_MAX_RETRIES + 1,
            status_code,
            code,
            message,
        )
        retryable = status_code in {408, 409, 425, 429} or status_code >= 500
        if retryable and attempt <= MULTIMODAL_EMBEDDING_MAX_RETRIES:
            # RateQuota needs a longer cool-down than transient 5xx errors.
            delay = min(5 * (2 ** (attempt - 1)), 30) if status_code == 429 else min(2 ** (attempt - 1), 8)
            time.sleep(delay)
            continue
        return

    output = getattr(response, "output", None)
    if not isinstance(output, dict):
        logger.error(
            "Multimodal embedding API returned an invalid output: kind=%s batch_size=%s output_type=%s",
            kind,
            len(entries),
            type(output).__name__,
        )
        return
    embeddings = output.get("embeddings")
    if not embeddings:
        logger.error(
            "Multimodal embedding API returned no embeddings: kind=%s batch_size=%s",
            kind,
            len(entries),
        )
        return
    if len(embeddings) != len(entries):
        logger.error(
            "Multimodal embedding count mismatch: kind=%s requested=%s returned=%s",
            kind,
            len(entries),
            len(embeddings),
        )

    for position, item in zip(positions, embeddings):
        vector = item.get("embedding") if isinstance(item, dict) else None
        if vector and len(vector) == MULTIMODAL_EMBEDDING_DIM:
            results[position] = [float(value) for value in vector]
        else:
            logger.error(
                "Multimodal embedding has an invalid dimension: kind=%s position=%s expected=%s actual=%s",
                kind,
                position,
                MULTIMODAL_EMBEDDING_DIM,
                len(vector) if vector else None,
            )
