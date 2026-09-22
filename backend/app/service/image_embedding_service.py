# 模块说明：图片 embedding 服务：把图标图片转成多模态向量，用于用户图标和手册图标之间的图片搜图片。
# 调用关系：由建库生成视觉 chunk 向量、图片问答检索用户图时调用。
# 返回/产出：返回图片向量；如果未配置模型则返回空向量。
# 代码说明：导入模块，os：读取环境变量，例如模型 key、阈值、数据库地址和存储目录。
import logging
import os
import time
import hashlib
# 代码说明：从 pathlib 导入：Path：使用 pathlib 统一处理本地文件和目录路径。
from pathlib import Path


# 代码说明：IMAGE_EMBEDDING_DIM 表示图片 embedding 向量维度，必须和 ES image_vector 字段维度一致；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
IMAGE_EMBEDDING_DIM = int(os.getenv("IMAGE_EMBEDDING_DIM", "768"))
# 代码说明：IMAGE_EMBEDDING_FIELD 表示ES 中保存图片向量的字段名；本行右侧表达式负责得到这个值。
IMAGE_EMBEDDING_FIELD = "image_embedding_vector"
IMAGE_EMBEDDING_V2_FIELD = "image_embedding_vector_v2"
IMAGE_EMBEDDING_PROFILE_NATIVE = "native"
IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE = "alpha_white"
# 代码说明：DEFAULT_IMAGE_EMBEDDING_MODEL 表示default / 图片 / 向量 / 模型名；本行右侧表达式负责得到这个值。
DEFAULT_IMAGE_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"
IMAGE_EMBEDDING_BATCH_SIZE = max(1, min(20, int(os.getenv("IMAGE_EMBEDDING_BATCH_SIZE", "10"))))
IMAGE_EMBEDDING_MAX_RETRIES = max(0, int(os.getenv("IMAGE_EMBEDDING_MAX_RETRIES", "3")))

logger = logging.getLogger(__name__)


# 函数说明：_clean_env：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _clean_env：封装当前模块的一段业务逻辑，供其他流程复用。
def _clean_env(value: str | None) -> str | None:
    # 代码说明：判断条件 `value is None` 是否成立，成立时执行下面缩进代码。
    if value is None:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None
    # 代码说明：value：当前正在清洗、追加或判断的单个值；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    value = value.strip().strip('"').strip("'")
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return value or None


# 函数说明：_api_key：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _api_key：封装当前模块的一段业务逻辑，供其他流程复用。
def _api_key() -> str | None:
    # 代码说明：返回调用结果：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里，调用方会拿这个结果继续下一步。
    return _clean_env(os.getenv("IMAGE_EMBEDDING_API_KEY")) or _clean_env(os.getenv("DASHSCOPE_API_KEY"))


# 函数说明：_model：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _model：封装当前模块的一段业务逻辑，供其他流程复用。
def _model() -> str:
    # 代码说明：返回调用结果：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里，调用方会拿这个结果继续下一步。
    return _clean_env(os.getenv("IMAGE_EMBEDDING_MODEL")) or DEFAULT_IMAGE_EMBEDDING_MODEL


# 函数说明：embed_image：为单张图片生成多模态 embedding，用于图片搜图标。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 embed_image：为单张图片生成多模态 embedding，用于图片搜图标。
def embed_image(
    image_path: str | Path | None,
    *,
    preprocessing: str = IMAGE_EMBEDDING_PROFILE_NATIVE,
) -> list[float] | None:
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    """Generate a real multimodal image embedding.

    This is intentionally backed by DashScope's multimodal embedding API, not
    by the local OpenCV shape features used for the separate opencv_icon_match
    fallback route.
    """
    # 代码说明：vectors：embedding 模型输出的向量列表；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    vectors = embed_images([image_path], preprocessing=preprocessing)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return vectors[0] if vectors else None


# 函数说明：embed_images：批量生成图片 embedding。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 embed_images：批量生成图片 embedding。
def embed_images(
    image_paths: list[str | Path | None],
    *,
    preprocessing: str = IMAGE_EMBEDDING_PROFILE_NATIVE,
) -> list[list[float] | None]:
    # 代码说明：进入异常保护块：下面的外部接口、文件、数据库或模型调用可能失败，需要捕获错误。
    try:
        # 代码说明：从 dashscope 导入：MultiModalEmbedding：本文件后续逻辑依赖的对象；MultiModalEmbeddingItemImage：本文件后续逻辑依赖的对象。
        from dashscope import MultiModalEmbedding, MultiModalEmbeddingItemImage
    # 代码说明：捕获异常 `Exception`，把底层错误转换成当前模块可处理的返回或提示。
    except Exception:
        # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
        return [None for _ in image_paths]

    # 代码说明：判断条件 `not _api_key()` 是否成立，成立时执行下面缩进代码。
    if not _api_key():
        # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
        return [None for _ in image_paths]

    # 代码说明：valid_positions 表示valid / positions；本行右侧表达式负责得到这个值。
    valid_positions: list[int] = []
    # 代码说明：valid_items 表示valid / 条目集合；本行右侧表达式负责得到这个值。
    valid_items = []
    # 代码说明：循环遍历 enumerate(image_paths)，每轮把当前元素放到 index, image_path 中处理。
    for index, image_path in enumerate(image_paths):
        # 代码说明：判断条件 `not image_path` 是否成立，成立时执行下面缩进代码。
        if not image_path:
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue
        # 代码说明：path：本地文件路径对象或字符串；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
        path = Path(image_path)
        # 代码说明：判断条件 `not path.exists()` 是否成立，成立时执行下面缩进代码。
        if not path.exists():
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue
        # 代码说明：把当前结果追加到 valid_positions 列表中；这表示它会参与后续排序、融合、返回或统计。
        embedding_path = prepare_image_embedding_input(path, preprocessing=preprocessing)
        if embedding_path is None or not embedding_path.exists():
            continue
        valid_positions.append(index)
        # 代码说明：valid_items 表示valid / 条目集合；本行右侧表达式负责得到这个值。
        valid_items.append(MultiModalEmbeddingItemImage(image=f"file://{embedding_path}", factor=1.0))

    # 代码说明：results 表示结果集合；本行右侧表达式负责得到这个值。
    results: list[list[float] | None] = [None for _ in image_paths]
    # 代码说明：判断条件 `not valid_items` 是否成立，成立时执行下面缩进代码。
    if not valid_items:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return results

    for start in range(0, len(valid_items), IMAGE_EMBEDDING_BATCH_SIZE):
        _embed_batch(
            MultiModalEmbedding,
            valid_positions[start : start + IMAGE_EMBEDDING_BATCH_SIZE],
            valid_items[start : start + IMAGE_EMBEDDING_BATCH_SIZE],
            results,
        )
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return results


def prepare_image_embedding_input(
    image_path: str | Path,
    *,
    preprocessing: str = IMAGE_EMBEDDING_PROFILE_NATIVE,
) -> Path | None:
    """Return the actual file sent to the image-embedding provider.

    ``native`` preserves the current production route. ``alpha_white`` creates
    a cached RGB PNG only when the source has an alpha channel. The cached file
    is an embedding input, never a second chunk or a second searchable object.
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
        if image.ndim != 3 or image.shape[2] != 4:
            return path

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
        if vector and len(vector) == IMAGE_EMBEDDING_DIM:
            results[position] = [float(value) for value in vector]
