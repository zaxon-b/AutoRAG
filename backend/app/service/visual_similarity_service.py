# 模块说明：视觉相似度服务：同时支持图片 embedding 检索和 OpenCV 传统图像特征匹配。
# 调用关系：由 qa_rt.image_qa 的视觉召回阶段调用。
# 返回/产出：返回候选图标 evidence，包括页码、图标名、图片路径、匹配分和匹配方法。
# 代码说明：从 pathlib 导入：Path：使用 pathlib 统一处理本地文件和目录路径。
from pathlib import Path

# 代码说明：从 sqlalchemy.orm 导入：Session：SQLAlchemy 数据库会话类型。
from sqlalchemy.orm import Session

# 代码说明：从 models.manual 导入数据库视觉对象。
from models.manual import ManualChunk, ManualIconPrototype
# 代码说明：从 service.es_index_service 导入：search_image_chunks：本文件后续逻辑依赖的对象。
from service.es_index_service import search_image_chunks, search_multimodal_chunks
# 代码说明：从 service.icon_catalog_service 导入：canonicalize_evidence：统一 evidence 字段和标准图标名，减少不同来源结果的结构差异。
from service.icon_catalog_service import canonicalize_evidence
# 代码说明：从 service.image_embedding_service 导入：embed_image：为单张图片生成多模态 embedding，用于图片搜图标。
from service.image_embedding_service import (
    IMAGE_EMBEDDING_FIELD,
    IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
    IMAGE_EMBEDDING_PROFILE_NATIVE,
    IMAGE_EMBEDDING_V2_FIELD,
    embed_image,
)
# 代码说明：从 service.manual_storage_service 导入：to_storage_url：把本地文件路径转换成前端可访问的 /storage URL。
from service.manual_storage_service import to_storage_url
from service.multimodal_embedding_service import embed_multimodal_image
from service.visual_candidate_service import VISUAL_MODALITIES

# 代码说明：IMAGE_EMBEDDING_MODALITIES 表示图片 / 向量 / 模态集合；本行右侧表达式负责得到这个值。
IMAGE_EMBEDDING_MODALITIES = sorted(VISUAL_MODALITIES)

# 代码说明：OPENCV_MATCH_MODALITIES 表示opencv / match / 模态集合；本行右侧表达式负责得到这个值。
OPENCV_MATCH_MODALITIES = sorted(VISUAL_MODALITIES)


# 函数说明：retrieve_visual_chunks_by_image_embedding：用用户图片向量检索四类 schema 中的视觉 chunk。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 retrieve_visual_chunks_by_image_embedding：用用户图片向量检索视觉 chunk。
def retrieve_visual_chunks_by_image_embedding(
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
    *,
    embedding_variant: str = "native",
) -> list[dict]:
    """Query the image vector over actual visual modalities only."""
    # Real visual semantic retrieval: compare the user's image embedding against
    # image_embedding_vector stored on manual visual chunks.
    # 代码说明：query_vector 表示查询 / 向量；本行右侧表达式负责得到这个值。
    if embedding_variant == "native":
        preprocessing = IMAGE_EMBEDDING_PROFILE_NATIVE
        vector_field = IMAGE_EMBEDDING_FIELD
    elif embedding_variant == "alpha_white":
        preprocessing = IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE
        vector_field = IMAGE_EMBEDDING_V2_FIELD
    else:
        raise ValueError(f"Unsupported image embedding variant: {embedding_variant}")
    query_vector = embed_image(query_image_path, preprocessing=preprocessing)
    # 代码说明：判断条件 `not query_vector` 是否成立，成立时执行下面缩进代码。
    if not query_vector:
        # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
        return []

    # 代码说明：chunks：建库过程中生成的多模态 chunk 列表；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    search_kwargs = {
        # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
        "manual_id": manual_id,
        # 代码说明：image_vector：图片 embedding 向量，用于图片搜图片；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
        "image_vector": query_vector,
        # 代码说明：top_k：top_k 控制每条召回路线最多取多少个候选，防止候选过多拖慢后续验证。
        "top_k": top_k,
        # 代码说明：modalities 表示模态集合；本行右侧表达式负责得到这个值。
        "modalities": IMAGE_EMBEDDING_MODALITIES,
    }
    if vector_field != IMAGE_EMBEDDING_FIELD:
        search_kwargs["vector_field"] = vector_field
    chunks = search_image_chunks(**search_kwargs)
    # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
    return [_format_es_icon_evidence(chunk) for chunk in chunks]


def retrieve_visual_chunks_by_image_embedding_v2(
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
) -> list[dict]:
    """Search the shadow alpha-on-white vector field for controlled ablation."""

    return retrieve_visual_chunks_by_image_embedding(
        manual_id,
        query_image_path,
        top_k=top_k,
        embedding_variant="alpha_white",
    )


def retrieve_image_embedding_icons(manual_id: str, query_image_path: Path, top_k: int = 5) -> list[dict]:
    """Compatibility wrapper for the former icon-only API."""

    return retrieve_visual_chunks_by_image_embedding(manual_id, query_image_path, top_k=top_k)


def retrieve_visual_chunks_by_multimodal(
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
) -> list[dict]:
    """Query multimodal vectors with a visual filter in the ES request."""

    query_vector = embed_multimodal_image(query_image_path)
    if not query_vector:
        return []

    chunks = search_multimodal_chunks(
        manual_id=manual_id,
        query_vector=query_vector,
        top_k=top_k,
        modality_filter=sorted(VISUAL_MODALITIES),
    )
    return [_format_multimodal_evidence(chunk) for chunk in chunks]


def retrieve_multimodal_chunks(manual_id: str, query_image_path: Path, top_k: int = 8) -> list[dict]:
    """Compatibility wrapper; results are now visual chunks only."""

    return retrieve_visual_chunks_by_multimodal(manual_id, query_image_path, top_k=top_k)


# 函数说明：retrieve_similar_icons：用 OpenCV 特征比较用户图片和数据库中的图标裁剪图。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 retrieve_similar_icons：用 OpenCV 特征比较用户图片和数据库中的图标裁剪图。
def retrieve_visual_chunks_by_opencv(
    db: Session,
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
) -> list[dict]:
    """Return weak shape/colour matches from visual chunks only."""
    # OpenCV is intentionally only an auxiliary route. It compares low-level
    # shape/color features and cannot decide the official icon meaning alone.
    # 代码说明：query_feature 表示查询 / feature；本行右侧表达式负责得到这个值。
    query_feature = _extract_icon_feature(query_image_path)
    # 代码说明：判断条件 `query_feature is None` 是否成立，成立时执行下面缩进代码。
    if query_feature is None:
        # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
        return []

    # 代码说明：rows 表示行集合；本行右侧表达式负责得到这个值。
    rows = (
        # 代码说明：调用 db.query 执行当前步骤的副作用，例如写库、建索引、保存文件或更新状态。
        db.query(ManualChunk)
        # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality.in_(("operation_figure", "page_snapshot")),
        )
        # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
        .all()
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    )
    # 代码说明：scored 表示scored；本行右侧表达式负责得到这个值。
    scored: list[tuple[float, dict]] = []
    # 代码说明：循环遍历 rows，每轮把当前元素放到 row 中处理。
    for row in rows:
        # 代码说明：判断条件 `not row.image_path` 是否成立，成立时执行下面缩进代码。
        if not row.image_path:
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue
        # 代码说明：candidate_feature 表示候选 / feature；本行右侧表达式负责得到这个值。
        candidate_feature = _extract_icon_feature(Path(row.image_path))
        # 代码说明：判断条件 `candidate_feature is None` 是否成立，成立时执行下面缩进代码。
        if candidate_feature is None:
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue
        # 代码说明：score：候选图标的视觉匹配分数；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
        score = _feature_similarity(query_feature, candidate_feature)
        # 代码说明：判断条件 `score >= 0.35` 是否成立，成立时执行下面缩进代码。
        if score >= 0.35:
            # 代码说明：把当前结果追加到 scored 列表中；这表示它会参与后续排序、融合、返回或统计。
            scored.append((score, _row_evidence(row)))

    prototypes = (
        db.query(ManualIconPrototype)
        .filter(ManualIconPrototype.manual_id == manual_id)
        .order_by(ManualIconPrototype.icon_group_id, ManualIconPrototype.prototype_rank)
        .all()
    )
    group_chunk_ids = [row.group_chunk_id for row in prototypes]
    group_chunks = (
        db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.chunk_id.in_(group_chunk_ids),
            ManualChunk.modality == "icon",
        )
        .all()
        if group_chunk_ids
        else []
    )
    group_chunk_by_id = {row.chunk_id: row for row in group_chunks}
    for prototype in prototypes:
        candidate_feature = _extract_icon_feature(Path(prototype.image_path))
        if candidate_feature is None:
            continue
        score = _feature_similarity(query_feature, candidate_feature)
        if score < 0.35:
            continue
        group_chunk = group_chunk_by_id.get(prototype.group_chunk_id)
        if group_chunk is None:
            continue
        evidence = _row_evidence(group_chunk)
        evidence["image_path"] = prototype.image_path
        evidence["thumbnail_path"] = prototype.image_path
        evidence["prototype_id"] = prototype.prototype_id
        evidence["asset_id"] = prototype.asset_id
        scored.append((score, evidence))

    # 代码说明：scored 表示scored；本行右侧表达式负责得到这个值。
    scored.sort(key=lambda item: item[0], reverse=True)
    # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
    return [_format_opencv_evidence(item, score) for score, item in scored[:top_k]]


def retrieve_similar_icons(db: Session, manual_id: str, query_image_path: Path, top_k: int = 5) -> list[dict]:
    """Compatibility wrapper for the former icon-only API."""

    return retrieve_visual_chunks_by_opencv(db, manual_id, query_image_path, top_k=top_k)


# 函数说明：_format_es_icon_evidence：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _format_es_icon_evidence：封装当前模块的一段业务逻辑，供其他流程复用。
def _format_es_icon_evidence(chunk: dict) -> dict:
    # 代码说明：raw_score 表示原始值 / 分数；这里调用 `float` 得到它，调用目的：把环境变量字符串转换成数字，方便后续比较阈值。
    raw_score = float(chunk.get("_score") or chunk.get("score") or 0)
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    score = max(0.0, min(100.0, (raw_score - 1.0) * 100.0 if raw_score <= 2.1 else raw_score))
    # 代码说明：返回调用结果：统一 evidence 字段和标准图标名，减少不同来源结果的结构差异，调用方会拿这个结果继续下一步。
    return canonicalize_evidence({
        # 代码说明：`"chunk_id": chunk.get("chunk_id") or chunk.get("es_doc_id") or ""` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "chunk_id": chunk.get("chunk_id") or chunk.get("es_doc_id") or "",
        # 代码说明：设置字典键 `page` 的值，用来表达接口字段、配置项或业务映射关系。
        "page": int(chunk.get("page") or chunk.get("page_number") or 0),
        # ES 文档一定包含 modality；缺失时使用合法的视觉类型兜底。
        "modality": chunk.get("modality") or "icon",
        # 代码说明：设置字典键 `section_title` 的值，用来表达接口字段、配置项或业务映射关系。
        "section_title": chunk.get("section_title"),
        # 代码说明：字典字段 `content_text`：content_text 保存 chunk 的主文本内容，LLM 回答和 BM25/文本向量检索主要依赖它。
        "content_text": chunk.get("content_text"),
        "object_name": chunk.get("object_name"),
        "icon_group_id": chunk.get("icon_group_id"),
        "prototype_id": chunk.get("prototype_id"),
        "asset_id": chunk.get("asset_id"),
        # 代码说明：字典字段 `risk_level`：risk_level 保存风险等级，前端用它显示不同颜色提示。
        "risk_level": chunk.get("risk_level"),
        # 代码说明：设置字典键 `image_path` 的值，用来表达接口字段、配置项或业务映射关系。
        "image_path": to_storage_url(chunk.get("image_path")),
        # 代码说明：设置字典键 `thumbnail_path` 的值，用来表达接口字段、配置项或业务映射关系。
        "thumbnail_path": to_storage_url(chunk.get("thumbnail_path")),
        # 代码说明：设置字典键 `score` 的值，用来表达接口字段、配置项或业务映射关系。
        "score": round(score, 2),
        # 代码说明：`"match_method": "image_embedding"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "match_method": "image_embedding",
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    })


def _format_multimodal_evidence(chunk: dict) -> dict:
    raw_score = float(chunk.get("_score") or chunk.get("score") or 0)
    score = max(0.0, min(100.0, (raw_score - 1.0) * 100.0 if raw_score <= 2.1 else raw_score))
    return canonicalize_evidence({
        "chunk_id": chunk.get("chunk_id") or chunk.get("es_doc_id") or "",
        "page": int(chunk.get("page") or chunk.get("page_number") or 0),
        "modality": chunk.get("modality") or "page_snapshot",
        "section_title": chunk.get("section_title"),
        "content_text": chunk.get("content_text"),
        "object_name": chunk.get("object_name"),
        "icon_group_id": chunk.get("icon_group_id"),
        "prototype_id": chunk.get("prototype_id"),
        "asset_id": chunk.get("asset_id"),
        "risk_level": chunk.get("risk_level"),
        "image_path": to_storage_url(chunk.get("image_path")),
        "thumbnail_path": to_storage_url(chunk.get("thumbnail_path")),
        "score": round(score, 2),
        "match_method": "multimodal_embedding",
    })


# 函数说明：_format_icon_evidence：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _format_icon_evidence：封装当前模块的一段业务逻辑，供其他流程复用。
def _row_evidence(row: ManualChunk) -> dict:
    return {
        "chunk_id": row.chunk_id,
        "page": row.page_number,
        "modality": row.modality,
        "section_title": row.section_title,
        "content_text": row.content_text,
        "object_name": row.object_name,
        "icon_group_id": row.icon_group_id,
        "risk_level": row.risk_level,
        "image_path": row.image_path,
        "thumbnail_path": row.thumbnail_path,
    }


def _format_opencv_evidence(item: dict, score: float) -> dict:
    return canonicalize_evidence(
        {
            **item,
            "score": round(score * 100, 2),
            "match_method": "opencv_visual_match",
        }
    )


def _format_icon_evidence(row: ManualChunk, score: float) -> dict:
    return _format_opencv_evidence(_row_evidence(row), score)


# 函数说明：_extract_icon_feature：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _extract_icon_feature：封装当前模块的一段业务逻辑，供其他流程复用。
def _extract_icon_feature(image_path: Path) -> dict | None:
    # 代码说明：进入异常保护块：下面的外部接口、文件、数据库或模型调用可能失败，需要捕获错误。
    try:
        # 代码说明：导入模块，cv2：提供本模块需要的标准库/第三方能力。
        import cv2
        # 代码说明：导入模块，numpy：提供本模块需要的标准库/第三方能力。
        import numpy as np
    # 代码说明：捕获异常 `Exception`，把底层错误转换成当前模块可处理的返回或提示。
    except Exception:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None

    # 代码说明：image：FastAPI 接收到的上传图片文件对象；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    image = cv2.imread(str(image_path))
    # 代码说明：判断条件 `image is None` 是否成立，成立时执行下面缩进代码。
    if image is None:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None

    # 代码说明：mask 表示掩码；本行右侧表达式负责得到这个值。
    mask = _icon_mask(cv2, image)
    # 代码说明：coords 表示coords；本行右侧表达式负责得到这个值。
    coords = cv2.findNonZero(mask)
    # 代码说明：判断条件 `coords is None` 是否成立，成立时执行下面缩进代码。
    if coords is None:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None

    # 代码说明：x, y, width, height 表示一组同时解包出来的值：x、y、width、height；本行右侧表达式负责得到这个值。
    x, y, width, height = cv2.boundingRect(coords)
    # 代码说明：判断条件 `width < 6 or height < 6` 是否成立，成立时执行下面缩进代码。
    if width < 6 or height < 6:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None

    # 代码说明：cropped_mask 表示cropped / 掩码；本行右侧表达式负责得到这个值。
    cropped_mask = mask[y : y + height, x : x + width]
    # 代码说明：cropped_image 表示cropped / 图片；本行右侧表达式负责得到这个值。
    cropped_image = image[y : y + height, x : x + width]

    # 代码说明：square_mask 表示square / 掩码；本行右侧表达式负责得到这个值。
    square_mask = _to_square(cv2, cropped_mask, border_value=0)
    # 代码说明：shape 表示shape；本行右侧表达式负责得到这个值。
    shape = cv2.resize(square_mask, (64, 64), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    # 代码说明：shape_vector 表示shape / 向量；本行右侧表达式负责得到这个值。
    shape_vector = shape.reshape(-1)

    # 代码说明：aspect_ratio 表示aspect / ratio；本行右侧表达式负责得到这个值。
    aspect_ratio = width / max(height, 1)
    # 代码说明：fill_ratio 表示fill / ratio；这里调用 `float` 得到它，调用目的：把环境变量字符串转换成数字，方便后续比较阈值。
    fill_ratio = float(cropped_mask.sum() / 255) / max(width * height, 1)
    # 代码说明：hue_hist 表示hue / hist；本行右侧表达式负责得到这个值。
    hue_hist = _hue_histogram(cv2, np, cropped_image, cropped_mask)
    # 代码说明：hu 表示hu；本行右侧表达式负责得到这个值。
    hu = _hu_moments(cv2, np, cropped_mask)

    # 代码说明：返回字典响应：下面每个键都会成为接口返回 JSON 或内部结构字段。
    return {
        # 代码说明：设置字典键 `shape` 的值，用来表达接口字段、配置项或业务映射关系。
        "shape": shape_vector,
        # 代码说明：设置字典键 `hue_hist` 的值，用来表达接口字段、配置项或业务映射关系。
        "hue_hist": hue_hist,
        # 代码说明：设置字典键 `hu` 的值，用来表达接口字段、配置项或业务映射关系。
        "hu": hu,
        # 代码说明：设置字典键 `aspect_ratio` 的值，用来表达接口字段、配置项或业务映射关系。
        "aspect_ratio": aspect_ratio,
        # 代码说明：设置字典键 `fill_ratio` 的值，用来表达接口字段、配置项或业务映射关系。
        "fill_ratio": fill_ratio,
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    }


# 函数说明：_icon_mask：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _icon_mask：封装当前模块的一段业务逻辑，供其他流程复用。
def _icon_mask(cv2, image):
    # 代码说明：导入模块，numpy：提供本模块需要的标准库/第三方能力。
    import numpy as np

    # 代码说明：hsv 表示hsv；本行右侧表达式负责得到这个值。
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 代码说明：hue, saturation, value 表示一组同时解包出来的值：hue、saturation、value；本行右侧表达式负责得到这个值。
    hue, saturation, value = cv2.split(hsv)
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    warm_mask = (((hue <= 50) | (hue >= 155)) & (saturation >= 35) & (value >= 45)).astype(np.uint8) * 255

    # 代码说明：gray 表示gray；本行右侧表达式负责得到这个值。
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    non_white = ((gray < 245) & (saturation >= 20)).astype(np.uint8) * 255
    # 代码说明：mask 表示掩码；本行右侧表达式负责得到这个值。
    mask = cv2.bitwise_or(warm_mask, non_white)
    # 代码说明：mask 表示掩码；本行右侧表达式负责得到这个值。
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # 代码说明：mask 表示掩码；本行右侧表达式负责得到这个值。
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return mask


# 函数说明：_to_square：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _to_square：封装当前模块的一段业务逻辑，供其他流程复用。
def _to_square(cv2, image, border_value: int):
    # 代码说明：height, width 表示一组同时解包出来的值：height、width；本行右侧表达式负责得到这个值。
    height, width = image.shape[:2]
    # 代码说明：side 表示side；本行右侧表达式负责得到这个值。
    side = max(height, width)
    # 代码说明：top：排序第一的视觉候选，它会被整理成 visual_match；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    top = (side - height) // 2
    # 代码说明：bottom 表示下边界；本行右侧表达式负责得到这个值。
    bottom = side - height - top
    # 代码说明：left 表示左边界；本行右侧表达式负责得到这个值。
    left = (side - width) // 2
    # 代码说明：right 表示右边界；本行右侧表达式负责得到这个值。
    right = side - width - left
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)


# 函数说明：_hue_histogram：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _hue_histogram：封装当前模块的一段业务逻辑，供其他流程复用。
def _hue_histogram(cv2, np, image, mask):
    # 代码说明：hsv 表示hsv；本行右侧表达式负责得到这个值。
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 代码说明：hist 表示hist；本行右侧表达式负责得到这个值。
    hist = cv2.calcHist([hsv], [0], mask, [18], [0, 180]).astype("float32").reshape(-1)
    # 代码说明：total 表示total；这里调用 `float` 得到它，调用目的：把环境变量字符串转换成数字，方便后续比较阈值。
    total = float(hist.sum())
    # 代码说明：判断条件 `total > 0` 是否成立，成立时执行下面缩进代码。
    if total > 0:
        # 代码说明：hist 表示hist；本行右侧表达式负责得到这个值。
        hist = hist / total
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return hist


# 函数说明：_hu_moments：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _hu_moments：封装当前模块的一段业务逻辑，供其他流程复用。
def _hu_moments(cv2, np, mask):
    # 代码说明：moments 表示moments；本行右侧表达式负责得到这个值。
    moments = cv2.moments(mask)
    # 代码说明：hu 表示hu；本行右侧表达式负责得到这个值。
    hu = cv2.HuMoments(moments).reshape(-1)
    # 代码说明：hu 表示hu；本行右侧表达式负责得到这个值。
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return hu.astype("float32")


# 函数说明：_feature_similarity：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _feature_similarity：封装当前模块的一段业务逻辑，供其他流程复用。
def _feature_similarity(query: dict, candidate: dict) -> float:
    # 代码说明：shape_score 表示shape / 分数；本行右侧表达式负责得到这个值。
    shape_score = _cosine(query["shape"], candidate["shape"])
    # 代码说明：color_score 表示color / 分数；本行右侧表达式负责得到这个值。
    color_score = _hist_intersection(query["hue_hist"], candidate["hue_hist"])
    # 代码说明：aspect_score 表示aspect / 分数；本行右侧表达式负责得到这个值。
    aspect_score = 1.0 - min(abs(query["aspect_ratio"] - candidate["aspect_ratio"]) / 2.5, 1.0)
    # 代码说明：fill_score 表示fill / 分数；本行右侧表达式负责得到这个值。
    fill_score = 1.0 - min(abs(query["fill_ratio"] - candidate["fill_ratio"]) / 0.7, 1.0)
    # 代码说明：hu_score 表示hu / 分数；这里调用 `float` 得到它，调用目的：把环境变量字符串转换成数字，方便后续比较阈值。
    hu_score = 1.0 - min(float(abs(query["hu"] - candidate["hu"]).mean()) / 8.0, 1.0)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return max(0.0, min(1.0, shape_score * 0.5 + color_score * 0.2 + aspect_score * 0.15 + fill_score * 0.1 + hu_score * 0.05))


# 函数说明：_cosine：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _cosine：封装当前模块的一段业务逻辑，供其他流程复用。
def _cosine(left, right) -> float:
    # 代码说明：导入模块，numpy：提供本模块需要的标准库/第三方能力。
    import numpy as np

    # 代码说明：denominator 表示denominator；这里调用 `float` 得到它，调用目的：把环境变量字符串转换成数字，方便后续比较阈值。
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    # 代码说明：判断条件 `denominator <= 1e-8` 是否成立，成立时执行下面缩进代码。
    if denominator <= 1e-8:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return 0.0
    # 代码说明：返回调用结果：把环境变量字符串转换成数字，方便后续比较阈值，调用方会拿这个结果继续下一步。
    return max(0.0, min(1.0, float(np.dot(left, right) / denominator)))


# 函数说明：_hist_intersection：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _hist_intersection：封装当前模块的一段业务逻辑，供其他流程复用。
def _hist_intersection(left, right) -> float:
    # 代码说明：导入模块，numpy：提供本模块需要的标准库/第三方能力。
    import numpy as np

    # 代码说明：返回调用结果：把环境变量字符串转换成数字，方便后续比较阈值，调用方会拿这个结果继续下一步。
    return max(0.0, min(1.0, float(np.minimum(left, right).sum())))
