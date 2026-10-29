"""视觉召回：图片向量检索、多模态向量检索和 OpenCV 形状匹配。"""

from pathlib import Path

from sqlalchemy.orm import Session

from models.manual import ManualChunk, ManualIconPrototype
from service.es_index_service import search_image_chunks, search_multimodal_chunks
from service.icon_catalog_service import canonicalize_evidence
from service.image_embedding_service import (
    IMAGE_EMBEDDING_FIELD,
    IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
    IMAGE_EMBEDDING_PROFILE_NATIVE,
    IMAGE_EMBEDDING_V2_FIELD,
    embed_image,
)
from service.manual_storage_service import to_storage_url
from service.multimodal_embedding_service import embed_multimodal_image
from service.visual_candidate_service import VISUAL_MODALITIES

# 两路检索都只在三类视觉模态中召回，正文另行检索。
IMAGE_EMBEDDING_MODALITIES = sorted(VISUAL_MODALITIES)
OPENCV_MATCH_MODALITIES = sorted(VISUAL_MODALITIES)


def retrieve_visual_chunks_by_image_embedding(
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
    *,
    embedding_variant: str = "native",
) -> list[dict]:
    """用用户图片向量检索视觉 chunk。

    native 使用原生透明图向量字段；alpha_white 使用白底预处理后的影子
    向量字段。两者都是同一条 ES 文档，不会产生重复候选。
    """

    if embedding_variant == "native":
        preprocessing = IMAGE_EMBEDDING_PROFILE_NATIVE
        vector_field = IMAGE_EMBEDDING_FIELD
    elif embedding_variant == "alpha_white":
        preprocessing = IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE
        vector_field = IMAGE_EMBEDDING_V2_FIELD
    else:
        raise ValueError(f"Unsupported image embedding variant: {embedding_variant}")
    query_vector = embed_image(query_image_path, preprocessing=preprocessing)
    if not query_vector:
        return []

    search_kwargs = {
        "manual_id": manual_id,
        "image_vector": query_vector,
        "top_k": top_k,
        "modalities": IMAGE_EMBEDDING_MODALITIES,
    }
    if vector_field != IMAGE_EMBEDDING_FIELD:
        search_kwargs["vector_field"] = vector_field
    chunks = search_image_chunks(**search_kwargs)
    return [_format_es_icon_evidence(chunk) for chunk in chunks]


def retrieve_visual_chunks_by_image_embedding_v2(
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
) -> list[dict]:
    """兼容旧调用名：固定使用白底影子向量。"""

    return retrieve_visual_chunks_by_image_embedding(
        manual_id,
        query_image_path,
        top_k=top_k,
        embedding_variant="alpha_white",
    )


def retrieve_image_embedding_icons(manual_id: str, query_image_path: Path, top_k: int = 5) -> list[dict]:
    """兼容旧图标专用调用名；结果仍是全部视觉模态。"""

    return retrieve_visual_chunks_by_image_embedding(manual_id, query_image_path, top_k=top_k)


def retrieve_visual_chunks_by_multimodal(
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
) -> list[dict]:
    """用多模态向量检索，视觉模态过滤放在 ES 查询里完成。"""

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
    """兼容旧调用名。"""

    return retrieve_visual_chunks_by_multimodal(manual_id, query_image_path, top_k=top_k)


def retrieve_visual_chunks_by_opencv(
    db: Session,
    manual_id: str,
    query_image_path: Path,
    top_k: int = 20,
) -> list[dict]:
    """用 OpenCV 形状/颜色特征返回弱匹配候选。

    这条路线只作为离线诊断；它比较的是低层图像特征，不能单独决定
    图标的官方含义。
    """

    query_feature = _extract_icon_feature(query_image_path)
    if query_feature is None:
        return []

    rows = (
        db.query(ManualChunk)
        .filter(
            ManualChunk.manual_id == manual_id,
            ManualChunk.modality.in_(("operation_figure", "page_snapshot")),
        )
        .all()
    )
    scored: list[tuple[float, dict]] = []
    for row in rows:
        if not row.image_path:
            continue
        candidate_feature = _extract_icon_feature(Path(row.image_path))
        if candidate_feature is None:
            continue
        score = _feature_similarity(query_feature, candidate_feature)
        if score >= 0.35:
            scored.append((score, _row_evidence(row)))

    # 图标组只比对在线代表图，命中后仍以组 Chunk 返回。
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

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_format_opencv_evidence(item, score) for score, item in scored[:top_k]]


def retrieve_similar_icons(db: Session, manual_id: str, query_image_path: Path, top_k: int = 5) -> list[dict]:
    """兼容旧图标专用调用名。"""

    return retrieve_visual_chunks_by_opencv(db, manual_id, query_image_path, top_k=top_k)


def _format_es_icon_evidence(chunk: dict) -> dict:
    raw_score = float(chunk.get("_score") or chunk.get("score") or 0)
    # ES 的 cosine script_score 落在 1~2，先还原成 0~100 的展示分。
    score = max(0.0, min(100.0, (raw_score - 1.0) * 100.0 if raw_score <= 2.1 else raw_score))
    return canonicalize_evidence({
        "chunk_id": chunk.get("chunk_id") or chunk.get("es_doc_id") or "",
        "page": int(chunk.get("page") or chunk.get("page_number") or 0),
        # ES 文档一定包含 modality；缺失时按合法视觉类型兜底。
        "modality": chunk.get("modality") or "icon",
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
        "match_method": "image_embedding",
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


def _extract_icon_feature(image_path: Path) -> dict | None:
    """提取形状、颜色和长宽比等低层特征。

    先按 HSV 找前景色块，再裁剪、补成正方形并缩放到 64×64，作为形状向量。
    """

    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    mask = _icon_mask(cv2, image)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, width, height = cv2.boundingRect(coords)
    if width < 6 or height < 6:
        return None

    cropped_mask = mask[y : y + height, x : x + width]
    cropped_image = image[y : y + height, x : x + width]

    square_mask = _to_square(cv2, cropped_mask, border_value=0)
    shape = cv2.resize(square_mask, (64, 64), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    shape_vector = shape.reshape(-1)

    aspect_ratio = width / max(height, 1)
    fill_ratio = float(cropped_mask.sum() / 255) / max(width * height, 1)
    hue_hist = _hue_histogram(cv2, np, cropped_image, cropped_mask)
    hu = _hu_moments(cv2, np, cropped_mask)

    return {
        "shape": shape_vector,
        "hue_hist": hue_hist,
        "hu": hu,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
    }


def _icon_mask(cv2, image):
    """用暖色和“非白但有色”两个条件组合前景掩码。"""

    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    warm_mask = (((hue <= 50) | (hue >= 155)) & (saturation >= 35) & (value >= 45)).astype(np.uint8) * 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_white = ((gray < 245) & (saturation >= 20)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(warm_mask, non_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def _to_square(cv2, image, border_value: int):
    height, width = image.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)


def _hue_histogram(cv2, np, image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [18], [0, 180]).astype("float32").reshape(-1)
    total = float(hist.sum())
    if total > 0:
        hist = hist / total
    return hist


def _hu_moments(cv2, np, mask):
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).reshape(-1)
    # Hu 矩数量级跨度很大，取对数后更适合做距离比较。
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype("float32")


def _feature_similarity(query: dict, candidate: dict) -> float:
    shape_score = _cosine(query["shape"], candidate["shape"])
    color_score = _hist_intersection(query["hue_hist"], candidate["hue_hist"])
    aspect_score = 1.0 - min(abs(query["aspect_ratio"] - candidate["aspect_ratio"]) / 2.5, 1.0)
    fill_score = 1.0 - min(abs(query["fill_ratio"] - candidate["fill_ratio"]) / 0.7, 1.0)
    hu_score = 1.0 - min(float(abs(query["hu"] - candidate["hu"]).mean()) / 8.0, 1.0)
    return max(0.0, min(1.0, shape_score * 0.5 + color_score * 0.2 + aspect_score * 0.15 + fill_score * 0.1 + hu_score * 0.05))


def _cosine(left, right) -> float:
    import numpy as np

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / denominator)))


def _hist_intersection(left, right) -> float:
    import numpy as np

    return max(0.0, min(1.0, float(np.minimum(left, right).sum())))
