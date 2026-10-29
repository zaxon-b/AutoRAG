"""文本问答与图片问答的请求/响应结构。"""

from typing import Literal

from pydantic import BaseModel


ChunkModality = Literal["text", "icon", "operation_figure", "page_snapshot"]


class TextQaRequest(BaseModel):
    manual_id: str = "demo_manual"
    question: str


class EvidenceItem(BaseModel):
    """前端证据面板与 LLM prompt 共用的统一证据结构。"""

    chunk_id: str
    page: int
    modality: ChunkModality
    section_title: str | None = None
    content_text: str | None = None
    # 视觉对象名称；正文证据固定为空。
    object_name: str | None = None
    # text 父子树字段；视觉证据固定为空。
    parent_chunk_id: str | None = None
    is_leaf: bool | None = None
    token_count: int | None = None
    risk_level: str | None = None
    image_path: str | None = None
    thumbnail_path: str | None = None
    icon_group_id: str | None = None
    score: float | None = None
    match_method: str | None = None
    # visual 只证明对象身份；text 才用于解释含义和使用方法。
    evidence_role: str | None = None


class VisualMatch(BaseModel):
    """图片问答中视觉锚点的摘要信息。"""

    matched: bool
    object_name: str | None = None
    section_title: str | None = None
    score: float | None = None
    page: int | None = None
    modality: ChunkModality | None = None
    risk_level: str | None = None
    image_path: str | None = None
    thumbnail_path: str | None = None
    icon_group_id: str | None = None
    match_method: str | None = None
    vlm_verified: bool | None = None
    vlm_confidence: float | None = None
    vlm_reason: str | None = None


class ImageQaResponse(BaseModel):
    """图片问答完整响应：识别结果、回答、建议和证据。"""

    recognized_object: str
    image_type: str | None = None
    is_target_icon: bool | None = None
    classification_confidence: float | None = None
    classification_reason: str | None = None
    visual_description: str | None = None
    visual_keywords: list[str] | None = None
    candidate_names: list[str] | None = None
    normalized_query: str | None = None
    risk_level: str
    confidence: float | None = None
    vision_error: str | None = None
    visual_match: VisualMatch | None = None
    answer: str
    suggestion: str
    evidence: list[EvidenceItem]
    # evidence 仍然是视觉在前、正文在后；这两个字段是给前端分组的显式别名。
    visual_evidence: EvidenceItem | None = None
    text_evidence: list[EvidenceItem] | None = None
    evidence_status: str | None = None
