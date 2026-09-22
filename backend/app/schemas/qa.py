# 模块说明：问答接口 Schema：定义文本问答请求、证据项、视觉匹配结果和图片问答响应结构。
# 调用关系：由 qa_rt 的请求参数和 response_model 使用。
# 返回/产出：让 FastAPI/Pydantic 检查输入输出字段是否符合约定。
# 代码说明：从 pydantic 导入：BaseModel：本文件后续逻辑依赖的对象。
from typing import Literal

from pydantic import BaseModel


ChunkModality = Literal["text", "icon", "operation_figure", "page_snapshot"]


# 类说明：TextQaRequest：文本问答请求体，包含 manual_id 和用户问题 question。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 TextQaRequest：文本问答请求体，包含 manual_id 和用户问题 question。
class TextQaRequest(BaseModel):
    # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual_id: str = "demo_manual"
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    question: str


# 类说明：EvidenceItem：统一证据结构，前端证据面板和 LLM prompt 都依赖这些字段。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 EvidenceItem：统一证据结构，前端证据面板和 LLM prompt 都依赖这些字段。
class EvidenceItem(BaseModel):
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    chunk_id: str
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    page: int
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    modality: ChunkModality
    # 代码说明：section_title：chunk 的章节标题或图标名称，是检索、展示和 prompt 引用的重要字段；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    section_title: str | None = None
    # 代码说明：content_text：content_text 保存 chunk 的主文本内容，LLM 回答和 BM25/文本向量检索主要依赖它。
    content_text: str | None = None
    # 视觉对象名称；正文证据固定为空。
    object_name: str | None = None
    # text 父子树字段；视觉证据固定为空。
    parent_chunk_id: str | None = None
    is_leaf: bool | None = None
    token_count: int | None = None
    # 代码说明：risk_level：前端展示的风险等级；强匹配优先使用手册证据，否则使用 VLM 判断；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    risk_level: str | None = None
    # 代码说明：image_path：用户查询图片在本地存储中的路径，是视觉理解和图像检索的输入；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    image_path: str | None = None
    # 代码说明：thumbnail_path：缩略图路径，用于前端证据面板快速显示；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    thumbnail_path: str | None = None
    # 代码说明：icon_group_id：icon_group_id 标记这个图标属于哪个标准图标组，用来把同一图标的多处出现合并。
    icon_group_id: str | None = None
    # 代码说明：score：候选图标的视觉匹配分数；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    score: float | None = None
    # 代码说明：match_method：match_method 记录证据来自哪条召回路线，例如 image_embedding、opencv_icon_match、vlm_rerank。
    match_method: str | None = None
    # visual identifies the uploaded object; text explains meaning/use/handling.
    evidence_role: str | None = None


# 类说明：VisualMatch：图片问答中的视觉匹配摘要，描述是否命中手册图标、分数、页码和 VLM 验证结果。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 VisualMatch：图片问答中的视觉匹配摘要，描述是否命中手册图标、分数、页码和 VLM 验证结果。
class VisualMatch(BaseModel):
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    matched: bool
    object_name: str | None = None
    section_title: str | None = None
    # 代码说明：score：候选图标的视觉匹配分数；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    score: float | None = None
    # 代码说明：page：候选证据所在手册页码；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    page: int | None = None
    modality: ChunkModality | None = None
    # 代码说明：risk_level：前端展示的风险等级；强匹配优先使用手册证据，否则使用 VLM 判断；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    risk_level: str | None = None
    # 代码说明：image_path：用户查询图片在本地存储中的路径，是视觉理解和图像检索的输入；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    image_path: str | None = None
    # 代码说明：thumbnail_path：缩略图路径，用于前端证据面板快速显示；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    thumbnail_path: str | None = None
    # 代码说明：icon_group_id：icon_group_id 标记这个图标属于哪个标准图标组，用来把同一图标的多处出现合并。
    icon_group_id: str | None = None
    # 代码说明：match_method：match_method 记录证据来自哪条召回路线，例如 image_embedding、opencv_icon_match、vlm_rerank。
    match_method: str | None = None
    # 代码说明：vlm_verified：vlm_verified 表示 VLM 是否确认用户图和候选图标匹配。
    vlm_verified: bool | None = None
    # 代码说明：vlm_confidence：vlm_confidence 保存 VLM 二次验证的置信度。
    vlm_confidence: float | None = None
    # 代码说明：vlm_reason：vlm_reason 保存 VLM 为什么认为候选匹配或不匹配的解释。
    vlm_reason: str | None = None


# 类说明：ImageQaResponse：图片问答完整响应，包含识别结果、回答、建议和 evidence。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 ImageQaResponse：图片问答完整响应，包含识别结果、回答、建议和 evidence。
class ImageQaResponse(BaseModel):
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    recognized_object: str
    image_type: str | None = None
    is_target_icon: bool | None = None
    classification_confidence: float | None = None
    classification_reason: str | None = None
    # 代码说明：visual_description：visual_description 保存 VLM 对用户图片外观的描述，方便前端展示和人工核对。
    visual_description: str | None = None
    # 代码说明：visual_keywords：VLM 从图片里提取的颜色、形状、符号等视觉关键词；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    visual_keywords: list[str] | None = None
    # 代码说明：candidate_names：VLM 给出的候选图标名称列表，用于扩展检索和展示模型不确定性；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    candidate_names: list[str] | None = None
    # 代码说明：normalized_query：normalized_query 保存系统真正用于检索的标准化问题。
    normalized_query: str | None = None
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    risk_level: str
    # 代码说明：confidence：前端展示的置信度；视觉匹配分会从百分制转换成 0-1，VLM 置信度则原样使用；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    confidence: float | None = None
    # 代码说明：vision_error：vision_error 保存 VLM 调用失败信息，让前端能显示具体原因。
    vision_error: str | None = None
    # 代码说明：visual_match：视觉候选第一名整理出的摘要，包含图标名、分数、页码、图片路径、验证状态；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    visual_match: VisualMatch | None = None
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    answer: str
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    suggestion: str
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    evidence: list[EvidenceItem]
    # Optional structured aliases. ``evidence`` remains visual first, then text.
    visual_evidence: EvidenceItem | None = None
    text_evidence: list[EvidenceItem] | None = None
    evidence_status: str | None = None
