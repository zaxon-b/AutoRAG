# 模块说明：SQLAlchemy ORM 模型：定义 manuals、manual_pages、manual_chunks、query_logs、manual_icon_groups 等数据库表。
# 调用关系：由数据库初始化、建库服务、检索服务和图标目录服务调用。
# 返回/产出：提供 Python 类和数据库表之间的字段映射。
# 代码说明：从 sqlalchemy 导入：Column：本文件后续逻辑依赖的对象；DateTime：本文件后续逻辑依赖的对象；Float：本文件后续逻辑依赖的对象；ForeignKey：本文件后续逻辑依赖的对象；Integer：本文件后续逻辑依赖的对象；String：本文件后续逻辑依赖的对象；Text：本文件后续逻辑依赖的对象。
from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
# 代码说明：从 sqlalchemy.dialects.postgresql 导入：JSONB：本文件后续逻辑依赖的对象。
from sqlalchemy.dialects.postgresql import JSONB
# 代码说明：从 sqlalchemy.orm 导入：relationship：本文件后续逻辑依赖的对象。
from sqlalchemy.orm import relationship
# 代码说明：从 sqlalchemy.sql 导入：func：本文件后续逻辑依赖的对象。
from sqlalchemy.sql import func

# 代码说明：从 models.base 导入：Base：本文件后续逻辑依赖的对象。
from models.base import Base


# 类说明：Manual：manuals 表模型，保存一本车主手册的元信息，例如 manual_id、文件路径、页数和状态。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 Manual：manuals 表模型，保存一本车主手册的元信息，例如 manual_id、文件路径、页数和状态。
class Manual(Base):
    # 代码说明：__tablename__ 表示SQLAlchemy 表名，决定这个 ORM 类对应数据库里的哪张表；本行右侧表达式负责得到这个值。
    __tablename__ = "manuals"

    # 代码说明：id 表示id；本行右侧表达式负责得到这个值。
    id = Column(Integer, primary_key=True, index=True)
    # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual_id = Column(String(64), unique=True, nullable=False, index=True)
    # 代码说明：vehicle_model 表示vehicle / 模型名；本行右侧表达式负责得到这个值。
    vehicle_model = Column(String(128), nullable=True)
    # 代码说明：manual_version 表示手册 / version；本行右侧表达式负责得到这个值。
    manual_version = Column(String(128), nullable=True)
    # 代码说明：file_name：file_name 保存用户上传 PDF 的原始文件名。
    file_name = Column(String(255), nullable=False)
    # 代码说明：file_path 表示文件 / 路径；本行右侧表达式负责得到这个值。
    file_path = Column(String(512), nullable=False)
    # 代码说明：page_count：page_count 保存手册 PDF 总页数。
    page_count = Column(Integer, nullable=False, default=0)
    # 代码说明：status：status 保存当前手册处理状态，例如 ready 表示可以问答。
    status = Column(String(32), nullable=False, default="ready")
    # 代码说明：created_at 表示created / at；本行右侧表达式负责得到这个值。
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # 代码说明：updated_at 表示updated / at；本行右侧表达式负责得到这个值。
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 代码说明：pages：PDF 渲染或解析出的页面列表；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    pages = relationship("ManualPage", back_populates="manual", cascade="all, delete-orphan")
    # 代码说明：chunks：建库过程中生成的多模态 chunk 列表；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    chunks = relationship("ManualChunk", back_populates="manual", cascade="all, delete-orphan")
    # 代码说明：icon_groups 表示图标 / 图标组集合；本行右侧表达式负责得到这个值。
    icon_groups = relationship("ManualIconGroup", back_populates="manual", cascade="all, delete-orphan")


# 类说明：ManualPage：manual_pages 表模型，保存 PDF 每一页的截图路径、尺寸和页面文本。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 ManualPage：manual_pages 表模型，保存 PDF 每一页的截图路径、尺寸和页面文本。
class ManualPage(Base):
    # 代码说明：__tablename__ 表示SQLAlchemy 表名，决定这个 ORM 类对应数据库里的哪张表；本行右侧表达式负责得到这个值。
    __tablename__ = "manual_pages"

    # 代码说明：id 表示id；本行右侧表达式负责得到这个值。
    id = Column(Integer, primary_key=True, index=True)
    # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual_id = Column(String(64), ForeignKey("manuals.manual_id", ondelete="CASCADE"), nullable=False, index=True)
    # 代码说明：page_number：PDF 页码，通常从 1 开始展示给用户；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    page_number = Column(Integer, nullable=False, index=True)
    # 代码说明：image_path：用户查询图片在本地存储中的路径，是视觉理解和图像检索的输入；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    image_path = Column(String(512), nullable=True)
    # 代码说明：width：width 保存图片或页面宽度，用于 bbox 缩放和裁剪。
    width = Column(Integer, nullable=True)
    # 代码说明：height：height 保存图片或页面高度，用于 bbox 缩放和裁剪。
    height = Column(Integer, nullable=True)
    # 代码说明：ocr_text：ocr_text 保存页面或图标邻近区域的文字，用来补充图片证据的可检索文本。
    ocr_text = Column(Text, nullable=True)
    # 代码说明：created_at 表示created / at；本行右侧表达式负责得到这个值。
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 代码说明：manual：数据库中的手册记录，代表一本文档的元数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual = relationship("Manual", back_populates="pages")


# 类说明：ManualChunk：manual_chunks 表模型，只保存 text/icon/operation_figure/page_snapshot。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 ManualChunk：manual_chunks 表模型，保存四种统一 chunk。
class ManualChunk(Base):
    # 代码说明：__tablename__ 表示SQLAlchemy 表名，决定这个 ORM 类对应数据库里的哪张表；本行右侧表达式负责得到这个值。
    __tablename__ = "manual_chunks"

    # 代码说明：id 表示id；本行右侧表达式负责得到这个值。
    id = Column(Integer, primary_key=True, index=True)
    # 代码说明：chunk_id：chunk 的唯一 ID，用来跨 PostgreSQL、ES、图标组和前端证据面板对齐同一条证据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    chunk_id = Column(String(128), nullable=False, index=True)
    # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual_id = Column(String(64), ForeignKey("manuals.manual_id", ondelete="CASCADE"), nullable=False, index=True)
    # 代码说明：page_number：PDF 页码，通常从 1 开始展示给用户；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    page_number = Column(Integer, nullable=False, index=True)
    __table_args__ = (
        # parent_chunk_id is self-referencing. PostgreSQL therefore needs an
        # actual UNIQUE constraint while CREATE TABLE is running; a unique
        # index emitted afterwards is too late for this foreign key.
        UniqueConstraint("chunk_id", name="uq_manual_chunks_chunk_id"),
        CheckConstraint(
            "modality IN ('text', 'icon', 'operation_figure', 'page_snapshot')",
            name="ck_manual_chunks_modality",
        ),
        CheckConstraint(
            "parent_chunk_id IS NULL OR parent_chunk_id <> chunk_id",
            name="ck_manual_chunks_parent_not_self",
        ),
        CheckConstraint(
            "(modality = 'text' AND is_leaf IS NOT NULL AND token_count IS NOT NULL "
            "AND object_name IS NULL AND icon_group_id IS NULL) OR "
            "(modality = 'icon' AND is_leaf IS NULL AND token_count IS NULL "
            "AND parent_chunk_id IS NULL) OR "
            "(modality IN ('operation_figure', 'page_snapshot') AND is_leaf IS NULL "
            "AND token_count IS NULL AND parent_chunk_id IS NULL AND icon_group_id IS NULL)",
            name="ck_manual_chunks_fields_by_modality",
        ),
        Index("ix_manual_chunks_manual_modality_leaf", "manual_id", "modality", "is_leaf"),
        Index("ix_manual_chunks_manual_page_modality", "manual_id", "page_number", "modality"),
    )

    # 代码说明：modality 表示模态类型；本行右侧表达式负责得到这个值。
    modality = Column(String(64), nullable=False, default="text", index=True)
    # 完整 PDF 书签路径，例如“车辆使用 > 驾驶车辆 > 驱动模式”。
    section_title = Column(Text, nullable=True)
    # 代码说明：content_text：content_text 保存 chunk 的主文本内容，LLM 回答和 BM25/文本向量检索主要依赖它。
    content_text = Column(Text, nullable=True)
    # 视觉对象名称；icon 必须来自手册原文，operation_figure 可使用受约束的短名称。
    object_name = Column(String(255), nullable=True)
    # icon 直接持有所属组，不再使用 manual_icon_group_members 重复保存成员关系。
    icon_group_id = Column(
        String(128),
        ForeignKey("manual_icon_groups.icon_group_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # text 节点组成书签父子树；只有叶子进入 ES 并被直接检索。
    parent_chunk_id = Column(
        String(128),
        ForeignKey(
            "manual_chunks.chunk_id",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=True,
        index=True,
    )
    is_leaf = Column(Boolean, nullable=True)
    token_count = Column(Integer, nullable=True)
    # 代码说明：risk_level：前端展示的风险等级；强匹配优先使用手册证据，否则使用 VLM 判断；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    risk_level = Column(String(32), nullable=True)
    # 代码说明：bbox：页面区域坐标，用于裁剪图标、表格或页面局部证据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    bbox = Column(JSONB, nullable=True)
    # 代码说明：image_path：用户查询图片在本地存储中的路径，是视觉理解和图像检索的输入；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    image_path = Column(String(512), nullable=True)
    # 代码说明：thumbnail_path：缩略图路径，用于前端证据面板快速显示；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    thumbnail_path = Column(String(512), nullable=True)
    # 代码说明：es_doc_id 表示es / doc / id；本行右侧表达式负责得到这个值。
    es_doc_id = Column(String(128), nullable=True)
    # 代码说明：score：候选图标的视觉匹配分数；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    score = Column(Float, nullable=True)
    # 代码说明：created_at 表示created / at；本行右侧表达式负责得到这个值。
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 代码说明：manual：数据库中的手册记录，代表一本文档的元数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual = relationship("Manual", back_populates="chunks")


# 类说明：QueryLog：query_logs 表模型，可选记录用户查询、识别结果、证据和回答，方便后续评测分析。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 QueryLog：query_logs 表模型，可选记录用户查询、识别结果、证据和回答，方便后续评测分析。
class QueryLog(Base):
    # 代码说明：__tablename__ 表示SQLAlchemy 表名，决定这个 ORM 类对应数据库里的哪张表；本行右侧表达式负责得到这个值。
    __tablename__ = "query_logs"

    # 代码说明：id 表示id；本行右侧表达式负责得到这个值。
    id = Column(Integer, primary_key=True, index=True)
    # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual_id = Column(String(64), nullable=False, index=True)
    # 代码说明：query_type 表示查询 / type；本行右侧表达式负责得到这个值。
    query_type = Column(String(32), nullable=False)
    # 代码说明：question：用户输入的问题文本；图片问答中它是可选补充问题；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    question = Column(Text, nullable=True)
    # 代码说明：image_path：用户查询图片在本地存储中的路径，是视觉理解和图像检索的输入；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    image_path = Column(String(512), nullable=True)
    # 代码说明：recognized_object：前端展示的识别对象；强匹配时使用手册图标名，否则使用 VLM 初判；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    recognized_object = Column(String(255), nullable=True)
    # 代码说明：evidence_json 表示证据 / json；本行右侧表达式负责得到这个值。
    evidence_json = Column(JSONB, nullable=True)
    # 代码说明：answer：最终返回给前端的回答文本，可能来自 LLM，也可能是低置信拒答模板；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    answer = Column(Text, nullable=True)
    # 代码说明：created_at 表示created / at；本行右侧表达式负责得到这个值。
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# 类说明：ManualIconGroup：manual_icon_groups 表模型，保存同一图标在手册中多处出现时的标准名称和代表图。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 ManualIconGroup：manual_icon_groups 表模型，保存同一图标在手册中多处出现时的标准名称和代表图。
class ManualIconGroup(Base):
    # 代码说明：__tablename__ 表示SQLAlchemy 表名，决定这个 ORM 类对应数据库里的哪张表；本行右侧表达式负责得到这个值。
    __tablename__ = "manual_icon_groups"
    __table_args__ = (UniqueConstraint("icon_group_id", name="uq_manual_icon_groups_icon_group_id"),)

    # 代码说明：id 表示id；本行右侧表达式负责得到这个值。
    id = Column(Integer, primary_key=True, index=True)
    # 代码说明：icon_group_id：icon_group_id 标记这个图标属于哪个标准图标组，用来把同一图标的多处出现合并。
    icon_group_id = Column(String(128), nullable=False, index=True)
    # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual_id = Column(String(64), ForeignKey("manuals.manual_id", ondelete="CASCADE"), nullable=False, index=True)
    # 严格来自手册的标准名称；无法可靠命名时允许为空。
    canonical_icon_name = Column(String(255), nullable=True, index=True)
    # 代码说明：representative_chunk_id：representative_chunk_id 指向图标组的代表 chunk，前端优先展示它。
    representative_chunk_id = Column(String(128), nullable=True, index=True)
    # 代码说明：member_count 表示member / 数量；本行右侧表达式负责得到这个值。
    member_count = Column(Integer, nullable=False, default=0)
    # 代码说明：created_at 表示created / at；本行右侧表达式负责得到这个值。
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 代码说明：manual：数据库中的手册记录，代表一本文档的元数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual = relationship("Manual", back_populates="icon_groups")


class ManualImageAsset(Base):
    """One native PDF image asset after exact xref/digest deduplication."""

    __tablename__ = "manual_image_assets"
    __table_args__ = (
        UniqueConstraint("asset_id", name="uq_manual_image_assets_asset_id"),
        Index("ix_manual_image_assets_manual_classification", "manual_id", "classification"),
    )

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(128), nullable=False, index=True)
    manual_id = Column(
        String(64),
        ForeignKey("manuals.manual_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # A digest can occur under several PDF xrefs. Preserve every source value
    # instead of pretending that the first xref is the identity.
    source_xrefs = Column(JSONB, nullable=False)
    source_smask_xrefs = Column(JSONB, nullable=False)
    digest = Column(String(128), nullable=True, index=True)
    pixel_width = Column(Integer, nullable=True)
    pixel_height = Column(Integer, nullable=True)
    native_image_path = Column(String(512), nullable=True)
    classification = Column(String(32), nullable=False, default="ignore", index=True)
    classification_confidence = Column(Float, nullable=True)
    classification_reason = Column(Text, nullable=True)
    noise_reason = Column(String(255), nullable=True)
    phash = Column(String(32), nullable=True)
    dominant_colour = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ManualImageOccurrence(Base):
    """One displayed occurrence of a native image inside the PDF."""

    __tablename__ = "manual_image_occurrences"
    __table_args__ = (
        UniqueConstraint("occurrence_id", name="uq_manual_image_occurrences_occurrence_id"),
        Index("ix_manual_image_occurrences_manual_page", "manual_id", "page_number"),
        Index("ix_manual_image_occurrences_manual_group", "manual_id", "icon_group_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    occurrence_id = Column(String(128), nullable=False, index=True)
    manual_id = Column(
        String(64),
        ForeignKey("manuals.manual_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    asset_id = Column(
        String(128),
        ForeignKey("manual_image_assets.asset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number = Column(Integer, nullable=False, index=True)
    bbox = Column(JSONB, nullable=False)
    transform = Column(JSONB, nullable=True)
    orientation = Column(String(32), nullable=False, default="r0")
    section_title = Column(Text, nullable=True)
    nearby_text = Column(JSONB, nullable=False)
    name_candidates = Column(JSONB, nullable=False)
    content_text = Column(Text, nullable=True)
    icon_group_id = Column(
        String(128),
        ForeignKey("manual_icon_groups.icon_group_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Logical pointer only. Keeping it free of a foreign key avoids a circular
    # insert/delete dependency between an occurrence, its group and its chunk.
    visual_chunk_id = Column(String(128), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ManualIconPrototype(Base):
    """A visual-index prototype selected from one persisted icon group."""

    __tablename__ = "manual_icon_prototypes"
    __table_args__ = (
        UniqueConstraint("prototype_id", name="uq_manual_icon_prototypes_prototype_id"),
        Index("ix_manual_icon_prototypes_group_rank", "icon_group_id", "prototype_rank"),
    )

    id = Column(Integer, primary_key=True, index=True)
    prototype_id = Column(String(128), nullable=False, index=True)
    manual_id = Column(
        String(64),
        ForeignKey("manuals.manual_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    icon_group_id = Column(
        String(128),
        ForeignKey("manual_icon_groups.icon_group_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    asset_id = Column(
        String(128),
        ForeignKey("manual_image_assets.asset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    group_chunk_id = Column(String(128), nullable=False, index=True)
    prototype_rank = Column(Integer, nullable=False)
    image_path = Column(String(512), nullable=False)
    phash = Column(String(32), nullable=True)
    dominant_colour = Column(JSONB, nullable=True)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
