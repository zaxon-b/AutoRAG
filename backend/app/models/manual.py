"""车主手册 RAG 的 ORM 表模型。

正文使用 text 叶子/父节点两级结构；视觉侧按
asset（唯一图片）→ occurrence（每页出现位置）→ group（同一图标的身份）
组织，每个组对应一个 icon Chunk 和一条 ES 文档。
"""

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from models.base import Base


class Manual(Base):
    """一本车主手册的元信息与建库状态。"""

    __tablename__ = "manuals"

    id = Column(Integer, primary_key=True, index=True)
    # 业务主键；PostgreSQL 和 ES 都按它隔离不同手册的数据。
    manual_id = Column(String(64), unique=True, nullable=False, index=True)
    vehicle_model = Column(String(128), nullable=True)
    manual_version = Column(String(128), nullable=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    page_count = Column(Integer, nullable=False, default=0)
    # processing 表示建库中，ready 表示可以问答。
    status = Column(String(32), nullable=False, default="ready")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    pages = relationship("ManualPage", back_populates="manual", cascade="all, delete-orphan")
    chunks = relationship("ManualChunk", back_populates="manual", cascade="all, delete-orphan")
    icon_groups = relationship("ManualIconGroup", back_populates="manual", cascade="all, delete-orphan")


class ManualPage(Base):
    """PDF 每一页的渲染截图、尺寸和提取文本。"""

    __tablename__ = "manual_pages"

    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String(64), ForeignKey("manuals.manual_id", ondelete="CASCADE"), nullable=False, index=True)
    # PDF 页码，从 1 开始。
    page_number = Column(Integer, nullable=False, index=True)
    image_path = Column(String(512), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    ocr_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    manual = relationship("Manual", back_populates="pages")


class ManualChunk(Base):
    """统一的四类 chunk：text / icon / operation_figure / page_snapshot。"""

    __tablename__ = "manual_chunks"

    id = Column(Integer, primary_key=True, index=True)
    # 跨 PostgreSQL、ES、图标组和前端证据面板对齐同一条证据。
    chunk_id = Column(String(128), nullable=False, index=True)
    manual_id = Column(String(64), ForeignKey("manuals.manual_id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False, index=True)
    __table_args__ = (
        # parent_chunk_id 是自引用外键，PostgreSQL 在 CREATE TABLE 阶段就要求
        # 目标列有真正的 UNIQUE 约束；事后补建唯一索引来不及。
        UniqueConstraint("chunk_id", name="uq_manual_chunks_chunk_id"),
        CheckConstraint(
            "modality IN ('text', 'icon', 'operation_figure', 'page_snapshot')",
            name="ck_manual_chunks_modality",
        ),
        CheckConstraint(
            "parent_chunk_id IS NULL OR parent_chunk_id <> chunk_id",
            name="ck_manual_chunks_parent_not_self",
        ),
        # 每种模态只允许出现自己该有的字段，防止建库代码写入错误结构。
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

    modality = Column(String(64), nullable=False, default="text", index=True)
    # 完整 PDF 书签路径，例如“车辆使用 > 驾驶车辆 > 驱动模式”。
    section_title = Column(Text, nullable=True)
    # BM25、文本向量检索和 LLM 回答的主要输入。
    content_text = Column(Text, nullable=True)
    # 跨页正文的真实来源页集合；page_number 只记录起始页。
    source_pages = Column(JSONB, nullable=True)
    # 视觉对象名称；icon 必须来自手册原文，operation_figure 可用受约束的短名称。
    object_name = Column(String(255), nullable=True)
    # icon 直接持有所属组，不需要额外的成员关系表。
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
    risk_level = Column(String(32), nullable=True)
    # PDF 页面坐标，用于裁剪图标、表格或页面局部证据。
    bbox = Column(JSONB, nullable=True)
    image_path = Column(String(512), nullable=True)
    thumbnail_path = Column(String(512), nullable=True)
    # 该 chunk 在 ES 中的文档 ID；当前与 chunk_id 相同，保留字段用于迁移与排查。
    es_doc_id = Column(String(128), nullable=True)
    # 视觉候选的匹配分，仅评测和诊断使用。
    score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    manual = relationship("Manual", back_populates="chunks")


class QueryLog(Base):
    """可选的查询日志，便于回放评测和排查线上问题。"""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String(64), nullable=False, index=True)
    query_type = Column(String(32), nullable=False)
    # 图片问答中它是可选补充问题。
    question = Column(Text, nullable=True)
    image_path = Column(String(512), nullable=True)
    recognized_object = Column(String(255), nullable=True)
    evidence_json = Column(JSONB, nullable=True)
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ManualIconGroup(Base):
    """同一图标在手册中多次出现时合并成的视觉身份。"""

    __tablename__ = "manual_icon_groups"
    __table_args__ = (UniqueConstraint("icon_group_id", name="uq_manual_icon_groups_icon_group_id"),)

    id = Column(Integer, primary_key=True, index=True)
    icon_group_id = Column(String(128), nullable=False, index=True)
    manual_id = Column(String(64), ForeignKey("manuals.manual_id", ondelete="CASCADE"), nullable=False, index=True)
    # 严格来自手册原文；无法可靠命名时允许为空。
    canonical_icon_name = Column(String(255), nullable=True, index=True)
    # 该组唯一 icon Chunk 的 ID，前端优先展示它。
    representative_chunk_id = Column(String(128), nullable=True, index=True)
    # occurrence 数，不是组内 Chunk 数（后者恒为 1）。
    member_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    manual = relationship("Manual", back_populates="icon_groups")


class ManualImageAsset(Base):
    """按 xref/digest 精确去重后的 PDF 原生图片资产。"""

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
    # 同一 digest 可能对应多个 PDF xref；保留全部来源，而不是只取第一个。
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
    """同一图片在 PDF 中的一次显示位置。"""

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
    # 只是逻辑指针，故意不加外键，避免 occurrence、group 和 chunk 之间
    # 形成循环插入/删除依赖。
    visual_chunk_id = Column(String(128), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ManualIconPrototype(Base):
    """一个图标组选出的在线检索代表图。

    当前每组只保留 rank=1 的一张；该表继续保留，供维护脚本和离线
    OpenCV 诊断使用，在线索引不再依赖它。
    """

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
