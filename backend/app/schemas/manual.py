"""手册上传与状态接口的响应结构。"""

from pydantic import BaseModel


class ManualUploadResponse(BaseModel):
    """建库完成后的规模统计。"""

    manual_id: str
    file_name: str
    page_count: int
    text_chunk_count: int
    # 只存在 PostgreSQL、用于 Auto-Merging 的父节点数量。
    text_parent_count: int
    visual_chunk_count: int
    icon_group_count: int = 0
    image_asset_count: int = 0
    image_occurrence_count: int = 0
    icon_prototype_count: int = 0
    status: str


class ManualStatusResponse(BaseModel):
    """页面刷新后恢复建库状态；空库时各计数为 0。"""

    manual_id: str | None = None
    file_name: str | None = None
    page_count: int = 0
    text_chunk_count: int = 0
    text_parent_count: int = 0
    visual_chunk_count: int = 0
    icon_group_count: int = 0
    image_asset_count: int = 0
    image_occurrence_count: int = 0
    icon_prototype_count: int = 0
    status: str = "empty"
