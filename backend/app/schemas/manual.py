# 模块说明：手册接口 Schema：定义上传/状态接口返回给前端的数据结构。
# 调用关系：由 manuals_rt 的 response_model 使用。
# 返回/产出：让 FastAPI 自动校验并序列化手册响应字段。
# 代码说明：从 pydantic 导入：BaseModel：本文件后续逻辑依赖的对象。
from pydantic import BaseModel


# 类说明：ManualUploadResponse：手册上传接口返回结构，告诉前端建库后的页数、chunk 数和状态。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 ManualUploadResponse：手册上传接口返回结构，告诉前端建库后的页数、chunk 数和状态。
class ManualUploadResponse(BaseModel):
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    manual_id: str
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    file_name: str
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    page_count: int
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    text_chunk_count: int
    # PostgreSQL 中仅通过 Auto-Merging 返回的正文父节点数量。
    text_parent_count: int
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    visual_chunk_count: int
    # 代码说明：icon_group_count：icon_group_count 保存自动整理出的标准图标组数量。
    icon_group_count: int = 0
    image_asset_count: int = 0
    image_occurrence_count: int = 0
    icon_prototype_count: int = 0
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    status: str


# 类说明：ManualStatusResponse：手册状态接口返回结构，和上传返回基本一致，用于页面刷新后恢复状态。
# 被谁调用：SQLAlchemy、Pydantic、FastAPI response_model 或业务 service 会创建/读取这个类。
# 返回内容：类定义本身不返回值；实例对象承载数据库记录或接口数据。
# 代码说明：定义类 ManualStatusResponse：手册状态接口返回结构，和上传返回基本一致，用于页面刷新后恢复状态。
class ManualStatusResponse(BaseModel):
    # 代码说明：manual_id：当前查询或上传对应的手册 ID，用来隔离不同车型/手册的数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    manual_id: str | None = None
    # 代码说明：file_name：file_name 保存用户上传 PDF 的原始文件名。
    file_name: str | None = None
    # 代码说明：page_count：page_count 保存手册 PDF 总页数。
    page_count: int = 0
    # 代码说明：text_chunk_count：text_chunk_count 保存建库后生成的文本 chunk 数量。
    text_chunk_count: int = 0
    text_parent_count: int = 0
    # 代码说明：visual_chunk_count：visual_chunk_count 保存建库后生成的图标/截图等视觉 chunk 数量。
    visual_chunk_count: int = 0
    # 代码说明：icon_group_count：icon_group_count 保存自动整理出的标准图标组数量。
    icon_group_count: int = 0
    image_asset_count: int = 0
    image_occurrence_count: int = 0
    icon_prototype_count: int = 0
    # 代码说明：status：status 保存当前手册处理状态，例如 ready 表示可以问答。
    status: str = "empty"
