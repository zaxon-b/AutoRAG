from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from schemas.manual import ManualStatusResponse, ManualUploadResponse
from service.manual_ingestion_service import get_manual_status, ingest_manual
from service.manual_storage_service import DEFAULT_MANUAL_ID
from utils.database import get_db

# 创建路由，当前router的前缀是 /manuals，所有接口都会以 /manuals 开头
# 这个模块只负责“手册生命周期”的 HTTP 入口：前端查状态、上传 PDF，都从这里进入 service 层。
router = APIRouter(prefix="/manuals", tags=["manuals"])


# 前端请求：GET http://localhost:8000/manuals/status
# 此路由带前缀 /manuals/status， FastAPI 查表，匹配到下面的前端请求直接匹配该函数
#
# 手册状态接口：给前端展示当前默认手册是否已经建库、建了多少文本/视觉证据。
# 输入来源：db 由 Depends(get_db) 注入，是本次请求的数据库会话；manual_id 固定使用 DEFAULT_MANUAL_ID。
# 核心流程：不接收请求体，只读取默认手册的建库统计信息，例如页数、文本 chunk 数、视觉 chunk 数、图标组数量和状态。
# 返回给前端：ManualStatusResponse，前端用它决定是否提示“请先上传手册”、是否可以开始文本/图片问答。
# 错误/兜底：这里不主动捕获异常；如果数据库不可用或状态查询失败，会交给 FastAPI 返回服务端错误。
@router.get("/status", response_model=ManualStatusResponse)
async def manual_status(db: Session = Depends(get_db)):
    # 阶段：读取默认手册建库状态。
    # get_manual_status 会访问数据库中的手册、chunk 或图标组统计；它没有写入副作用，只负责把当前索引状态汇总出来。
    return get_manual_status(db, DEFAULT_MANUAL_ID)


# 上传手册接口：接收前端上传的 PDF，并触发完整的多模态建库流程。
# file：前端注册了一个form表单，然后添加一个 file 字段，FastAPI从表单里获取file 字段，传给后端的这个函数
# 输入来源：file 是 multipart/form-data 里的上传文件；db 是请求级数据库会话。
# 核心流程：把 UploadFile.file 交给 ingest_manual -> service 层保存文件、解析页面、切文本/表格/视觉 chunk、生成 embedding、写 PostgreSQL 和 ES。
# 返回给前端：ManualUploadResponse，通常包含上传后的手册 ID、文件名、建库状态和统计信息，前端据此刷新状态卡片。
# 错误/兜底：ValueError 通常代表用户输入或文件内容不合法，转成 400；其他异常可能来自 PDF 解析、文件系统、数据库、ES 或模型调用，转成 500。
@router.post("/upload", response_model=ManualUploadResponse)
async def upload_manual(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 阶段：触发手册建库，并把底层异常转换成前端可理解的 HTTP 错误。
    try:
        # ingest_manual 是本接口最重要的副作用入口：它会消费上传文件流，并可能写本地文件、写数据库、写 ES 索引、生成文本/图片 embedding。
        # filename 为空时使用 owner_manual.pdf 兜底，避免存储层拿到空文件名后无法确定扩展名或展示名。
        result = ingest_manual(db=db, file_obj=file.file, filename=file.filename or "owner_manual.pdf")
        # result 直接返回给前端，表示本次建库完成后的状态汇总；这里不改写字段，避免和 response_model 不一致。
        return result
    except ValueError as error:
        # 业务可预期错误，例如文件格式不支持或 PDF 无法解析，返回 400 让前端提示用户更换文件。
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        # 非预期错误可能来自解析器、文件系统、数据库、ES 或模型服务；返回 500，detail 保留失败原因便于本地调试。
        raise HTTPException(status_code=500, detail=f"Manual ingestion failed: {error}")
