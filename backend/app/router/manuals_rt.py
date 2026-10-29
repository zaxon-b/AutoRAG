"""手册生命周期的 HTTP 入口：查询建库状态、上传 PDF。"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from schemas.manual import ManualStatusResponse, ManualUploadResponse
from service.manual_ingestion_service import get_manual_status, ingest_manual
from service.manual_storage_service import DEFAULT_MANUAL_ID
from utils.database import get_db

router = APIRouter(prefix="/manuals", tags=["manuals"])


@router.get("/status", response_model=ManualStatusResponse)
async def manual_status(db: Session = Depends(get_db)):
    """返回默认手册的页数、chunk 数、图标组数和建库状态。"""

    return get_manual_status(db, DEFAULT_MANUAL_ID)


@router.post("/upload", response_model=ManualUploadResponse)
async def upload_manual(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """接收 PDF 并同步完成整本手册的建库。

    建库包含解析、embedding 和 VLM 调用，耗时较长；前端需要按
    response 中的统计刷新状态卡片。
    """

    try:
        result = ingest_manual(db=db, file_obj=file.file, filename=file.filename or "owner_manual.pdf")
        return result
    except ValueError as error:
        # 文件类型或 PDF 内容不合法，属于用户可修复的输入错误。
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        # PDF 解析、ES、数据库或模型服务失败。
        raise HTTPException(status_code=500, detail=f"Manual ingestion failed: {error}")
