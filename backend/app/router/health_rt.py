"""健康检查：确认 API、PostgreSQL 和 Elasticsearch 是否可用。"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from service.es_index_service import INDEX_NAME, ping_es
from utils.database import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health(db: Session = Depends(get_db)):
    """返回各依赖的状态和当前 ES 索引名。

    数据库探活失败只降级为 ``postgres=error``，不中断接口，
    便于运维区分是 API 进程还是底层依赖出了问题。
    """

    postgres = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        postgres = "error"

    return {
        "api": "ok",
        "postgres": postgres,
        "elasticsearch": "ok" if ping_es() else "error",
        "index": INDEX_NAME,
    }
