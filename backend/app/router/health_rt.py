from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from service.es_index_service import INDEX_NAME, ping_es
from utils.database import get_db

# 健康检查接口单独放在 /health 下，方便开发环境、Docker 编排或前端启动前快速确认后端依赖是否可用。
router = APIRouter(prefix="/health", tags=["health"])


# 健康检查接口：回答“API、PostgreSQL、Elasticsearch 以及当前 ES 索引配置是否正常”。
# 输入来源：没有请求体；db 由 Depends(get_db) 注入，用于执行一条最轻量的数据库探测 SQL。
# 核心流程：先默认 PostgreSQL 为 ok -> 执行 SELECT 1 探活 -> 失败时把 postgres 标为 error ->
# 再调用 ping_es 探测 Elasticsearch -> 把当前使用的 INDEX_NAME 一并返回。
# 返回给前端/运维：api 表示 FastAPI 进程能响应；postgres/elasticsearch 是依赖服务状态；index 是当前配置的 ES 索引名。
# 错误/兜底：数据库探活失败不会抛出异常中断接口，而是降级返回 postgres=error；ES 探活由 ping_es 返回布尔值。
@router.get("")
async def health(db: Session = Depends(get_db)):
    # 阶段：探测 PostgreSQL。
    # postgres 初始为 ok，只有 SELECT 1 抛异常才改成 error；这让接口本身尽量可用，便于定位是哪一个依赖坏了。
    postgres = "ok"
    try:
        # SELECT 1 不依赖业务表，主要验证连接池、网络和数据库进程是否可用；它只读不写，没有业务副作用。
        db.execute(text("SELECT 1"))
    except Exception:
        # 捕获所有异常是为了健康检查不被数据库错误直接打断，而是把失败状态放进 JSON 返回。
        postgres = "error"

    # 阶段：组装健康检查结果。
    # 这里不访问手册业务表，只返回基础服务状态；前端可据此提示“后端启动了但依赖未就绪”。
    return {
        "api": "ok",
        "postgres": postgres,
        # ping_es 会访问 Elasticsearch；失败时用 error 表示索引检索能力不可用。
        "elasticsearch": "ok" if ping_es() else "error",
        # index 是当前后端会读写的 ES 索引名，用于排查环境变量或索引初始化问题。
        "index": INDEX_NAME,
    }
