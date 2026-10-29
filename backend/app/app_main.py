"""FastAPI 应用入口：注册路由、CORS 与 /storage 静态目录。"""

import os
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from router import health_rt
from router import manuals_rt
from router import qa_rt


def normalize_root_path(value: str | None) -> str:
    """把 ROOT_PATH 规范成 URL 路径前缀。

    该环境变量可能被误配成完整域名；FastAPI 的 root_path 只接受路径，
    因此这里剥掉 scheme/host，只保留以 / 开头的部分。
    """

    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        return parsed.path if parsed.path and parsed.path != "/" else ""
    return value if value.startswith("/") else ""


root_path = normalize_root_path(os.getenv("ROOT_PATH"))

app = FastAPI(root_path=root_path)

# 当前是本地演示配置；生产部署应把 allow_origins 收敛到具体前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 手册 PDF、页面截图、原生图片资产和用户上传图片都放在这里，
# 再通过 /storage 暴露给前端。
storage_dir = Path(os.getenv("STORAGE_ROOT", Path(__file__).resolve().parent / "storage"))
storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_dir)), name="storage")

app.include_router(health_rt.router)
app.include_router(manuals_rt.router)
app.include_router(qa_rt.router)

if __name__=='__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
