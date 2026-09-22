# 模块说明：FastAPI 应用入口：创建 app、配置跨域、暴露 storage 静态图片目录，并注册 health/manuals/qa 三组路由。
# 调用关系：由 uvicorn app_main:app 或 Docker 容器启动命令加载。
# 返回/产出：产出全局 app 对象，浏览器和前端请求最终都会进入这个对象。
# 代码说明：导入模块，os：读取环境变量，例如模型 key、阈值、数据库地址和存储目录。
import os
# 代码说明：从 pathlib 导入：Path：使用 pathlib 统一处理本地文件和目录路径。
from pathlib import Path
# 代码说明：从 urllib.parse 导入：urlparse：解析 ROOT_PATH，区分完整 URL 和路径前缀。
from urllib.parse import urlparse

# 代码说明：从 fastapi 导入：FastAPI：创建后端应用实例，所有路由和中间件都挂载到它上面。
from fastapi import FastAPI
# 代码说明：从 fastapi.middleware.cors 导入：CORSMiddleware：允许前端跨端口访问后端接口。
from fastapi.middleware.cors import CORSMiddleware
# 代码说明：从 fastapi.staticfiles 导入：StaticFiles：把本地 storage 目录暴露成 /storage 静态文件 URL。
from fastapi.staticfiles import StaticFiles
# 代码说明：从 router 导入：health_rt：本文件后续逻辑依赖的对象。
from router import health_rt
# 代码说明：从 router 导入：manuals_rt：本文件后续逻辑依赖的对象。
from router import manuals_rt
# 代码说明：从 router 导入：qa_rt：本文件后续逻辑依赖的对象。
from router import qa_rt


# 函数说明：normalize_root_path：把 ROOT_PATH 环境变量规范成 FastAPI 能接受的 URL 前缀，避免把完整域名误传给 root_path。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 normalize_root_path：把 ROOT_PATH 环境变量规范成 FastAPI 能接受的 URL 前缀，避免把完整域名误传给 root_path。
def normalize_root_path(value: str | None) -> str:
    # 代码说明：`"""FastAPI root_path is a URL path prefix, not a full origin."""` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    """FastAPI root_path is a URL path prefix, not a full origin."""
    # 代码说明：判断条件 `not value` 是否成立，成立时执行下面缩进代码。
    if not value:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return ""
    # 代码说明：parsed 表示parsed；这里调用 `urlparse` 得到它，调用目的：解析 ROOT_PATH，区分完整 URL 和路径前缀。
    parsed = urlparse(value)
    # 代码说明：判断条件 `parsed.scheme or parsed.netloc` 是否成立，成立时执行下面缩进代码。
    if parsed.scheme or parsed.netloc:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return parsed.path if parsed.path and parsed.path != "/" else ""
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return value if value.startswith("/") else ""


# 代码说明：root_path 表示FastAPI 部署前缀，反向代理部署时会影响接口路径；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
root_path = normalize_root_path(os.getenv("ROOT_PATH"))


# 创建FastAPI app，用于前后端交互
# 代码说明：app 表示app；这里调用 `FastAPI` 得到它，调用目的：创建后端应用实例，所有路由和中间件都挂载到它上面。
app = FastAPI(root_path=root_path)


# 避免前后端隔离，让前端能访问后端
# 添加 CORS 中间件
# 代码说明：开始多行调用 `app.add_middleware`，下面几行会逐项传入参数。
app.add_middleware(
    # 代码说明：导入 CORSMiddleware：允许前端跨端口访问后端接口。
    CORSMiddleware,
    # 代码说明：allow_origins 表示允许跨域访问后端的前端来源列表；本行右侧表达式负责得到这个值。
    allow_origins=["*"],  # 允许所有源，生产环境中应该设置具体的源
    # 代码说明：allow_credentials 表示是否允许跨域请求携带 cookie/认证信息；本行右侧表达式负责得到这个值。
    allow_credentials=True,
    # 代码说明：allow_methods 表示允许跨域访问的 HTTP 方法列表；本行右侧表达式负责得到这个值。
    allow_methods=["*"],  # 允许所有方法
    # 代码说明：allow_headers 表示允许跨域请求携带的请求头列表；本行右侧表达式负责得到这个值。
    allow_headers=["*"],  # 允许所有头
# 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
)

# 代码说明：storage_dir 表示后端本地 storage 目录，用来存 PDF、页面截图、图标裁剪图和查询图片；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
storage_dir = Path(os.getenv("STORAGE_ROOT", Path(__file__).resolve().parent / "storage"))
# 代码说明：storage_dir 表示后端本地 storage 目录，用来存 PDF、页面截图、图标裁剪图和查询图片；本行右侧表达式负责得到这个值。
storage_dir.mkdir(parents=True, exist_ok=True)
# 把浏览器 URL静态文件挂载到 /storage 路径
# 这样就可以通过http://localhost:8000/storage/manuals/demo_manual/crops/page_051_icon_003.png 访问backend/app/storage/manuals/demo_manual/crops/page_051_icon_003.png
# 代码说明：app 表示app；本行右侧表达式负责得到这个值。
app.mount("/storage", StaticFiles(directory=str(storage_dir)), name="storage")

# 代码说明：调用 app.include_router 执行当前步骤的副作用，例如写库、建索引、保存文件或更新状态。
app.include_router(health_rt.router)
# 代码说明：调用 app.include_router 执行当前步骤的副作用，例如写库、建索引、保存文件或更新状态。
app.include_router(manuals_rt.router)
# 代码说明：调用 app.include_router 执行当前步骤的副作用，例如写库、建索引、保存文件或更新状态。
app.include_router(qa_rt.router)

# 代码说明：判断条件 `__name__=='__main__'` 是否成立，成立时执行下面缩进代码。
if __name__=='__main__':
    # 代码说明：导入模块，uvicorn：提供本模块需要的标准库/第三方能力。
    import uvicorn
    # 代码说明：uvicorn 表示ASGI 服务器模块，本地直接运行 app_main.py 时用它启动 FastAPI；本行右侧表达式负责得到这个值。
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
