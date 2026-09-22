# 模块说明：本地存储服务：统一管理 PDF、页面截图、图标裁剪图、用户查询图片的保存路径和浏览器访问 URL。
# 调用关系：由手册上传、页面渲染、视觉裁剪、图片问答调用。
# 返回/产出：返回本地文件路径或 /storage 开头的静态访问路径。
# 代码说明：导入模块，os：读取环境变量，例如模型 key、阈值、数据库地址和存储目录。
import os
# 代码说明：导入模块，shutil：提供本模块需要的标准库/第三方能力。
import shutil
# 代码说明：从 pathlib 导入：Path：使用 pathlib 统一处理本地文件和目录路径。
from pathlib import Path
# 代码说明：从 typing 导入：BinaryIO：本文件后续逻辑依赖的对象。
from typing import BinaryIO


# 代码说明：DEFAULT_MANUAL_ID 表示默认手册 ID，单手册 Demo 下通常是 demo_manual；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
DEFAULT_MANUAL_ID = os.getenv("DEMO_MANUAL_ID", "demo_manual")


# 函数说明：storage_root：返回本地存储根目录，默认是 backend/app/storage，也可由 STORAGE_ROOT 配置。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 storage_root：返回本地存储根目录，默认是 backend/app/storage，也可由 STORAGE_ROOT 配置。
def storage_root() -> Path:
    # 代码说明：default_root 表示default / 根目录；本行右侧表达式负责得到这个值。
    default_root = Path(__file__).resolve().parents[1] / "storage"
    # 代码说明：返回调用结果：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里，调用方会拿这个结果继续下一步。
    return Path(os.getenv("STORAGE_ROOT", str(default_root))).resolve()


# 函数说明：manual_root：返回某本手册在本地存储中的目录。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 manual_root：返回某本手册在本地存储中的目录。
def manual_root(manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return storage_root() / "manuals" / manual_id


# 函数说明：reset_manual_storage：删除并重建某本手册的存储目录，避免旧截图污染新建库。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 reset_manual_storage：删除并重建某本手册的存储目录，避免旧截图污染新建库。
def reset_manual_storage(manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    # 代码说明：root：本地存储根目录；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    root = manual_root(manual_id)
    # 代码说明：判断条件 `root.exists()` 是否成立，成立时执行下面缩进代码。
    if root.exists():
        # 代码说明：调用 shutil.rmtree 执行当前步骤的副作用，例如写库、建索引、保存文件或更新状态。
        shutil.rmtree(root)
    # 代码说明：(root / "pages") 表示(root / / / "pages")；本行右侧表达式负责得到这个值。
    (root / "pages").mkdir(parents=True, exist_ok=True)
    # 代码说明：(root / "crops") 表示(root / / / "crops")；本行右侧表达式负责得到这个值。
    (root / "crops").mkdir(parents=True, exist_ok=True)
    # 代码说明：(root / "query_images") 表示(root / / / "query / images")；本行右侧表达式负责得到这个值。
    (root / "query_images").mkdir(parents=True, exist_ok=True)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return root


# 函数说明：ensure_manual_storage：确保某本手册的 pages/crops/query_images 等目录存在。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 ensure_manual_storage：确保某本手册的 pages/crops/query_images 等目录存在。
def ensure_manual_storage(manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    # 代码说明：root：本地存储根目录；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    root = manual_root(manual_id)
    # 代码说明：(root / "pages") 表示(root / / / "pages")；本行右侧表达式负责得到这个值。
    (root / "pages").mkdir(parents=True, exist_ok=True)
    # 代码说明：(root / "crops") 表示(root / / / "crops")；本行右侧表达式负责得到这个值。
    (root / "crops").mkdir(parents=True, exist_ok=True)
    # 代码说明：(root / "query_images") 表示(root / / / "query / images")；本行右侧表达式负责得到这个值。
    (root / "query_images").mkdir(parents=True, exist_ok=True)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return root


# 函数说明：save_manual_pdf：把上传的 PDF 写入本地手册目录。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 save_manual_pdf：把上传的 PDF 写入本地手册目录。
def save_manual_pdf(file_obj: BinaryIO, filename: str, manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    # 代码说明：root：本地存储根目录；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    root = reset_manual_storage(manual_id)
    # 代码说明：target 表示target；本行右侧表达式负责得到这个值。
    target = root / "original.pdf"
    # 代码说明：进入上下文管理 `open(target, "wb") as buffer`，用于自动关闭文件、数据库连接或外部资源。
    with open(target, "wb") as buffer:
        # 代码说明：调用 shutil.copyfileobj 执行当前步骤的副作用，例如写库、建索引、保存文件或更新状态。
        shutil.copyfileobj(file_obj, buffer)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return target


# 函数说明：save_query_image：把用户图片问答上传的图片保存到 query_images 目录。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 save_query_image：把用户图片问答上传的图片保存到 query_images 目录。
def save_query_image(content: bytes, filename: str, manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    # 代码说明：root：本地存储根目录；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    root = ensure_manual_storage(manual_id)
    # 代码说明：suffix：上传文件名的后缀，用来拦截非图片格式，避免后续 OpenCV/VLM 读取失败；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    suffix = Path(filename or "query.png").suffix.lower() or ".png"
    # 代码说明：target 表示target；本行右侧表达式负责得到这个值。
    target = root / "query_images" / f"query{suffix}"
    # 代码说明：把处理结果写入文件，用于保存 PDF、图片、评测报告或图标目录。
    target.write_bytes(content)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return target


# 函数说明：to_storage_url：把本地文件路径转换成前端可访问的 /storage URL。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 to_storage_url：把本地文件路径转换成前端可访问的 /storage URL。
def to_storage_url(path: str | Path | None) -> str | None:
    # 代码说明：判断条件 `not path` 是否成立，成立时执行下面缩进代码。
    if not path:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None
    # 代码说明：path_obj 表示路径 / obj；本行右侧表达式负责得到这个值。
    path_obj = Path(path).resolve()
    # 代码说明：进入异常保护块：下面的外部接口、文件、数据库或模型调用可能失败，需要捕获错误。
    try:
        # 代码说明：relative 表示相对路径；本行右侧表达式负责得到这个值。
        relative = path_obj.relative_to(storage_root())
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return f"/storage/{relative.as_posix()}"
    # 代码说明：捕获异常 `ValueError`，把底层错误转换成当前模块可处理的返回或提示。
    except ValueError:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return str(path_obj)
