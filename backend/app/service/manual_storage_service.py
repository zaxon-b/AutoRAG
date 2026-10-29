"""手册本地存储：统一管理 PDF、页面截图、图片资产和用户查询图的路径。"""

import os
import shutil
from pathlib import Path
from typing import BinaryIO


DEFAULT_MANUAL_ID = os.getenv("DEMO_MANUAL_ID", "demo_manual")


def storage_root() -> Path:
    """存储根目录，默认 backend/app/storage，可用 STORAGE_ROOT 覆盖。"""

    default_root = Path(__file__).resolve().parents[1] / "storage"
    return Path(os.getenv("STORAGE_ROOT", str(default_root))).resolve()


def manual_root(manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    return storage_root() / "manuals" / manual_id


def reset_manual_storage(manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    """删除旧目录后重建，避免上一次建库的截图混进新结果。"""

    root = manual_root(manual_id)
    if root.exists():
        shutil.rmtree(root)
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "crops").mkdir(parents=True, exist_ok=True)
    (root / "query_images").mkdir(parents=True, exist_ok=True)
    return root


def ensure_manual_storage(manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    """只补建缺失目录，不清空已有文件。"""

    root = manual_root(manual_id)
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "crops").mkdir(parents=True, exist_ok=True)
    (root / "query_images").mkdir(parents=True, exist_ok=True)
    return root


def save_manual_pdf(file_obj: BinaryIO, filename: str, manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    """把上传的 PDF 保存为 original.pdf，并重置该手册的存储目录。"""

    root = reset_manual_storage(manual_id)
    target = root / "original.pdf"
    with open(target, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)
    return target


def save_query_image(content: bytes, filename: str, manual_id: str = DEFAULT_MANUAL_ID) -> Path:
    """保存图片问答的查询图；同名文件会被覆盖。"""

    root = ensure_manual_storage(manual_id)
    suffix = Path(filename or "query.png").suffix.lower() or ".png"
    target = root / "query_images" / f"query{suffix}"
    target.write_bytes(content)
    return target


def to_storage_url(path: str | Path | None) -> str | None:
    """把 storage 内的绝对路径转成 /storage/... 静态 URL。

    不在 storage 目录下的路径原样返回，便于调用方发现配置错误。
    """

    if not path:
        return None
    path_obj = Path(path).resolve()
    try:
        relative = path_obj.relative_to(storage_root())
        return f"/storage/{relative.as_posix()}"
    except ValueError:
        return str(path_obj)
