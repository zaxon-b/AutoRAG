"""把 PDF 每页渲染成 PNG，供页面截图证据和前端缩略图使用。"""

from pathlib import Path


def render_pdf_pages(pdf_path: Path, output_dir: Path, zoom: float = 1.6) -> dict[int, dict]:
    """渲染全部页面，返回以 1-based 页码为键的图片路径与尺寸。

    未安装 PyMuPDF 或渲染失败时返回空字典，让建库流程决定是否继续。
    """

    try:
        import fitz
    except Exception:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[int, dict] = {}
    document = fitz.open(str(pdf_path))
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{index:03d}.png"
            pixmap.save(str(image_path))
            rendered[index] = {
                "image_path": str(image_path),
                "width": pixmap.width,
                "height": pixmap.height,
            }
    finally:
        document.close()

    return rendered
