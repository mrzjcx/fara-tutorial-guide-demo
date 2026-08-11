"""
PDF 解析器：提取文本 + 渲染页面图片。
"""
import os
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """单页解析结果"""
    page_num: int           # 1-indexed
    text: str               # 该页提取的文本
    char_count: int         # 文本字数
    image_count: int        # 嵌入图片数
    page_image_path: str = ""  # 渲染的页面截图路径


@dataclass
class PDFData:
    """整份 PDF 解析结果"""
    path: str
    total_pages: int
    pages: List[PageData] = field(default_factory=list)


def parse_pdf(pdf_path: str, page_image_dir: str = "") -> PDFData:
    """
    解析 PDF 文件，提取每页文本并可选渲染页面图片。

    Args:
        pdf_path: PDF 文件路径
        page_image_dir: 如果不为空，将每页渲染为 PNG 存入此目录

    Returns:
        PDFData: 包含所有页面的文本和元数据
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    pdf_data = PDFData(path=pdf_path, total_pages=doc.page_count)

    if page_image_dir:
        os.makedirs(page_image_dir, exist_ok=True)

    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text().strip()
        images = page.get_images()

        page_data = PageData(
            page_num=i + 1,
            text=text,
            char_count=len(text),
            image_count=len(images),
        )

        # 渲染页面为 150 DPI 的图片
        if page_image_dir:
            img_name = f"page_{i+1:03d}.png"
            img_path = os.path.join(page_image_dir, img_name)
            # 只对首次构建渲染（已存在就跳过）
            if not os.path.exists(img_path):
                pix = page.get_pixmap(dpi=150)
                pix.save(img_path)
                logger.debug(f"页面 {i+1} 渲染: {img_path}")
            page_data.page_image_path = img_path

        pdf_data.pages.append(page_data)
        logger.info(f"第{i+1}页: {page_data.char_count}字, {page_data.image_count}图")

    doc.close()
    logger.info(f"PDF 解析完成: {pdf_data.total_pages}页")
    return pdf_data
