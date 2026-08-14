"""
RAG 配置 —— 所有路径和参数集中管理，禁止在其他文件中硬编码路径。
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import EMBEDDING_MODEL_ID, EMBEDDING_BACKEND, EMBEDDING_API_URL


@dataclass
class RAGConfig:
    # ========== 路径 ==========
    # PDF 说明书路径（单个 PDF 文件）
    pdf_path: str = ""

    # 索引持久化目录（存放 BM25 和 FAISS 索引）
    index_dir: str = ""

    # 提取的页面图片输出目录（可选，用于可视化/调试）
    page_image_dir: str = ""

    # ModelScope 模型缓存目录
    models_cache_dir: str = ""

    # ========== 嵌入模型（ModelScope / Ollama） ==========
    embedding_model_id: str = EMBEDDING_MODEL_ID  # 默认来自 config.py（可覆盖）
    # 嵌入后端: local=本地 sentence_transformers；ollama=远程 Ollama /v1/embeddings
    embedding_backend: str = EMBEDDING_BACKEND
    embedding_api_url: str = EMBEDDING_API_URL
    # 下载后的本地路径（由 download 逻辑自动填充）
    embedding_model_local: str = ""

    # ========== 分块参数 ==========
    chunk_min_chars: int = 20   # 少于这个字数的片段丢弃
    chunk_max_chars: int = 2000 # 超过则尝试再拆（按句子）

    # ========== 检索参数 ==========
    bm25_top_k: int = 20   # BM25 粗召回数量
    final_top_k: int = 3   # 向量精排后返回数量
    context_window: int = 1  # 邻块扩展：命中 chunk 时附带前后各 N 个相邻 chunk

    # ========== 设备 ==========
    device: str = "cpu"    # "cpu" | "cuda:0" | "cuda:1"

    # ========== 篇章结构头（用于分块识别） ==========
    # 标题自动探测: True=扫描全文自动发现编号风格; False=使用 section_patterns
    title_auto_detect: bool = True

    # 自动探测候选模式（按优先级排列）
    title_candidate_patterns: tuple = (
        r"^第[一二三四五六七八九十百\d]+[章节部分篇][：: ]?\S{1,30}$",
        r"^[一二三四五六七八九十百]+[、.．]\s*\S{1,30}$",
        r"^\d+(?:\.\d+)*[、.．]\s*\S{1,30}$",
        r"^(Chapter|Section|Step|Part)\s*\d+[.:]?\s*\S{1,30}$",
    )

    # 手动指定（title_auto_detect=False 时使用）
    section_patterns: tuple = (
        "一、", "二、", "三、", "四、", "五、", "六、",
        "七、", "八、", "九、", "十、",
    )
    step_pattern_prefix: tuple = ("1.", "2.", "3.", "4.", "5.")

    # 多级层级支持: 从粗到细的正则列表，空列表则仅按 section_patterns 切分
    level_patterns: tuple = ()  # 如 (r"第\d+章", r"\d+\.\d+", r"步骤\d+")

    # ========== 嵌入查询前缀 ==========
    query_prefix: str = "为这个句子生成表示以用于检索相关文章："

    def __post_init__(self):
        """自动推导未指定的路径。默认输出到 rag/ 子目录。"""
        rag_dir = os.path.dirname(os.path.abspath(__file__))  # rag/ 目录本身
        if not self.index_dir:
            self.index_dir = os.path.join(rag_dir, "index")
        if not self.page_image_dir:
            self.page_image_dir = os.path.join(rag_dir, "pages")
        if not self.models_cache_dir:
            self.models_cache_dir = os.path.expanduser("~/workspace/models")

        # 归一化路径
        for attr in ("pdf_path", "index_dir", "page_image_dir",
                     "models_cache_dir", "embedding_model_local"):
            val = getattr(self, attr, "")
            if val:
                setattr(self, attr, os.path.abspath(os.path.expanduser(val)))
