"""
RAG 模块：PDF 说明书解析 → 切片 → BM25+向量双索引 → 两阶段检索

Usage:
    from rag import RAGRetriever, RAGConfig

    config = RAGConfig(
        pdf_path="docs/manual/manual.pdf",
        index_dir="rag_index",
        models_cache_dir="~/workspace/models",
    )
    retriever = RAGRetriever(config)
    retriever.build_index()          # 首次构建
    results = retriever.query("如何在线挂号", top_k=3)
"""

from .config import RAGConfig
from .retriever import RAGRetriever
from .chunker import Chunk

__all__ = ["RAGConfig", "RAGRetriever", "Chunk"]
