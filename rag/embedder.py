"""
嵌入模型封装：ModelScope BGE 模型，将文本转为向量。
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import EMBEDDING_DIM

from .config import RAGConfig

logger = logging.getLogger(__name__)


class Embedder:
    """
    文本向量化器，基于 ModelScope 的 BGE 中文嵌入模型。

    Usage:
        emb = Embedder(config)
        vectors = emb.encode(["文本1", "文本2"])  # shape: (2, 512)
    """

    def __init__(self, config: RAGConfig):
        self._config = config
        self._model = None
        self._dim = EMBEDDING_DIM  # 默认来自 config.py（模型加载后以实际维度为准）

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_model(self):
        """延迟加载模型（首次 encode 时才下载/加载）。"""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        model_path = self._resolve_model_path()
        logger.info(f"加载嵌入模型: {model_path}")
        self._model = SentenceTransformer(
            model_path,
            device=self._config.device,
        )
        # 获取实际维度
        test_vec = self._model.encode("test", show_progress_bar=False)
        self._dim = test_vec.shape[0]
        logger.info(f"嵌入模型就绪: dim={self._dim}")

    def _resolve_model_path(self) -> str:
        """解析模型本地路径 —— 优先本地已下载，否则从 ModelScope 下载。"""
        # 1) 用户显式指定的本地路径
        if self._config.embedding_model_local and os.path.isdir(
            self._config.embedding_model_local
        ):
            return self._config.embedding_model_local

        # 2) 检查 ModelScope 缓存目录下的标准命名
        model_id = self._config.embedding_model_id
        # ModelScope 下载后路径格式: {cache}/models/{org}--{name}/snapshots/master
        org, name = model_id.split("/", 1)
        cache_path = os.path.join(
            self._config.models_cache_dir, "models",
            f"{org}--{name}", "snapshots", "master"
        )
        if os.path.isdir(cache_path):
            return cache_path

        # 3) 从 ModelScope 下载
        logger.info(f"从 ModelScope 下载模型: {model_id}")
        from modelscope import snapshot_download
        downloaded = snapshot_download(
            model_id,
            cache_dir=self._config.models_cache_dir,
        )
        # 更新配置中的本地路径供后续复用
        self._config.embedding_model_local = downloaded
        return downloaded

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        将文本列表编码为向量矩阵。

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            np.ndarray: shape (len(texts), dim)
        """
        self._ensure_model()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # BGE 模型推荐对查询加前缀 "为这个句子生成表示以用于检索相关文章："
        embeddings = self._model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2 归一化，便于内积相似度
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本（P1-⑥: 前缀从 config 读取）。"""
        prefix = getattr(self._config, 'query_prefix', "为这个句子生成表示以用于检索相关文章：")
        query_text = f"{prefix}{query}"
        return self.encode([query_text])
