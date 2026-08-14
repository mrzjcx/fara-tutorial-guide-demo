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
    文本向量化器。

    后端可选：
      - local : 本地 sentence_transformers + ModelScope bge（默认）
      - ollama: 远程 Ollama /v1/embeddings（如 10.17.83.10:11434，bge-m3:latest）

    Usage:
        emb = Embedder(config)
        vectors = emb.encode(["文本1", "文本2"])  # shape: (n, dim)
    """

    def __init__(self, config: RAGConfig):
        self._config = config
        self._model = None
        self._dim = EMBEDDING_DIM  # 默认来自 config.py（模型加载后以实际维度为准）
        self._ollama = (config.embedding_backend == "ollama" and bool(config.embedding_api_url))

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_model(self):
        """延迟加载模型（首次 encode 时才下载/加载）；ollama 远程模式无需本地模型。"""
        if self._ollama:
            return
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "本地嵌入需要 sentence-transformers，但未安装。两种解决方式：\n"
                "  1) pip install sentence-transformers（本地 CPU bge 嵌入）\n"
                "  2) 在项目根目录 .env 配置外部嵌入（无需额外安装）:\n"
                "     EMBEDDING_BACKEND=ollama\n"
                "     EMBEDDING_API_URL=http://<外部IP>:11434/v1/embeddings\n"
                "     EMBEDDING_MODEL_ID=bge-m3:latest\n"
                "     EMBEDDING_DIM=1024"
            ) from e

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

        Returns:
            np.ndarray: shape (len(texts), dim)，L2 归一化
        """
        if self._ollama:
            return self._encode_ollama(texts)
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

    def _encode_ollama(self, texts: List[str]) -> np.ndarray:
        """远程 Ollama /v1/embeddings 编码（OpenAI 兼容格式）。"""
        import requests
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        # 批量上限 32，分批调用
        out = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            resp = requests.post(
                self._config.embedding_api_url,
                json={"model": self._config.embedding_model_id, "input": batch},
                timeout=(5, 120),  # 5s 连接超时（外部不可达快速失败），120s 读超时
            )
            resp.raise_for_status()
            data = resp.json()
            out.extend(item["embedding"] for item in data.get("data", []))
        arr = np.array(out, dtype=np.float32)
        # L2 归一化（与本地 bge 行为一致，便于余弦/内积检索）
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self._dim = arr.shape[1]
        return arr

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本（P1-⑥: 前缀从 config 读取）。"""
        prefix = getattr(self._config, 'query_prefix', "为这个句子生成表示以用于检索相关文章：")
        query_text = f"{prefix}{query}"
        return self.encode([query_text])
