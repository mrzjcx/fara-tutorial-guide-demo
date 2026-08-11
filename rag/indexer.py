"""
索引构建器：构建 BM25（关键词）+ FAISS（向量）双索引，支持持久化。
"""
import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

from .chunker import Chunk
from .embedder import Embedder
from .config import RAGConfig

logger = logging.getLogger(__name__)


class DualIndex:
    """
    BM25 + FAISS 双索引。

    - BM25: 关键词粗召回（高召回率，速度快）
    - FAISS: 向量精排（语义匹配，精度高）
    """

    def __init__(self, config: RAGConfig):
        self._config = config
        self._chunks: List[Chunk] = []
        self._chunk_texts: List[str] = []

        # BM25
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []

        # FAISS
        self._faiss_index = None
        self._embeddings: np.ndarray = None  # 保留用于重加载

        # chunk_id → 索引位置 映射
        self._id_to_idx: Dict[str, int] = {}

    # ==================== 构建 ====================

    def build(self, chunks: List[Chunk], embedder: Embedder):
        """
        构建双索引。

        Args:
            chunks: 分块列表
            embedder: 嵌入模型
        """
        if not chunks:
            raise ValueError("chunks 为空，无法构建索引")

        self._chunks = chunks
        self._chunk_texts = [c.text for c in chunks]

        # chunk_id → 索引位置 映射（供邻块扩展使用）
        self._id_to_idx: Dict[str, int] = {
            c.chunk_id: i for i, c in enumerate(chunks)
        }

        # ---- BM25 ----
        self._build_bm25()

        # ---- FAISS ----
        self._build_faiss(embedder)

        logger.info(f"索引构建完成: {len(chunks)} chunks, "
                     f"BM25={len(self._tokenized_corpus)} docs, "
                     f"FAISS={self._embeddings.shape}")

    def _build_bm25(self):
        """构建 BM25 关键词索引。"""
        from rank_bm25 import BM25Okapi
        import jieba

        self._tokenized_corpus = [
            list(jieba.cut(text)) for text in self._chunk_texts
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 索引: {len(self._tokenized_corpus)} 文档")

    def _build_faiss(self, embedder: Embedder):
        """构建 FAISS 向量索引。"""
        import faiss

        logger.info("编码所有 chunks...")
        self._embeddings = embedder.encode(self._chunk_texts, show_progress=True)

        dim = self._embeddings.shape[1]
        # 使用内积搜索（因为向量已 L2 归一化，内积 = 余弦相似度）
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._embeddings)
        logger.info(f"FAISS 索引: {self._faiss_index.ntotal} 向量, dim={dim}")

    # ==================== 检索 ====================

    def search_bm25(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        BM25 关键词检索。

        Returns:
            [(chunk_index, bm25_score), ...] 按分数降序
        """
        import jieba

        if top_k is None:
            top_k = self._config.bm25_top_k
        if not self._bm25:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 返回 top-k 索引和分数
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in indices if scores[i] > 0]

    def search_faiss(self, query_vector: np.ndarray, top_k: int = None) -> List[Tuple[int, float]]:
        """
        FAISS 向量检索。

        Args:
            query_vector: shape (dim,) or (1, dim)
            top_k: 返回数量，默认 final_top_k

        Returns:
            [(chunk_index, similarity_score), ...] 按分数降序
        """
        if top_k is None:
            top_k = self._config.final_top_k
        if self._faiss_index is None:
            return []

        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        scores, indices = self._faiss_index.search(vec, min(top_k, self._faiss_index.ntotal))
        return [(int(indices[0][j]), float(scores[0][j]))
                for j in range(len(indices[0])) if indices[0][j] >= 0]

    def get_chunk(self, idx: int) -> Chunk:
        """按索引获取 Chunk。"""
        return self._chunks[idx]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """按 chunk_id 获取 Chunk（供邻块扩展使用）。"""
        idx = self._id_to_idx.get(chunk_id)
        return self._chunks[idx] if idx is not None else None

    def __len__(self):
        return len(self._chunks)

    # ==================== 持久化 ====================

    def save(self, dir_path: str):
        """保存索引到磁盘。"""
        os.makedirs(dir_path, exist_ok=True)

        # BM25 用 pickle
        bm25_path = os.path.join(dir_path, "bm25_index.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "chunk_texts": self._chunk_texts,
                "tokenized_corpus": self._tokenized_corpus,
                "id_to_idx": self._id_to_idx,
            }, f)

        # FAISS 用原生 save
        if self._faiss_index is not None:
            import faiss
            faiss_path = os.path.join(dir_path, "faiss.index")
            faiss.write_index(self._faiss_index, faiss_path)

        # Embeddings
        if self._embeddings is not None:
            emb_path = os.path.join(dir_path, "embeddings.npy")
            np.save(emb_path, self._embeddings)

        logger.info(f"索引已保存: {dir_path}")

    @classmethod
    def load(cls, dir_path: str, config: RAGConfig) -> "DualIndex":
        """从磁盘加载索引。"""
        import faiss

        obj = cls(config)

        # BM25
        bm25_path = os.path.join(dir_path, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            from rank_bm25 import BM25Okapi
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            obj._chunks = data["chunks"]
            obj._chunk_texts = data["chunk_texts"]
            obj._tokenized_corpus = data["tokenized_corpus"]
            obj._id_to_idx = data.get("id_to_idx", {})
            obj._bm25 = BM25Okapi(obj._tokenized_corpus)
            logger.info(f"BM25 索引已加载: {len(obj._chunks)} chunks")

        # FAISS
        faiss_path = os.path.join(dir_path, "faiss.index")
        if os.path.exists(faiss_path):
            obj._faiss_index = faiss.read_index(faiss_path)

        # Embeddings
        emb_path = os.path.join(dir_path, "embeddings.npy")
        if os.path.exists(emb_path):
            obj._embeddings = np.load(emb_path)

        logger.info(f"索引已加载: {dir_path}")
        return obj
