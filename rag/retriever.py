"""
RAG 检索器：两阶段检索（BM25 粗召回 → 向量精排），对外暴露统一查询接口。
"""
import os
import re
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import FARA_API_URL, FARA_MODEL_PATH

from .config import RAGConfig
from .pdf_parser import parse_pdf
from .chunker import chunk_pdf, Chunk
from .embedder import Embedder
from .indexer import DualIndex

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    RAG 检索器 —— 外部唯一入口。

    Usage:
        config = RAGConfig(
            pdf_path="docs/manual/manual.pdf",
            index_dir="rag_index",
            models_cache_dir="~/workspace/models",
        )
        rag = RAGRetriever(config)

        # 首次使用需要构建索引
        rag.build_index()

        # 查询
        results = rag.query("如何在线挂号", top_k=3)
        for r in results:
            print(r["section"], r["text"][:100], r["score"])
    """

    def __init__(self, config: RAGConfig):
        self._config = config
        self._embedder: Optional[Embedder] = None
        self._index: Optional[DualIndex] = None
        self._chunks: List[Chunk] = []

        # 尝试加载已有索引
        self._try_load_index()

    # ==================== 索引构建 ====================

    def build_index(self, force: bool = False):
        """
        构建 RAG 索引（解析 PDF → 分块 → 编码 → 建索引 → 持久化）。

        Args:
            force: 是否强制重建（即使已有索引）
        """
        if self._index is not None and len(self._index) > 0 and not force:
            logger.info("索引已存在，跳过构建。使用 force=True 强制重建。")
            return

        pdf_path = self._config.pdf_path
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

        # 1) 解析 PDF
        logger.info("=== Step 1: 解析 PDF ===")
        pdf_data = parse_pdf(pdf_path, page_image_dir=self._config.page_image_dir)

        # 2) 分块
        logger.info("=== Step 2: 文本分块 ===")
        chunks = chunk_pdf(pdf_data, self._config)
        if not chunks:
            raise ValueError("未能从 PDF 中提取有效文本块")

        self._chunks = chunks

        # 3) 构建双索引
        logger.info("=== Step 3: 构建索引 ===")
        embedder = self._get_embedder()
        self._index = DualIndex(self._config)
        self._index.build(chunks, embedder)

        # 4) 持久化
        logger.info("=== Step 4: 持久化索引 ===")
        self._index.save(self._config.index_dir)
        logger.info("[OK] 索引构建完成")

    # ==================== 查询 ====================

    def query(self, intent: str, top_k: int = None) -> List[Dict]:
        """
        两阶段检索：BM25 粗召回 → 向量精排。

        Args:
            intent: 用户意图/查询文本
            top_k: 最终返回数量，默认取 config.final_top_k

        Returns:
            [{"chunk_id", "section", "step", "text", "page", "score", "image_path"}, ...]
        """
        if top_k is None:
            top_k = self._config.final_top_k

        if self._index is None:
            raise RuntimeError("索引未初始化，请先调用 build_index()")

        # ---- Stage 1: BM25 粗召回 ----
        bm25_candidates = self._index.search_bm25(intent)
        if not bm25_candidates:
            logger.warning("BM25 无结果，回退到纯 FAISS")
            # 回退：全量 FAISS
            return self._faiss_only_query(intent, top_k)

        logger.debug(f"BM25 召回 {len(bm25_candidates)} 候选")

        # ---- Stage 2: 向量精排 ----
        embedder = self._get_embedder()
        query_vec = embedder.encode_query(intent)

        # 获取候选文本
        candidate_indices = [idx for idx, _ in bm25_candidates]
        candidate_texts = [self._index.get_chunk(idx).text for idx in candidate_indices]

        # 编码候选文本
        candidate_vecs = embedder.encode(candidate_texts)

        # 计算余弦相似度（向量已 L2 归一化，内积 = 余弦）
        import numpy as np
        query_vec_2d = query_vec.reshape(1, -1)
        similarities = np.dot(query_vec_2d, candidate_vecs.T).flatten()

        # 按相似度降序排列
        ranked = sorted(
            zip(candidate_indices, similarities),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return self._format_results(ranked)

    def _faiss_only_query(self, intent: str, top_k: int) -> List[Dict]:
        """纯 FAISS 检索（BM25 无结果时的回退）。"""
        embedder = self._get_embedder()
        query_vec = embedder.encode_query(intent)
        results = self._index.search_faiss(query_vec, top_k)
        return self._format_results(results)

    def _format_results(self, ranked: List[tuple]) -> List[Dict]:
        """将 (index, score) 列表格式化为结果字典，自动附加上下文邻块。"""
        output = []
        for idx, score in ranked:
            chunk = self._index.get_chunk(idx)
            neighbors = self._get_neighbor_texts(chunk.chunk_id)
            context_text = "\n".join(
                [f"[{n['chunk_id']}] {n['text']}" for n in neighbors]
            ) if neighbors else ""

            output.append({
                "chunk_id": chunk.chunk_id,
                "section": chunk.section_title,
                "step": chunk.step_title,
                "text": chunk.text,
                "hint": chunk.metadata.get("hint", ""),
                "context_text": context_text,
                "neighbors": neighbors,
                "page": chunk.page_start,
                "score": round(float(score), 4),
                "image_path": chunk.page_image_path,
            })
        return output

    # ==================== 说明书摘要 ====================

    def get_summary(self, force_regenerate: bool = False) -> str:
        """
        生成说明书结构摘要。

        优先从磁盘加载 Fara 生成的缓存摘要（一次调用，永久复用）；
        若缓存不存在且 Fara 可用则调 Fara 生成；
        否则回退到基于 chunk 字段的规则摘要。
        """
        if not self._index or len(self._index) == 0:
            return "(index not loaded)"

        # 1) 尝试加载缓存
        cache_path = os.path.join(self._config.index_dir, "summary_fara.txt")
        if not force_regenerate and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        # 2) 尝试调用 Fara 生成（离线阶段，一次性）
        fara_summary = self._try_fara_summary()
        if fara_summary:
            os.makedirs(self._config.index_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(fara_summary)
            logger.info(f"Fara 摘要已缓存: {cache_path}")
            return fara_summary

        # 3) 回退：规则摘要
        return self._rule_based_summary()

    def _try_fara_summary(self) -> str:
        """尝试调用 Fara 生成摘要。需 Fara 服务运行中。"""
        try:
            # 收集所有 chunk 文本（限制总长度避免超出 context）
            chunk_lines = []
            total_chars = 0
            for i in range(len(self._index)):
                c = self._index.get_chunk(i)
                line = f"[{c.chunk_id}] {c.step_title}: {c.text[:120]}"
                chunk_lines.append(line)
                total_chars += len(line)
                if total_chars > 6000:  # 限制 ~6000 字符，约 1500 tokens
                    chunk_lines.append("...(后续省略)")
                    break

            all_chunks_text = "\n".join(chunk_lines)

            prompt = """You are a document structure analyst. Given a list of text chunks from a manual, generate a structured summary.

Requirements:
1. List all major sections/workflows found in the document
2. Under each section, list its steps (one per line)
3. Output ONLY the summary, no extra commentary
4. Format:
Document Structure:
[Section Title]
  [Step ID] [Step Name] - [brief description]

Chunks:
"""
            import requests
            resp = requests.post(
                FARA_API_URL,
                json={
                    "model": FARA_MODEL_PATH,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": all_chunks_text},
                    ],
                    "max_tokens": 600,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Fara 摘要生成失败: {e}")
        return ""

    def _rule_based_summary(self) -> str:
        """回退规则摘要（不依赖 Fara）。"""
        lines = ["Document Structure:"]
        current_section = ""

        for i in range(len(self._index)):
            chunk = self._index.get_chunk(i)
            section = chunk.section_title
            step = chunk.step_title

            if section != current_section:
                current_section = section
                if section not in ("概述", ""):
                    name = self._get_section_name(i)
                    lines.append(f"{section}{name}")

            if step == "概述" or step == section:
                continue

            text = chunk.text.replace("\n", " ")[:80]
            lines.append(f"  {step} - {text}")

        return "\n".join(lines)

    def _get_section_name(self, start_idx: int) -> str:
        for offset in range(3):
            idx = start_idx + offset
            if idx >= len(self._index):
                break
            chunk = self._index.get_chunk(idx)
            if chunk.step_title == "概述" and chunk.section_title != "概述":
                name = chunk.text.split("\n")[0].strip()
                if name and len(name) < 20:
                    return name
        return ""

    # ==================== 邻块扩展（P0-②: 按数组位置下标） ====================

    def _get_neighbor_texts(self, chunk_id: str) -> List[Dict]:
        """按 chunk 在索引数组中的相邻位置获取邻块，不依赖 chunk_id 格式。"""
        idx = self._index._id_to_idx.get(chunk_id)
        if idx is None:
            return []
        window = self._config.context_window
        lo = max(0, idx - window)
        hi = min(len(self._index), idx + window + 1)
        result = []
        for i in range(lo, hi):
            if i == idx:
                continue
            c = self._index.get_chunk(i)
            result.append({
                "chunk_id": c.chunk_id,
                "step": c.step_title,
                "text": c.text,
                "offset": i - idx,
            })
        return result

    # ==================== 内部 ====================

    def _get_embedder(self) -> Embedder:
        """获取或创建嵌入模型（单例）。"""
        if self._embedder is None:
            self._embedder = Embedder(self._config)
        return self._embedder

    def _try_load_index(self):
        """尝试从磁盘加载已有索引。"""
        idx_dir = self._config.index_dir
        bm25_file = os.path.join(idx_dir, "bm25_index.pkl")
        faiss_file = os.path.join(idx_dir, "faiss.index")

        if os.path.exists(bm25_file) and os.path.exists(faiss_file):
            logger.info(f"加载已有索引: {idx_dir}")
            try:
                self._index = DualIndex.load(idx_dir, self._config)
                self._chunks = self._index._chunks if hasattr(self._index, '_chunks') else []
            except Exception as e:
                logger.warning(f"索引加载失败: {e}，将重新构建")
                self._index = None
                self._chunks = []  # 清理旧数据
