"""
RAG 模块 CLI 入口：构建索引 / 交互查询。

Usage:
    # 构建索引（models-cache 默认 ~/workspace/models，可用 config.py 的 MODELS_CACHE）
    python -m rag build --pdf docs/manual/manual.pdf --models-cache ~/workspace/models

    # 交互查询
    python -m rag query --pdf docs/manual/manual.pdf --models-cache ~/workspace/models
"""
import argparse
import logging
import sys
import os

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig
from rag.retriever import RAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rag")


def cmd_build(args):
    config = RAGConfig(
        pdf_path=args.pdf,
        index_dir=args.index_dir,
        models_cache_dir=args.models_cache,
        device=args.device,
    )
    retriever = RAGRetriever(config)
    retriever.build_index(force=args.force)
    print(f"\n[OK] 索引已保存到: {config.index_dir}")


def cmd_query(args):
    config = RAGConfig(
        pdf_path=args.pdf,
        index_dir=args.index_dir,
        models_cache_dir=args.models_cache,
        device=args.device,
    )
    retriever = RAGRetriever(config)

    if args.interactive:
        print("\n[检索] RAG 交互查询模式（输入 quit/exit/q 退出）\n")
        while True:
            try:
                query = input("查询> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break

            results = retriever.query(query, top_k=args.top_k)
            _print_results(results)
    else:
        results = retriever.query(args.query, top_k=args.top_k)
        _print_results(results)


def _print_results(results):
    if not results:
        print("  (无结果)")
        return
    for i, r in enumerate(results, 1):
        print(f"\n--- 结果 {i} (score={r['score']}, page={r['page']}) ---")
        print(f"  章节: {r['section']}")
        print(f"  步骤: {r['step']}")
        print(f"  内容: {r['text'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Fara RAG 模块")
    sub = parser.add_subparsers(dest="command")

    # ---- build ----
    p_build = sub.add_parser("build", help="构建索引")
    p_build.add_argument("--pdf", required=True, help="PDF 说明书路径")
    p_build.add_argument("--index-dir", default="", help="索引保存目录（默认自动推导）")
    p_build.add_argument("--models-cache", required=True, help="ModelScope 模型缓存目录")
    p_build.add_argument("--device", default="cpu", help="设备 (cpu/cuda:0/cuda:1)")
    p_build.add_argument("--force", action="store_true", help="强制重建索引")

    # ---- query ----
    p_query = sub.add_parser("query", help="查询")
    p_query.add_argument("--pdf", required=True, help="PDF 说明书路径")
    p_query.add_argument("--index-dir", default="", help="索引目录（默认自动推导）")
    p_query.add_argument("--models-cache", required=True, help="ModelScope 模型缓存目录")
    p_query.add_argument("--device", default="cpu", help="设备")
    p_query.add_argument("--query", default="", help="单次查询文本")
    p_query.add_argument("--top-k", type=int, default=3, help="返回数量")
    p_query.add_argument("-i", "--interactive", action="store_true", help="交互模式")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
