#!/usr/bin/env python3
"""
Fara Tutorial Guide RAG — Demo 完整工作流

用法:
    # 用已有截图测试
    python demo.py --screenshot docs/img/02_register_dept.png --intent "在线挂号选骨科"

    # 实时截图（需 Playwright + 网页服务运行中）
    python demo.py --serve-url http://localhost:<SERVE_PORT> --intent "我要缴费"   # 端口见 config.py

    # 不启用 Checker
    python demo.py --screenshot xxx.png --intent "挂号" --no-checker
"""
import argparse
import base64
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# 项目根加入 sys.path：scripts.config 通过包路径导入（与 fara/checker/rag 的 config 无冲突）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.config as pc
from rag import RAGConfig, RAGRetriever
from fara import FaraConfig, FaraActor
from checker import CheckerConfig, Checker
from loop import FaraCheckerLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("demo")


# ============================================================
# 默认配置
# ============================================================

DEFAULTS = {
    "fara_api": pc.FARA_API_URL,
    "fara_model": pc.FARA_MODEL_PATH,
    "checker_api": pc.CHECKER_API_URL,
    "checker_model": pc.CHECKER_MODEL_PATH,
    "pdf_path": pc.PDF_PATH,
    "models_cache": pc.MODELS_CACHE,
    "rag_top_k": pc.RAG_TOP_K,
    "max_retries": pc.MAX_RETRIES,
    "serve_url": pc.SERVE_URL,
}


# ============================================================
# 工具函数
# ============================================================

def load_screenshot(path: str) -> str:
    """读取截图文件 → base64 编码。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def capture_screenshot(url: str, save_path: str = None) -> str:
    """用 Playwright 截取网页截图 → base64。"""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 720, "height": 900})
        page.goto(url)
        page.wait_for_timeout(500)
        img_bytes = page.screenshot(full_page=True)
        browser.close()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(img_bytes)

    return base64.b64encode(img_bytes).decode("utf-8")


# ============================================================
# 主流程
# ============================================================

def run_demo(
    screenshot_b64: str,
    user_intent: str,
    loop: FaraCheckerLoop,
    top_k_rag: int = 3,
):
    """运行完整工作流，委托 FaraCheckerLoop.run() 处理。"""

    print("\n" + "=" * 65)
    print(f"  [目标] 用户意图: {user_intent}")
    print("=" * 65)

    # ── 主流程（RAG摘要 + Fara推演 + 检索 + 坐标 + Checker 全在 loop.run() 内）──
    print(f"\n[开始] 启动工作流 (Fara纯文本推演 → RAG检索 → Fara坐标 → Checker验证)...\n")
    result, history = loop.run(
        screenshot_b64=screenshot_b64,
        user_intent=user_intent,
        top_k_rag=top_k_rag,
    )

    # ── 输出结果 ──
    print("\n" + "=" * 65)
    print("  [结果] 结果汇总")
    print("=" * 65)

    for entry in history:
        fr = entry["fara"]
        cr = entry.get("checker")

        print(f"\n  第 {entry['attempt']} 轮:")
        if fr.success:
            print(f"    [坐标] Fara 坐标: {fr.coordinate}")
            print(f"    [推理] 推理: {fr.reasoning[:100]}...")
        else:
            print(f"    [失败] Fara 失败: {fr.error}")

        if cr and cr.enabled:
            verdict = "[OK] 正确" if cr.verified else ("[失败] 错误" if cr.verified is False else "[未知] 无法判断")
            print(f"    [检查] Checker: {verdict}")
            if cr.reason:
                print(f"    [推理] {cr.reason[:100]}")
            if cr.error:
                print(f"    [警告]  {cr.error}")

    # 最终坐标
    if result.success and result.is_valid:
        print(f"\n  [OK] 最终坐标: {result.coordinate}")
    else:
        print(f"\n  [失败] 未能获取有效坐标")

    return result, history


def run_rag_only(
    user_intent: str,
    rag: RAGRetriever,
    top_k_rag: int = 3,
):
    """仅运行 RAG 检索部分（不需要 Fara/Checker 模型），展示命中切片。"""

    print("\n" + "=" * 65)
    print(f"  [检索] RAG-Only 模式: {user_intent}")
    print("=" * 65)

    # ── 摘要 ──
    summary = rag.get_summary()
    print(f"\n[摘要] 说明书摘要 ({len(summary)} 字):")
    print(summary)
    print(f"{'─'*50}")

    # ── BM25 粗召回 ──
    print("\n[检索] BM25 粗召回 (Top-8):")
    import jieba
    tokens_raw = list(jieba.cut(user_intent))
    bm25_scores = rag._index._bm25.get_scores(tokens_raw)
    ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    for i, s in ranked[:8]:
        c = rag._index.get_chunk(i)
        print(f"  [{s:>8.4f}] {c.chunk_id:<10} {c.step_title[:35]}")

    # ── FAISS 精排 ──
    print(f"\n[目标] FAISS 向量精排 → BM25候选重排 (Top-{top_k_rag}):")
    results = rag.query(user_intent, top_k=top_k_rag)
    all_ids = set()
    for r in results:
        all_ids.add(r["chunk_id"])
        for n in r.get("neighbors", []):
            all_ids.add(n["chunk_id"])
    ordered = sorted(all_ids, key=lambda x: rag._index._id_to_idx.get(x, 999))

    for i, r in enumerate(results, 1):
        n_ids = [n["chunk_id"] for n in r.get("neighbors", [])]
        print(f"  [{i}] {r['chunk_id']} (score={r['score']}) 邻块={n_ids}")
        print(f"      {r['step']}")
        print(f"      {r['text'][:100]}...")
        if r.get("context_text"):
            print(f"      上下文: {r['context_text'][:100]}...")
        print()

    # ── 最终覆盖 ──
    print(f"[覆盖] 最终覆盖 ({len(all_ids)} 块): {ordered}")
    print()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fara Tutorial Guide RAG Demo")
    parser.add_argument("--screenshot", help="截图文件路径")
    parser.add_argument("--intent", required=True, help="用户意图")
    parser.add_argument("--serve-url", default=DEFAULTS["serve_url"])
    parser.add_argument("--no-checker", action="store_true")
    parser.add_argument("--rag-only", action="store_true",
                        help="仅运行 RAG 检索，不调用 Fara/Checker")
    parser.add_argument("--fara-api", default=DEFAULTS["fara_api"])
    parser.add_argument("--fara-model", default=DEFAULTS["fara_model"])
    parser.add_argument("--checker-api", default=DEFAULTS["checker_api"])
    parser.add_argument("--checker-model", default=DEFAULTS["checker_model"])
    parser.add_argument("--pdf", default=DEFAULTS["pdf_path"])
    parser.add_argument("--models-cache", default=DEFAULTS["models_cache"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["rag_top_k"])
    parser.add_argument("--max-retries", type=int, default=DEFAULTS["max_retries"])
    parser.add_argument("--save-screenshot", help="实时截图保存路径")
    args = parser.parse_args()

    # ── 初始化 RAG（两种模式都需要）──
    rag = RAGRetriever(RAGConfig(
        pdf_path=args.pdf,
        models_cache_dir=args.models_cache,
        context_window=1, device="cpu",
    ))

    # ── RAG-Only 模式 ──
    if args.rag_only:
        run_rag_only(
            user_intent=args.intent,
            rag=rag,
            top_k_rag=args.top_k,
        )
        return

    # ── 完整模式：需要截图 ──
    if args.screenshot:
        screenshot_b64 = load_screenshot(args.screenshot)
        logger.info("截图来源: %s", args.screenshot)
    else:
        screenshot_b64 = capture_screenshot(args.serve_url, args.save_screenshot)
        logger.info("实时截图: %s", args.serve_url)

    actor = FaraActor(FaraConfig(
        api_url=args.fara_api,
        model_path=args.fara_model,
    ))

    checker = Checker(CheckerConfig(
        api_url="" if args.no_checker else args.checker_api,
        model_path=args.checker_model,
        max_retries=args.max_retries,
    ))

    loop = FaraCheckerLoop(rag, actor, checker, max_retries=args.max_retries)

    run_demo(
        screenshot_b64=screenshot_b64,
        user_intent=args.intent,
        loop=loop,
        top_k_rag=args.top_k,
    )


if __name__ == "__main__":
    main()
