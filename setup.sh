#!/usr/bin/env bash
# ============================================================
# Fara Tutorial Guide RAG — 一键环境准备（仅 Web / 无 GPU 场景）
# 用法:  bash setup.sh
# 功能:  ① 创建 venv（已存在则跳过）
#        ② 安装轻量依赖 requirements-web.txt
#        ③ 生成 .env（已存在则跳过；缺失时从 .env.example 复制）
# 之后启动:  source venv/bin/activate && python start_all.py --no-models
# ============================================================
set -e
cd "$(dirname "$0")"

# ① venv
if [ ! -d venv ]; then
  echo "[1/3] 创建 Python venv ..."
  python3 -m venv venv
else
  echo "[1/3] venv 已存在，跳过"
fi
# shellcheck disable=SC1091
source venv/bin/activate

# ② 轻量依赖
echo "[2/3] 安装轻量依赖 (requirements-web.txt) ..."
pip install -q -r requirements-web.txt

# ③ .env
if [ ! -f .env ]; then
  echo "[3/3] 生成 .env（从 .env.example 复制，请按需修改外部 IP）..."
  cp .env.example .env
else
  echo "[3/3] .env 已存在，跳过"
fi

echo ""
echo "✅ 环境就绪。启动命令:"
echo "   source venv/bin/activate && python start_all.py --no-models"
echo "   （.env 中 EMBEDDING_BACKEND=ollama 时无需额外安装；若用本地嵌入需 pip install sentence-transformers）"
