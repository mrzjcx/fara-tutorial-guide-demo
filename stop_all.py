#!/usr/bin/env python3
"""
Fara RAG Demo 一键停止 —— 根目录入口（与 start_all.py 对称）。

实际逻辑在 scripts/stopper.py（本项目所有脚本统一放在 scripts/ 目录）。

用法:
    python stop_all.py            # 停止全部（模型 + Web）
    python stop_all.py --models   # 只停止两个模型（保留 Web）
    python stop_all.py --web      # 只停止 Web（保留模型）
    python stop_all.py --status   # 查看状态（不停止）

停止后可随时用 `python start_all.py` 一键重新启动。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from stopper import main

if __name__ == "__main__":
    main()
