#!/usr/bin/env python3
"""
Fara RAG Demo 一键启动 —— 唯一入口。

实际逻辑在 scripts/launcher.py（本项目所有脚本统一放在 scripts/ 目录）。

用法:
    python start_all.py                 # 一键启动全部（模型 + Web）
    python start_all.py --no-checker    # 只启动 Fara 模型
    python start_all.py --no-web        # 只启动模型，不起 Web
    python start_all.py --stop          # 停止全部（模型 + Web）
    python start_all.py --status        # 查看状态

访问（端口见 scripts/config.py 的 WEB_DEMO_PORT / SERVE_PORT）:
    Web Demo (主界面):  http://<服务器IP>:<WEB_DEMO_PORT>
    前端 HTML (截图源):  http://<服务器IP>:<SERVE_PORT>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from launcher import main

if __name__ == "__main__":
    main()
