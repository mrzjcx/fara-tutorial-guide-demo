#!/usr/bin/env python3
"""
一键停止全部服务：两个 vLLM 模型（Fara / Checker）+ Web（static_server / web_demo）。

端口/路径统一从 scripts/config.py 读取，修改端口无需改本脚本。

用法:
    python scripts/stopper.py            # 停止全部（模型 + Web）
    python scripts/stopper.py --models   # 只停止两个模型（保留 Web）
    python scripts/stopper.py --web      # 只停止 Web（保留模型）
    python scripts/stopper.py --status   # 查看状态（不停止）

停止后可随时用 `python start_all.py` 重新一键启动。
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.config as pc


def _pkill(pattern: str):
    # 加 "--" 分隔符：避免 pattern 以 "--" 开头（如 "--port 5002"）被 pkill 误解析为选项
    subprocess.run(["pkill", "-f", "--", pattern], check=False)


def stop_models():
    """停止 Fara + Checker 两个 vLLM 服务。"""
    for port in (pc.FARA_PORT, pc.CHECKER_PORT):
        _pkill(f"--port {port}")          # 按端口精确匹配
    _pkill("vllm serve")                  # 兜底：全部 vLLM 进程
    print(f"[停止] 已向 Fara(:{pc.FARA_PORT}) / Checker(:{pc.CHECKER_PORT}) 发送停止信号")


def stop_web():
    """停止 Web Demo + 前端静态页。"""
    _pkill("web_demo.py")
    _pkill("static_server.py")
    print(f"[停止] 已向 web_demo(:{pc.WEB_DEMO_PORT}) / static_server(:{pc.SERVE_PORT}) 发送停止信号")


def _is_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def status():
    """打印各服务状态（不停止）。"""
    for name, port in (("Fara", pc.FARA_PORT),
                       ("Checker", pc.CHECKER_PORT),
                       ("前端静态页", pc.SERVE_PORT),
                       ("Web Demo", pc.WEB_DEMO_PORT)):
        ok = _is_running(port)
        print(f"{name:<10} :{port:<6} {'运行中' if ok else '已停止'}")


def main():
    parser = argparse.ArgumentParser(description="一键停止 vLLM 模型 + Web 服务")
    parser.add_argument("--models", action="store_true", help="只停止两个模型")
    parser.add_argument("--web", action="store_true", help="只停止 Web 服务")
    parser.add_argument("--status", action="store_true", help="查看状态（不停止）")
    args = parser.parse_args()

    if args.status:
        status()
        return
    if args.models:
        stop_models()
    elif args.web:
        stop_web()
    else:
        stop_models()
        stop_web()
    print("停止完成，可执行 `python scripts/stopper.py --status` 确认。")


if __name__ == "__main__":
    main()
