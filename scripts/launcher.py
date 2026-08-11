#!/usr/bin/env python3
"""
Fara RAG Demo —— 一键启动全部（模型服务 + Web 界面 + HTML 前端）。

启动顺序:
  1) Fara vLLM     (端口 FARA_PORT, GPU0) —— 坐标推理
  2) Checker vLLM  (端口 CHECKER_PORT, GPU1) —— 坐标验证
  3) static_server.py (端口 SERVE_PORT) —— 前端 HTML 静态页（iframe 截图源）
  4) web_demo.py   (端口 WEB_DEMO_PORT) —— Web 主界面（RAG 检索/全流程/红圈标注）

用法:
    python start_all.py                 # 一键启动全部（模型 + Web）
    python start_all.py --no-checker    # 只启动 Fara 模型
    python start_all.py --no-web        # 只启动模型，不起 Web
    python start_all.py --stop          # 停止全部（模型 + Web）
    python start_all.py --status        # 查看状态

环境要求:
    - Python 3.12 venv，已安装 requirements.txt + requirements-vllm.txt
    - 模型已下载到 config.py 的 MODELS_CACHE 目录（默认 ~/workspace/models）
    - NVIDIA GPU（默认 Fara→GPU0, Checker→GPU1，可在 config.py 调整）

访问地址（端口见 config.py 的 WEB_DEMO_PORT / SERVE_PORT；把 <服务器IP> 换成实际 IP）:
    - Web Demo (主界面):  http://<服务器IP>:<WEB_DEMO_PORT>
    - 前端 HTML (截图源):  http://<服务器IP>:<SERVE_PORT>
"""
import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根（scripts.config 包导入）
import scripts.config as pc
from model_server import is_running, wait_health, start_service, stop_all

LOG_DIR = Path(pc.LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 端口统一从 config.py 读取：pc.WEB_DEMO_PORT / pc.SERVE_PORT


def is_port_open(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_web():
    """启动前端静态页 + Web Demo（端口由 config.py 管理），幂等。"""
    python = pc.PYTHON_BIN  # 用 venv python（系统 python3 可能缺少 flask 等依赖）
    scripts_dir = Path(__file__).resolve().parent
    if is_port_open(pc.SERVE_PORT):
        print(f"ℹ 前端静态页已在运行 (:{pc.SERVE_PORT})，跳过")
    else:
        log = open(LOG_DIR / "serve.log", "w")
        subprocess.Popen([python, str(scripts_dir / "static_server.py"), str(pc.SERVE_PORT)], cwd=str(pc.PROJECT_ROOT),
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[启动] 前端静态页: :{pc.SERVE_PORT} (日志 {LOG_DIR}/serve.log)")

    if is_port_open(pc.WEB_DEMO_PORT):
        print(f"ℹ Web Demo 已在运行 (:{pc.WEB_DEMO_PORT})，跳过")
    else:
        log = open(LOG_DIR / "web_demo.log", "w")
        subprocess.Popen([python, str(scripts_dir / "web_demo.py"), "--port", str(pc.WEB_DEMO_PORT)],
                         cwd=str(pc.PROJECT_ROOT),
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[启动] Web Demo: :{pc.WEB_DEMO_PORT} (日志 {LOG_DIR}/web_demo.log)")


def stop_web():
    for pat in ("web_demo.py", "static_server.py"):
        subprocess.run(["pkill", "-f", pat], check=False)
    print("[停止] 已停止 Web 服务")


def status():
    status_fn = __import__("model_server").status
    status_fn()
    for name, port in (("前端静态页", pc.SERVE_PORT), ("Web Demo", pc.WEB_DEMO_PORT)):
        print(f"{name:<10} :{port:<6} {'[OK] 运行中' if is_port_open(port) else '[失败] 未运行'}")


def main():
    parser = argparse.ArgumentParser(description="Fara RAG Demo 一键启动（模型 + Web）")
    parser.add_argument("--no-checker", action="store_true", help="不启动 Checker")
    parser.add_argument("--no-web", action="store_true", help="只启动模型服务，不起 Web")
    parser.add_argument("--stop", action="store_true", help="停止全部（模型 + Web）")
    parser.add_argument("--status", action="store_true", help="查看全部服务状态")
    parser.add_argument("--timeout", type=int, default=300, help="模型就绪等待超时（秒）")
    args = parser.parse_args()

    if args.stop:
        stop_all()
        stop_web()
        return
    if args.status:
        status()
        return

    # ---- 1) 模型服务 ----
    if is_running(pc.FARA_PORT):
        print(f"ℹ Fara 已在运行 (:{pc.FARA_PORT})，跳过")
    else:
        start_service("fara", pc.FARA_MODEL_PATH, pc.FARA_PORT, pc.FARA_GPU,
                      pc.FARA_GPU_MEM_UTIL, pc.FARA_MAX_MODEL_LEN,
                      pc.FARA_ENFORCE_EAGER, pc.FARA_USE_BF16)

    if not args.no_checker:
        if is_running(pc.CHECKER_PORT):
            print(f"ℹ Checker 已在运行 (:{pc.CHECKER_PORT})，跳过")
        else:
            start_service("checker", pc.CHECKER_MODEL_PATH, pc.CHECKER_PORT, pc.CHECKER_GPU,
                          pc.CHECKER_GPU_MEM_UTIL, pc.CHECKER_MAX_MODEL_LEN,
                          pc.CHECKER_ENFORCE_EAGER, pc.CHECKER_USE_BF16)

    ok = wait_health(pc.FARA_PORT, "Fara", args.timeout)
    if not args.no_checker:
        ok &= wait_health(pc.CHECKER_PORT, "Checker", args.timeout)

    # ---- 2) Web 服务（模型就绪后再启动）----
    if not args.no_web:
        start_web()
        for name, port in (("前端静态页", pc.SERVE_PORT), ("Web Demo", pc.WEB_DEMO_PORT)):
            deadline = time.time() + 90
            while time.time() < deadline:
                if is_port_open(port):
                    print(f"[OK] {name} 就绪 (:{port})")
                    break
                time.sleep(2)
            else:
                print(f"[失败] {name} 未在 90s 内就绪，请查看 {LOG_DIR}")

    print()
    print("一键启动完成！访问地址（端口见 config.py，把 IP 换成实际地址或经 SSH 隧道）:")
    print(f"   Web Demo (主界面): http://<服务器IP>:{pc.WEB_DEMO_PORT}")
    print(f"   前端 HTML (截图源): http://<服务器IP>:{pc.SERVE_PORT}")
    print(f"   命令行 demo: python demo.py --screenshot docs/img/02_register_dept.png --intent \"在线挂号选骨科\"")


if __name__ == "__main__":
    main()
