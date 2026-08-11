#!/usr/bin/env python3
"""
一键启动 / 停止 / 检查 Fara + Checker 两个 vLLM 模型服务。

配置来源: config.py（所有参数可用环境变量覆盖，见该文件头部注释）

用法:
    python model_server.py                  # 启动 Fara + Checker 并等待就绪
    python model_server.py --no-checker     # 只启动 Fara
    python model_server.py --stop           # 停止全部 vLLM 服务
    python model_server.py --status         # 查看服务状态

与 demo.py / web_demo.py 的分工:
    - model_server.py  启动模型服务（vLLM 后台进程，长期运行）
    - demo.py          调用已运行的服务执行单次端到端流程（RAG+Fara+Checker）
    - web_demo.py      调用已运行的服务启动 Web 界面
    推荐顺序: 先 `python model_server.py`，再 `python demo.py ...` / `python web_demo.py ...`
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根（scripts.config 包导入）
import scripts.config as pc

LOG_DIR = Path(pc.LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def resolve_vllm_bin() -> str:
    """定位 vllm 可执行文件：VLLM_BIN 环境变量 → PATH → 常见 venv 位置。"""
    if pc.VLLM_BIN and Path(pc.VLLM_BIN).exists():
        return pc.VLLM_BIN
    found = shutil.which("vllm")
    if found:
        return found
    for cand in (
        Path.home() / "workspace" / "fara15-venv" / "bin" / "vllm",
        Path(pc.PROJECT_ROOT).parent / "fara15-venv" / "bin" / "vllm",
    ):
        if cand.exists():
            return str(cand)
    print("[失败] 未找到 vllm 可执行文件，请通过 VLLM_BIN 环境变量指定")
    sys.exit(1)


def vllm_command(model_path, port, mem_util, max_len, enforce_eager, bf16, mm_image_limit=""):
    """构造 vllm serve 命令（served-model-name 与模型路径严格一致，含尾部 /）。"""
    cmd = [
        resolve_vllm_bin(), "serve", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--gpu-memory-utilization", str(mem_util),
        "--max-model-len", str(max_len),
        "--trust-remote-code",
        "--served-model-name", model_path,
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if bf16:
        cmd += ["--dtype", "bfloat16"]
    if mm_image_limit:
        cmd += ["--limit-mm-per-prompt", f'{{"image": {mm_image_limit}}}']
    return cmd


def is_running(port: int) -> bool:
    """端口健康检查是否 200。"""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_service(name, model_path, port, gpu, mem_util, max_len, enforce_eager, bf16, mm_image_limit=""):
    """后台启动单个 vLLM 服务，日志写入 LOG_DIR/{name}.log。"""
    log = LOG_DIR / f"{name}.log"
    cmd = vllm_command(model_path, port, mem_util, max_len, enforce_eager, bf16, mm_image_limit)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    print(f"[启动] {name}: PID={proc.pid} | 端口={port} | GPU={gpu} | 显存={mem_util} | "
          f"日志={log}\n  模型: {model_path}")
    return proc


def wait_health(port, name, timeout=300):
    """轮询等待服务健康检查通过。"""
    start = time.time()
    while time.time() - start < timeout:
        if is_running(port):
            print(f"[OK] {name} 就绪 ({port})")
            return True
        time.sleep(5)
    print(f"[失败] {name} 未在 {timeout}s 内就绪，请查看日志: {LOG_DIR}/{name}.log")
    return False


def stop_all():
    """停止全部 vLLM 服务（按端口精确匹配 + 兜底 pkill）。"""
    for port in (pc.FARA_PORT, pc.CHECKER_PORT):
        # 加 "--" 分隔符：避免 pattern 以 "--" 开头（如 "--port 5002"）被 pkill 误解析为选项
        subprocess.run(["pkill", "-f", "--", f"--port {port}"], check=False)
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    print("[停止] 已发送停止信号")


def status():
    """打印两个服务的运行状态。"""
    for name, port, model in (
        ("Fara", pc.FARA_PORT, pc.FARA_MODEL_PATH),
        ("Checker", pc.CHECKER_PORT, pc.CHECKER_MODEL_PATH),
    ):
        ok = is_running(port)
        print(f"{name:<8} :{port:<6} {'[OK] 运行中' if ok else '[失败] 未运行'}   模型={model}")


def main():
    parser = argparse.ArgumentParser(description="一键启动 Fara + Checker vLLM 服务")
    parser.add_argument("--stop", action="store_true", help="停止全部 vLLM 服务")
    parser.add_argument("--status", action="store_true", help="查看服务状态")
    parser.add_argument("--no-checker", action="store_true", help="只启动 Fara，不启动 Checker")
    parser.add_argument("--timeout", type=int, default=300, help="等待就绪超时（秒）")
    args = parser.parse_args()

    if args.stop:
        stop_all()
        return
    if args.status:
        status()
        return

    # ---- 启动 Fara ----
    started = []
    if is_running(pc.FARA_PORT):
        print(f"ℹ Fara 已在运行 ({pc.FARA_PORT})，跳过启动")
    else:
        started.append(start_service(
            "fara", pc.FARA_MODEL_PATH, pc.FARA_PORT, pc.FARA_GPU,
            pc.FARA_GPU_MEM_UTIL, pc.FARA_MAX_MODEL_LEN,
            pc.FARA_ENFORCE_EAGER, pc.FARA_USE_BF16))

    # ---- 启动 Checker ----
    if args.no_checker:
        print("ℹ --no-checker，跳过 Checker")
    elif is_running(pc.CHECKER_PORT):
        print(f"ℹ Checker 已在运行 ({pc.CHECKER_PORT})，跳过启动")
    else:
        started.append(start_service(
            "checker", pc.CHECKER_MODEL_PATH, pc.CHECKER_PORT, pc.CHECKER_GPU,
            pc.CHECKER_GPU_MEM_UTIL, pc.CHECKER_MAX_MODEL_LEN,
            pc.CHECKER_ENFORCE_EAGER, pc.CHECKER_USE_BF16,
            mm_image_limit=pc.CHECKER_MM_IMAGES))

    if not started:
        print("两个服务均已运行，无需启动。")
        return

    # ---- 等待就绪 ----
    ok = wait_health(pc.FARA_PORT, "Fara", args.timeout)
    if not args.no_checker:
        ok &= wait_health(pc.CHECKER_PORT, "Checker", args.timeout)

    print()
    if ok:
        print("[OK] 全部服务就绪。接下来可运行:")
        print(f"   python demo.py --screenshot docs/img/02_register_dept.png --intent \"在线挂号选骨科\"")
        print(f"   python web_demo.py --port {pc.WEB_DEMO_PORT}")
    else:
        print("[警告] 部分服务未就绪，请查看日志目录:", LOG_DIR)


if __name__ == "__main__":
    main()
