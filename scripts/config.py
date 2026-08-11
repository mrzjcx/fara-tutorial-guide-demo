"""
项目集中配置文件

设计原则:
- 所有路径与 URL 集中在此定义，其他模块一律从此处读取
- 环境变量优先，其次使用自动推导的默认值
- 默认值基于项目根目录推导，不依赖固定的 /root/workspace 布局
- 完全向后兼容：不设置任何环境变量时，默认行为与旧硬编码一致
  （root 用户下 ~/workspace/models == /root/workspace/models）

可用环境变量:
- FARA_MODELS_CACHE / MODELS_CACHE : 模型缓存根目录
- FARA_MODEL_PATH / CHECKER_MODEL_PATH : 模型路径（含尾部 /）
- FARA_PORT / CHECKER_PORT         : 模型服务端口（默认 5002 / 5003）
- FARA_API_URL / CHECKER_API_URL   : vLLM OpenAI 兼容端点（默认由端口构造）
- SERVE_PORT / WEB_DEMO_PORT       : 前端静态页 / Web Demo 端口（默认 8080 / 8090）
- SERVE_URL                        : 前端静态页地址（默认由端口构造）
- SCREENSHOT_DIR                   : 截图输出目录
- PDF_PATH                         : 说明书 PDF 路径
- EMBEDDING_MODEL_ID / EMBEDDING_DIM : BGE 嵌入模型 ID / 维度（默认 512）
- DEFAULT_INTENT                   : Web 界面默认用户意图
- FARA_SPACE                       : Fara 归一化坐标空间（默认 1000）
- RAG_TOP_K / MAX_RETRIES          : RAG 返回切片数 / Checker 最大重试

以下为 vLLM 服务启动参数（model_server.py 使用）:
- FARA_GPU / CHECKER_GPU           : CUDA_VISIBLE_DEVICES（默认 0 / 1）
- FARA_GPU_MEM_UTIL / CHECKER_GPU_MEM_UTIL : 显存利用率（默认 0.85 / 0.30）
- FARA_MAX_MODEL_LEN / CHECKER_MAX_MODEL_LEN : 上下文长度（默认 4096 / 8192）
- FARA_ENFORCE_EAGER / CHECKER_ENFORCE_EAGER : 禁用 CUDA Graph（默认 1）
- FARA_USE_BF16 / CHECKER_USE_BF16 : 是否 --dtype bfloat16（默认 1 / 0）
- VLLM_BIN                         : vllm 可执行文件路径（默认自动查找）
- LOG_DIR                          : 服务日志目录（默认 ./logs）
"""
import os
from pathlib import Path

# 项目根目录（scripts/ 的上级目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_or(default: str, *keys: str) -> str:
    """按顺序检查环境变量，全部未设置时返回默认值。"""
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return default


def _env_bool(key: str, default: bool) -> bool:
    """标准布尔环境变量解析：1/true/yes/on（忽略大小写）为真，其余为假；未设置用默认值。"""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---- 模型缓存目录 ----
# 默认 ~/workspace/models
MODELS_CACHE = _env_or(str(Path.home() / "workspace" / "models"),
                       "FARA_MODELS_CACHE", "MODELS_CACHE")

# ---- 模型本地路径（也是 vLLM --served-model-name，含尾部 /）----
# 用 Path().as_posix() 统一为正斜杠，Windows / Linux 兼容（vLLM 的 model 标识统一用 /）


def _model_dir(model_name: str) -> str:
    return (Path(MODELS_CACHE) / model_name).as_posix() + "/"


FARA_MODEL_PATH = _env_or(_model_dir("microsoft--Fara1.5-4B"), "FARA_MODEL_PATH")
CHECKER_MODEL_PATH = _env_or(_model_dir("Qwen--Qwen3.5-0.8B"), "CHECKER_MODEL_PATH")

# ============================================================
# 端口（API URL 与启动参数共用，避免重复维护）
# ============================================================

# ---- vLLM 模型服务端口 ----
FARA_PORT = _env_or("5002", "FARA_PORT")
CHECKER_PORT = _env_or("5003", "CHECKER_PORT")

# ---- Web 服务端口 ----
SERVE_PORT = _env_or("8080", "SERVE_PORT")        # 前端静态页（HTML 截图源）
WEB_DEMO_PORT = _env_or("8090", "WEB_DEMO_PORT")  # Web 主界面


# ============================================================
# vLLM OpenAI 兼容端点（端口与上方一致，改端口只需改 PORT）
# ============================================================
FARA_API_URL = _env_or(f"http://localhost:{FARA_PORT}/v1/chat/completions", "FARA_API_URL")
CHECKER_API_URL = _env_or(f"http://localhost:{CHECKER_PORT}/v1/chat/completions", "CHECKER_API_URL")


# ---- 前端静态页 / 截图 ----
SERVE_URL = _env_or(f"http://localhost:{SERVE_PORT}", "SERVE_URL")
SCREENSHOT_DIR = _env_or(str(PROJECT_ROOT / "docs" / "img"), "SCREENSHOT_DIR")

# ---- RAG ----
PDF_PATH = _env_or(str(PROJECT_ROOT / "docs" / "manual" / "manual.pdf"), "PDF_PATH")
EMBEDDING_MODEL_ID = _env_or("BAAI/bge-small-zh-v1.5", "EMBEDDING_MODEL_ID")
EMBEDDING_DIM = int(_env_or("512", "EMBEDDING_DIM"))  # bge-small-zh-v1.5 维度

# ---- Web 默认业务值 ----
DEFAULT_INTENT = _env_or("14号挂骨科 选吴医生 早上九点到九点半的号", "DEFAULT_INTENT")
FARA_SPACE = int(_env_or("1000", "FARA_SPACE"))  # Fara 归一化坐标空间（红圈标注换算用）

# ---- Demo 默认参数 ----
RAG_TOP_K = int(_env_or("3", "RAG_TOP_K"))
MAX_RETRIES = int(_env_or("3", "MAX_RETRIES"))


# ============================================================
# vLLM 服务启动参数（model_server.py 使用）
# ============================================================

# ---- GPU 自动分配 ----
# >=2 张卡: Fara→0, Checker→1；只有 1 张卡: 都放 0。环境变量可覆盖。


def _detect_gpu_assignment() -> "tuple[str, str]":
    """自动检测 GPU 数量并分配 (fara_gpu, checker_gpu)
    """
    count = 0
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5
        )
        count = int(out.strip())
    except Exception:
        try:
            import torch
            count = torch.cuda.device_count()
        except Exception:
            count = 0
    if count >= 2:
        return "0", "1"
    return "0", "0"


_FARA_GPU_DEFAULT, _CHECKER_GPU_DEFAULT = _detect_gpu_assignment()

# ---- Fara（Actor）----
FARA_GPU = _env_or(_FARA_GPU_DEFAULT, "FARA_GPU")        # CUDA_VISIBLE_DEVICES（自动分配）
FARA_GPU_MEM_UTIL = _env_or("0.85", "FARA_GPU_MEM_UTIL")
FARA_MAX_MODEL_LEN = _env_or("4096", "FARA_MAX_MODEL_LEN")
FARA_ENFORCE_EAGER = _env_bool("FARA_ENFORCE_EAGER", True)   # 禁用 CUDA Graph
FARA_USE_BF16 = _env_bool("FARA_USE_BF16", True)             # --dtype bfloat16

# ---- Checker ----
CHECKER_GPU = _env_or(_CHECKER_GPU_DEFAULT, "CHECKER_GPU")
CHECKER_GPU_MEM_UTIL = _env_or("0.30", "CHECKER_GPU_MEM_UTIL")
CHECKER_MAX_MODEL_LEN = _env_or("8192", "CHECKER_MAX_MODEL_LEN")  # 容纳原图+放大图两张图 token
CHECKER_ENFORCE_EAGER = _env_bool("CHECKER_ENFORCE_EAGER", True)   # 禁用 CUDA Graph
CHECKER_USE_BF16 = _env_bool("CHECKER_USE_BF16", False)           # --dtype bfloat16
CHECKER_MM_IMAGES = _env_or("2", "CHECKER_MM_IMAGES")             # 每请求最多图片数（原图+放大=2）

# ---- 通用 ----
VLLM_BIN = _env_or("", "VLLM_BIN")                       # vllm 可执行文件，空则自动查找
LOG_DIR = _env_or(str(PROJECT_ROOT / "logs"), "LOG_DIR")  # 服务日志目录


def _resolve_python_bin() -> str:
    """定位 venv 的 python 解释器（与 vllm 同一 venv），供 Web 等子进程使用。"""
    # 1) 与 vllm 同目录（venv/bin/vllm → venv/bin/python）
    if VLLM_BIN and Path(VLLM_BIN).exists():
        cand = Path(VLLM_BIN).with_name("python")
        if cand.exists():
            return str(cand)
    # 2) 常见 venv 位置
    for cand in (
        Path.home() / "workspace" / "fara15-venv" / "bin" / "python",
        PROJECT_ROOT.parent / "fara15-venv" / "bin" / "python",
    ):
        if cand.exists():
            return str(cand)
    # 3) 当前解释器
    import sys
    return sys.executable


# venv python 解释器（环境变量 PYTHON_BIN 可覆盖）；启动 Web 等服务时使用，
# 避免用系统 python3（可能缺少 flask 等依赖）
PYTHON_BIN = _env_or(_resolve_python_bin(), "PYTHON_BIN")
