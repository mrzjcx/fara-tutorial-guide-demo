"""
Fara Actor 模块：调用 Fara1.5 视觉模型，结合 RAG 切片给出可执行坐标。

设计原则：
- 配置通过 FaraConfig 传入
- 支持单点坐标和多候选坐标解析
- 自动处理 API 异常和超时
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.config as pc

logger = logging.getLogger(__name__)


@dataclass
class FaraConfig:
    """Fara Actor 配置"""

    # vLLM API 端点（统一来自 scripts/config.py，环境变量 FARA_API_URL 可覆盖）
    api_url: str = pc.FARA_API_URL

    # 模型本地路径（统一来自 scripts/config.py，环境变量 FARA_MODEL_PATH 可覆盖）
    model_path: str = pc.FARA_MODEL_PATH

    # 请求 model 字段（默认=model_path；指向外部 vLLM/Ollama 时用 FARA_MODEL_NAME）
    model_name: str = pc.FARA_MODEL_NAME

    # 生成参数
    max_tokens: int = 300
    temperature: float = 0.0
    timeout: int = 120

    # 是否启用 thinking 模式（Qwen 系需要显式禁用）
    enable_thinking: bool = False

    # 停止词（在首个完整 </tool_call> 后截断）
    stop_tokens: List[str] = field(default_factory=lambda: ["</tool_call>"])

    # System Prompt（可选覆盖，为空则用内置默认）
    system_prompt: str = ""

    # 是否提示模型可以输出多候选坐标
    allow_multi_candidates: bool = True

    # RAG 查询规划参数（plan_rag_query 纯文本推演，不带截图）
    plan_rag_query_max_tokens: int = 80
    plan_rag_query_timeout: int = 60
    plan_rag_query_prompt: str = ""  # 为空则用内置默认
