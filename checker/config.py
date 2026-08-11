"""
Checker 模块：调用 Qwen Checker 模型验证 Fara 坐标，支持反馈迭代。
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.config as pc

logger = logging.getLogger(__name__)


@dataclass
class CheckerConfig:
    """Checker 配置 —— 无硬编码。"""

    # vLLM API 端点（统一来自 scripts/config.py，环境变量 CHECKER_API_URL 可覆盖）
    api_url: str = pc.CHECKER_API_URL

    # 模型本地路径（统一来自 scripts/config.py，环境变量 CHECKER_MODEL_PATH 可覆盖）
    model_path: str = pc.CHECKER_MODEL_PATH

    # 生成参数
    max_tokens: int = 150
    temperature: float = 0.0
    timeout: int = 60

    # 是否启用 thinking 模式
    enable_thinking: bool = False

    stop_tokens: List[str] = field(default_factory=lambda: ["</tool_call>"])

    # System Prompt（为空则用内置默认）
    system_prompt: str = ""

    # 反馈迭代参数
    max_retries: int = 3   # 最大重试次数（Fara→Checker→Fara 循环）
