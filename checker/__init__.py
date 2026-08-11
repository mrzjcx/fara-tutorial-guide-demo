"""
Checker 模块入口。
"""
from .config import CheckerConfig
from .checker import Checker, CheckerResult

__all__ = ["CheckerConfig", "Checker", "CheckerResult"]
