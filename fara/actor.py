"""
Fara Actor：看图 + RAG 切片 + 用户意图 → 输出点击坐标。
"""
import re
import json
import time
import requests
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from .config import FaraConfig

logger = logging.getLogger(__name__)


# ============================================================
# 默认 System Prompt
# ============================================================

DEFAULT_FARA_SYSTEM_PROMPT = """You are a GUI tutorial assistant. You help users complete multi-step operations by looking at their screenshots and providing guidance.

Your ONLY job: analyze the screenshot against the tutorial steps and output ONE suggested click position.

Output format (STRICT):
<tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}</tool_call>

Rules:
- Compare the screenshot with the tutorial to determine which step the user is currently on.
- Within a step, suggest the NEXT SINGLE micro-action (one click, one field to fill), not the step's final action.
- If a step requires filling multiple fields → suggest the FIRST empty field, not the submit button.
- If a step is already completed, move to the next step.
- How to tell a step is completed: the target item is highlighted/selected — e.g., its border is highlighted (turns blue or another highlight color), or it appears selected/checked.
- When the current step's required items are ALL already selected/highlighted, the correct next action is clicking the visible "下一步 →" button — even if the NEXT step's content (e.g., doctor list) is not yet visible. Do NOT output [-1, -1] in this case.
- Output [-1, -1] ONLY when the element needed for the CURRENT step is not visible in the screenshot. A next-screen's content being hidden is NOT a reason to stop — click the visible 下一步/confirm button instead.
- x is horizontal (left to right), y is vertical (top to bottom).
- Coordinates are in 1000x1000 normalized space (top-left=0,0; bottom-right=1000,1000).
- If you cannot determine the next action, output coordinate: [-1, -1].
- Output EXACTLY one line of GUIDANCE in Chinese that directly instructs the user what to do (second person, actionable), e.g. "请点击左侧第二张'骨科'卡片". Then output the tool_call.
- The guidance line is shown to the user as the instruction — write it for a human, not as internal reasoning.
- Do NOT suggest submitting forms or irreversible actions unless the user's instructions explicitly allow it.

FORM-FILLING GUIDANCE (Any step with input fields):
- When the current step contains text input fields, check each one CAREFULLY from top to bottom.
 - [WARNING] CRITICAL: If a field already has text inside → SKIP IT. Move to the next field. Do NOT output its coordinate.
- Only output the coordinate of a field that is COMPLETELY EMPTY (no visible characters at all).
- Do NOT output coordinates for fields that are already filled — the user does not need to click them.
- Decision flow:
  1. Scan all input fields from top to bottom.
  2. Field has text? → SKIP. Go to next field.
  3. Field is completely blank? → OUTPUT its coordinate. STOP scanning.
  4. ALL fields have text? → OUTPUT the "下一步" or "确认" button.
- Do NOT output the submit/confirm button if any field is still blank.

IMPORTANT — Handling Ambiguity:
- If the user's intent is vague (e.g., "选个上午的时间") and the screenshot shows multiple possible targets, DO NOT guess a single coordinate.
- Instead, list ALL matching candidates with their approximate coordinates in this format:
  <tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [[x1,y1], [x2,y2], ...]}}</tool_call>
- For each candidate, briefly describe it in the guidance line so the user can choose (e.g. "多个可选：a) 骨科（左侧第二张卡片） b) 内科（右侧第一张）").
- If the exact target is NOT visible on the current screenshot, output coordinate: [-1, -1] and explain what the user should look for.
"""


@dataclass
class FaraResult:
    """Fara Actor 单次调用结果"""
    success: bool
    coordinate: Union[List[int], List[List[int]]]  # [x,y] 或 [[x1,y1],[x2,y2]]
    reasoning: str                                  # 引导语/推理文本（中文）
    raw_response: str                               # 模型原始输出
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    degraded: bool = False                          # ④ 降级为文字指导（无有效坐标）
    needs_confirmation: bool = False                # ⑤ 涉及不可逆操作，需人工确认

    @property
    def is_valid(self) -> bool:
        """坐标是否有效（非 [-1, -1]）"""
        if not self.success:
            return False
        coord = self.coordinate
        if not coord:
            return False
        # 多候选
        if isinstance(coord[0], list):
            return len(coord) > 0 and any(
                len(c) >= 2 and (c[0] > 0 or c[1] > 0) for c in coord
            )
        # 单点
        return len(coord) >= 2 and (coord[0] > 0 or coord[1] > 0)

    @property
    def single_coord(self) -> Optional[List[int]]:
        """如果是单点坐标则返回，否则返回 None"""
        if self.coordinate and isinstance(self.coordinate[0], int):
            return self.coordinate  # type: ignore
        return None

    @property
    def multi_coords(self) -> List[List[int]]:
        """返回多候选坐标列表"""
        if self.coordinate and isinstance(self.coordinate[0], list):
            return self.coordinate  # type: ignore
        return [self.coordinate] if self.coordinate else []  # type: ignore


class FaraActor:
    """
    Fara Actor：调用 Fara1.5 vLLM 端点，看图 + RAG 切片 → 输出点击坐标。

    Usage:
        # 端点/模型路径统一见 scripts/config.py（此处示例直接引用）
        from scripts.config import FARA_API_URL, FARA_MODEL_PATH
        config = FaraConfig(
            api_url=FARA_API_URL,
            model_path=FARA_MODEL_PATH,
        )
        actor = FaraActor(config)
        result = actor.act(
            screenshot_b64="...",
            user_intent="在线挂号选骨科",
            rag_snippets=[{"step": "2.1", "text": "点击目标科室..."}],
            step_context="尚未开始",
        )
        print(result.coordinate)  # [320, 450]
    """

    def __init__(self, config: FaraConfig):
        self._cfg = config
        self._system_prompt = config.system_prompt or DEFAULT_FARA_SYSTEM_PROMPT
        logger.info(f"FaraActor 初始化: api={config.api_url}, model={config.model_path}")

    # ==================== 核心调用 ====================

    def act(
        self,
        screenshot_b64: str,
        user_intent: str = "",
        rag_snippets: List[Dict] = None,
        step_context: str = "",
        checker_feedback: str = "",
    ) -> FaraResult:
        """
        调用 Fara 模型分析截图并输出点击坐标。

        Args:
            screenshot_b64: 截图 base64 编码（不含 data URI 前缀）
            user_intent: 用户意图描述
            rag_snippets: RAG 检索到的相关说明书切片列表
            step_context: 当前已完成的步骤状态
            checker_feedback: Checker 的反馈意见（重试时传入）

        Returns:
            FaraResult: 包含坐标、推理文本等
        """
        # 构建 user message
        user_content = self._build_messages(
            screenshot_b64, user_intent, rag_snippets,
            step_context, checker_feedback,
        )

        payload = {
            "model": self._cfg.model_path,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "chat_template_kwargs": {"enable_thinking": self._cfg.enable_thinking},
            "stop": self._cfg.stop_tokens,
        }

        try:
            resp = requests.post(self._cfg.api_url, json=payload, timeout=self._cfg.timeout)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            return FaraResult(
                success=False,
                coordinate=[-1, -1],
                reasoning="",
                raw_response="",
                error=f"无法连接 Fara 服务 ({self._cfg.api_url})",
            )
        except requests.exceptions.Timeout:
            return FaraResult(
                success=False,
                coordinate=[-1, -1],
                reasoning="",
                raw_response="",
                error=f"Fara 请求超时 ({self._cfg.timeout}s)",
            )
        except Exception as e:
            return FaraResult(
                success=False,
                coordinate=[-1, -1],
                reasoning="",
                raw_response="",
                error=f"Fara 请求异常: {e}",
            )

        if "choices" not in result:
            return FaraResult(
                success=False,
                coordinate=[-1, -1],
                reasoning="",
                raw_response=json.dumps(result, ensure_ascii=False),
                error=f"Fara API 返回异常: {result}",
            )

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        # 解析坐标 + 推理
        coord = self._parse_coordinate(content)
        reasoning = self._parse_reasoning(content)

        return FaraResult(
            success=True,
            coordinate=coord,
            reasoning=reasoning,
            raw_response=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    # ==================== 内部方法 ====================

    def _build_messages(
        self,
        screenshot_b64: str,
        user_intent: str,
        rag_snippets: List[Dict],
        step_context: str,
        checker_feedback: str,
    ) -> List[Dict]:
        """构建多模态 user message 内容。"""
        # 组装 RAG 切片文本
        if rag_snippets:
            snippets_text = "\n\n".join(
                f"[{s.get('step', '?')}] {s.get('text', '')}"
                for s in rag_snippets
            )
            if not snippets_text.strip():
                snippets_text = "(无相关说明书内容)"
        else:
            snippets_text = "(未提供说明书切片)"

        # P0-2: 结构化输入 — 拆分为独立字段
        text_parts = [
            f"【当前任务】\n{user_intent if user_intent else '未指定'}",
            f"\n【已完成的操作】\n{step_context if step_context else '尚未执行任何操作'}",
            f"\n【说明书参考】\n{snippets_text}",
        ]
        if checker_feedback:
            text_parts.append(
                f"\n【上一轮验证反馈】\n{checker_feedback}\n请根据此反馈修正坐标输出。"
            )

        user_text = "\n".join(text_parts)

        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            },
            {"type": "text", "text": user_text},
        ]

    def _parse_coordinate(self, content: str) -> Union[List[int], List[List[int]]]:
        """
        从模型输出中提取坐标。

        支持两种格式：
        - 单点: coordinate: [x, y]
        - 多候选: coordinate: [[x1,y1], [x2,y2], ...]
        """
        # 先尝试多候选 [[...], [...]]
        # 匹配 "coordinate": [[100,200], [300,400]]，捕获整个外层数组
        multi_match = re.search(
            r'coordinate["\']?\s*:\s*(\[\s*\[.+?\]\s*\])', content, re.DOTALL
        )
        if multi_match:
            try:
                parsed = json.loads(multi_match.group(1))
                if isinstance(parsed, list) and all(
                    isinstance(c, list) and len(c) >= 2 for c in parsed
                ):
                    return [[int(c[0]), int(c[1])] for c in parsed]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 单点 [x, y]（支持负数如 [-1, -1]）
        single_match = re.search(
            r'coordinate["\']?\s*:\s*\[(-?\d+)\s*,\s*(-?\d+)\]', content
        )
        if single_match:
            return [int(single_match.group(1)), int(single_match.group(2))]

        # 无法解析
        return [-1, -1]

    # ==================== RAG 查询规划（纯文本，不带截图） ====================

    PLAN_RAG_QUERY_PROMPT = """You are a manual retrieval planner. Given a user's intent, the manual's structure summary, and the current progress, output the EXACT step titles to retrieve.

RULES:
- Copy step titles VERBATIM from the structure summary (e.g. "2.2 步骤二：选择医生"), space-separated, in workflow order.
- Prioritize the step(s) matching the CURRENT progress; then cover the remaining steps of the same workflow.
- Output ONLY the step titles. NO explanation. NO punctuation. NO extra words.

Examples:
用户: "我要挂号"
摘要: [2.1 步骤一：选择就诊日期与科室] 日期滑块... [2.2 步骤二：选择医生] 点击医生卡片...
→ 2.1 步骤一：选择就诊日期与科室 2.2 步骤二：选择医生 2.3 步骤三：选择就诊时间 2.4 步骤四：确认挂号"""

    def plan_rag_query(self, user_intent: str, manual_summary: str = "", step_context: str = "") -> str:
        """
        纯文本推演：根据用户意图 + 说明书摘要 + 当前进度，推理需要哪些步骤标题。
        不带截图，节省图像 token 开销。

        Args:
            user_intent: 用户意图描述
            manual_summary: RAG 自动生成的说明书结构摘要
            step_context: 当前已完成/进行中的步骤描述（用于优先检索下一步）

        Returns:
            中文步骤标题关键词（空格分隔），如 "2.2 步骤二：选择医生 2.3 步骤三：选择就诊时间"
        """
        prompt = self._cfg.plan_rag_query_prompt or self.PLAN_RAG_QUERY_PROMPT
        summary_block = f"\n\n说明书结构摘要：\n{manual_summary}" if manual_summary else ""
        progress_block = f"\n\n当前进度（已完成/进行中的步骤）：{step_context}" if step_context else ""
        user_text = f"用户意图：{user_intent if user_intent else '未指定'}{summary_block}{progress_block}\n\n请输出需要检索的说明书步骤标题。"

        payload = {
            "model": self._cfg.model_path,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": self._cfg.plan_rag_query_max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            resp = requests.post(self._cfg.api_url, json=payload,
                                timeout=self._cfg.plan_rag_query_timeout)
            resp.raise_for_status()
            result = resp.json()
            if "choices" in result:
                keywords = result["choices"][0]["message"]["content"].strip()
                # 清理 thinking 标签
                keywords = re.sub(r'</?think>', '', keywords).strip()
                # 如果输出是「解释句。 关键词1 关键词2 ...」，提取纯关键词部分
                # 策略: 取最后一个句号/分号之后、且含空格的纯中文词组段
                parts = re.split(r'[。；;]', keywords)
                for part in reversed(parts):
                    part = part.strip()
                    # 关键词段特征: 含空格、中文为主、无长句
                    if ' ' in part and len(part) < 200 and not any(kw in part for kw in ['用户', '需要', '因此', '输出', '意图']):
                        keywords = part
                        break
                logger.info(f"RAG 查询规划: {keywords}")
                return keywords
        except Exception as e:
            logger.warning(f"RAG 查询规划失败: {e}")

        # 回退：直接用用户意图（清理换行符，确保适合 BM25 查询）
        return (user_intent or "系统首页 服务大厅").replace("\n", " ").strip()

    def _parse_reasoning(self, content: str) -> str:
        """提取 <tool_call> 之前的推理文字。"""
        if "<tool_call>" in content:
            return content.split("<tool_call>")[0].strip()
        return content.strip()
