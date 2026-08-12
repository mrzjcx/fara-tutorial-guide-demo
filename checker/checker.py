"""
Checker：看图 + RAG 切片 + Fara 坐标 → 验证正确性 → 返回反馈。
"""
import re
import io
import os
import json
import base64
import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from .config import CheckerConfig

logger = logging.getLogger(__name__)


# ============================================================
# 截图局部放大（②）：坐标附近区域裁剪放大，辅助精确判断
# ============================================================

def crop_zoom_base64(screenshot_b64: str, x: int, y: int,
                     scale: int = 3, radius: int = 90) -> str:
    """将 1000x1000 归一化坐标 (x, y) 附近的区域裁剪并放大，返回 base64 PNG。"""
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))
    w, h = img.size
    px = int(x * w / 1000)
    py = int(y * h / 1000)
    box = (max(0, px - radius), max(0, py - radius),
           min(w, px + radius), min(h, py + radius))
    crop = img.crop(box)
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_combined_view(screenshot_b64: str, zoom_b64: str, target_h: int = 360) -> str:
    """将全景缩略图 + 局部放大图横向拼接为单张图（满足 vLLM image=1 限制）。"""
    from PIL import Image
    full = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))
    zoom = Image.open(io.BytesIO(base64.b64decode(zoom_b64)))
    full = full.resize((max(1, int(full.width * target_h / full.height)), target_h), Image.LANCZOS)
    zoom = zoom.resize((max(1, int(zoom.width * target_h / zoom.height)), target_h), Image.LANCZOS)
    canvas = Image.new("RGB", (full.width + zoom.width, target_h), "white")
    canvas.paste(full, (0, 0))
    canvas.paste(zoom, (full.width, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# 默认 Checker System Prompt
# ============================================================

DEFAULT_CHECKER_SYSTEM_PROMPT = """You are a GUI verification assistant. Your job is to CHECK whether a suggested click position is correct.

You will receive:
1. A screenshot of a web/app interface
2. Tutorial steps (retrieved from a manual via RAG)
3. A suggested click coordinate [x, y] from another model
4. That model's reasoning

Your task: Look at the screenshot and decide if clicking at [x, y] is the RIGHT next action according to the tutorial.

IMPORTANT — Your only task (no coordinate conversion):
- The SECOND image is a zoomed-in crop whose CENTER is exactly where Fara suggests clicking.
- Judge ONLY the content at the CENTER of the SECOND image: does it match what the tutorial step requires for the next action?
- Do NOT convert coordinates and do NOT locate anything by percentage — the crop already centers the suggested location.
- Examples: center shows "14号" date card while the tutorial says select 14号 → correct; center shows empty space or an unrelated element → incorrect.
- Judge based ONLY on elements ACTUALLY VISIBLE in the screenshots. NEVER invent or assume UI elements that are not present.
- If the center is empty space / no clear element / you cannot see it clearly, do NOT invent an element name — say the target is unclear.
- Tutorial text mentioning "医生/科室/支付" does NOT mean those elements exist in the screenshot. The screenshots are the ONLY ground truth.

IMPORTANT — Single-step guidance flow:
- This system guides the user ONE action at a time: Fara outputs exactly ONE click per round.
- Judge ONLY whether the center of the second image is the correct target for THIS round's single action.
- Do NOT require the whole tutorial step (e.g., "select date AND department") to be completed in one round — the remaining sub-actions (like selecting department) are the NEXT rounds, not mistakes.
- If Fara's guidance says "click 14号" and the center shows 14号 → correct:true, even if department/doctor are not yet selected.

IMPORTANT — Step completion check (do this BEFORE judging the coordinate):
- First check whether the current step's required items are ALREADY completed on the screenshot. A required item is completed when it is highlighted/selected — e.g., its border is highlighted (turns blue or another highlight color), or it appears selected/checked.
- If ALL required items of the current step are already selected/highlighted, then clicking the "下一步"/confirm button to move on is CORRECT — it is NOT skipping a step.
- Only judge "skipping a step" as WRONG when the required items are NOT yet selected/highlighted on the screenshot.

Output format (STRICT):
<tool_call>{"name": "verify", "arguments": {"correct": true_or_false, "reason": "one sentence in Chinese"}}</tool_call>

Rules:
- Compare the element at [x, y] against the tutorial's expected next action.
- If the element at [x, y] matches the tutorial's purpose → correct: true
- If the element is wrong, irrelevant, or skips a step → correct: false
- Examples of WRONG suggestions:
  - Tutorial says "select a department" but the department is NOT selected and coordinate points to "下一步" button (skipping selection)
  - Tutorial says "fill name field" but coordinate points to submit button (skipping required fields)
  - Coordinate points to empty space or wrong UI element
- Examples of CORRECT suggestions:
  - Tutorial says "select 骨科" and coordinate points to "骨科" in the department list
  - Tutorial says "click 下一步" and an item IS already selected, coordinate points to the button
  - Tutorial says "select date and department" but both are ALREADY highlighted/selected on the screenshot, and coordinate points to "下一步" → correct (the step is already completed, moving on is right)
- Keep reason short (one sentence in Chinese).
- If you are unsure, use your best judgment."""


@dataclass
class CheckerResult:
    """Checker 单次验证结果"""
    enabled: bool
    success: bool = True
    verified: Optional[bool] = None      # True=正确, False=错误, None=无法判断
    reason: str = ""                     # 验证理由（中文）
    raw_response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    skipped: bool = False
    error: str = ""

    @property
    def is_correct(self) -> bool:
        """Checker 认为 Fara 的坐标正确。"""
        return self.enabled and self.verified is True

    @property
    def needs_retry(self) -> bool:
        """是否需要让 Fara 重试（明确判错即可；reason 为空时 verify() 已兜底）。"""
        return self.enabled and self.verified is False


class Checker:
    """
    Checker 验证器：调用 Qwen Checker 模型验证 Fara 输出的坐标。

    Usage:
        # 端点/模型路径统一见 scripts/config.py（此处示例直接引用）
        from scripts.config import CHECKER_API_URL, CHECKER_MODEL_PATH
        config = CheckerConfig(
            api_url=CHECKER_API_URL,
            model_path=CHECKER_MODEL_PATH,
        )
        checker = Checker(config)
        result = checker.verify(
            screenshot_b64="...",
            fara_coord=[320, 450],
            fara_reasoning="骨科高亮可见，下一步应点击下一步按钮",
            rag_snippets=[...],
            user_intent="在线挂号选骨科",
        )
        if result.needs_retry:
            print(f"需要重试: {result.reason}")
    """

    def __init__(self, config: CheckerConfig):
        self._cfg = config
        self._enabled = bool(config.api_url)
        self._system_prompt = config.system_prompt or DEFAULT_CHECKER_SYSTEM_PROMPT
        self._last_coord: Optional[List[int]] = None  # P1-4: 轻量去重
        if self._enabled:
            logger.info(f"Checker 初始化: api={config.api_url}, model={config.model_path}")
        else:
            logger.info("Checker 未启用（api_url 为空）")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ==================== 核心调用 ====================

    def verify(
        self,
        screenshot_b64: str,
        fara_coord: List[int],
        fara_reasoning: str = "",
        rag_snippets: List[Dict] = None,
        user_intent: str = "",
    ) -> CheckerResult:
        """
        验证 Fara 输出的坐标是否正确。

        Args:
            screenshot_b64: 截图 base64 编码
            fara_coord: Fara 输出的坐标 [x, y]
            fara_reasoning: Fara 的推理文字
            rag_snippets: RAG 检索到的说明书切片
            user_intent: 用户意图

        Returns:
            CheckerResult: 验证结果
        """
        if not self._enabled:
            return CheckerResult(enabled=False)

        # P1-4: 轻量校验前置 — 坐标合法 + 去重
        if not fara_coord or len(fara_coord) < 2 or fara_coord[0] < 0 or fara_coord[1] < 0:
            return CheckerResult(enabled=True, skipped=True,
                                reason="坐标不合法，跳过 LLM 验证")

        if self._last_coord and abs(fara_coord[0] - self._last_coord[0]) < 5 \
                and abs(fara_coord[1] - self._last_coord[1]) < 5:
            logger.info("检测到与上轮坐标相近 (diff<5px)，仍需 Checker LLM 核实")
            # 不跳过——让 Checker 模型判断是否真的是无效重复

        self._last_coord = list(fara_coord)

        x, y = fara_coord[0], fara_coord[1]

        # ② 局部放大：坐标附近区域裁剪放大，作为第二张图辅助 Checker 确认细节
        zoom_b64 = ""
        try:
            zoom_b64 = crop_zoom_base64(screenshot_b64, x, y)
        except Exception as e:
            logger.debug(f"局部放大失败: {e}")

        # 调试保存：CHECKER_DEBUG_SAVE（config 默认开启）时保存 Checker 实际看到的图与信息
        if getattr(self._cfg, "debug_save", False):
            try:
                from pathlib import Path
                dbg_dir = Path("output/checker_debug")
                dbg_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                (dbg_dir / f"{ts}_{x}_{y}_full.png").write_bytes(base64.b64decode(screenshot_b64))
                if zoom_b64:
                    (dbg_dir / f"{ts}_{x}_{y}_zoom.png").write_bytes(base64.b64decode(zoom_b64))
                (dbg_dir / f"{ts}_{x}_{y}_info.txt").write_text(
                    f"坐标: [{x}, {y}]\n引导语: {fara_reasoning}\n", encoding="utf-8")
                logger.info(f"Checker 调试图已保存: output/checker_debug/")
            except Exception as e:
                logger.debug(f"调试保存失败: {e}")

        # 构建消息
        user_content = self._build_messages(
            screenshot_b64, x, y, fara_reasoning, rag_snippets, user_intent, zoom_b64,
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
            return CheckerResult(
                enabled=True, success=False,
                error=f"无法连接 Checker 服务 ({self._cfg.api_url})",
            )
        except Exception as e:
            return CheckerResult(
                enabled=True, success=False,
                error=f"Checker 异常: {e}",
            )

        if "choices" not in result:
            return CheckerResult(
                enabled=True, success=False,
                error=f"Checker API 返回异常: {result}",
            )

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        # 解析验证结果
        verified, reason = self._parse_verification(content)
        # 判错但 reason 为空 → 兜底反馈，保证重试信息能传达给 Fara
        if verified is False and not reason:
            reason = "Checker 认为该坐标不正确，请重新分析截图后给出正确的点击位置。"

        return CheckerResult(
            enabled=True,
            success=True,
            verified=verified,
            reason=reason,
            raw_response=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    # ==================== 内部方法 ====================

    def _build_messages(
        self,
        screenshot_b64: str,
        x: int, y: int,
        fara_reasoning: str,
        rag_snippets: List[Dict],
        user_intent: str,
        zoom_b64: str = "",
    ) -> List[Dict]:
        """构建 Checker 的 user message（全景图 + 可选局部放大图 + 文本）。"""
        # RAG 切片
        if rag_snippets:
            snippets_text = "\n\n".join(
                f"[{s.get('step', '?')}] {s.get('text', '')}"
                for s in rag_snippets
            )
        else:
            snippets_text = "(无说明书切片)"

        text = (
            f"[教程] 说明书相关步骤（RAG 检索）：\n{snippets_text}\n\n"
            f"[目标] 用户意图：{user_intent if user_intent else '未指定'}\n\n"
            f"[模型] Fara 建议点击的位置 = 第二张放大图的中心（无需换算坐标）\n"
            f"[记录] Fara 的引导: {fara_reasoning}\n\n"
            f"请判断：第二张放大图中心位置的内容，是否符合教程当前步骤的要求。"
        )

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            },
        ]
        if zoom_b64:
            # 第二张图：坐标附近的局部放大图（Checker 服务已允许 image=2）
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{zoom_b64}"},
            })
        content.append({"type": "text", "text": text})
        return content

    def _parse_verification(self, content: str) -> tuple:
        """解析 Checker 输出的 correct + reason（多格式鲁棒解析）。"""
        correct_match = re.search(
            r'"correct"\s*:\s*(true|false)', content, re.IGNORECASE
        )
        verified = None
        if correct_match:
            verified = correct_match.group(1).lower() == "true"

        reason = ""
        # ① 优先：整体解析 <tool_call> 内 JSON（json.loads 正确处理中文/转义）
        tc = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
        if tc:
            try:
                obj = json.loads(tc.group(1))
                args = obj.get("arguments", obj) if isinstance(obj, dict) else {}
                if isinstance(args, dict):
                    reason = args.get("reason", "") or ""
            except (json.JSONDecodeError, TypeError):
                pass
        # ② 整体 JSON 解析失败 → 双引号正则
        if not reason:
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', content)
            if reason_match:
                reason = reason_match.group(1)
        # ③ 单引号
        if not reason:
            reason_match = re.search(r"'reason'\s*:\s*'([^']*)'", content)
            if reason_match:
                reason = reason_match.group(1)
        # ④ fallback：<tool_call> 之前的文本（去除思维链标记）
        if not reason:
            if "<tool_call>" in content:
                reason = content.split("<tool_call>")[0].strip()
            else:
                reason = content.strip()
            reason = re.sub(r'</?think>', '', reason).strip()

        return verified, reason
