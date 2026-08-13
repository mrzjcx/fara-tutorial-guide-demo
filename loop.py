"""
Fara ↔ RAG ↔ Checker 联合编排 —— 新架构核心。

流程:
    用户意图 → Fara 纯文本推演所需章节 → RAG 检索相关说明书切片
        → Fara 看图+切片 输出坐标 → Checker 验证
             ├─ [OK] 正确 → 返回最终坐标和建议
            └─ [失败] 错误 → Checker 反馈 → Fara 重试（最多 max_retries 次，现在不再重试而是输出建议）

"""
import logging
from typing import List, Dict, Optional, Tuple

from rag.config import RAGConfig
from rag.retriever import RAGRetriever
from fara.config import FaraConfig
from fara.actor import FaraActor, FaraResult
from checker.config import CheckerConfig
from checker.checker import Checker, CheckerResult

logger = logging.getLogger(__name__)


# ============================================================
# ⑤ 人机确认点：检测不可逆操作（提交/支付/确认等）
# ============================================================

_CONFIRM_KEYWORDS = (
    "确认挂号", "确认支付", "确认缴费", "确认取药",
    "去缴费", "立即缴费", "提交", "支付密码",
)


def _needs_confirmation(reasoning: str, snippets_text: str) -> bool:
    """检测当前建议动作是否涉及不可逆操作。"""
    text = f"{reasoning or ''} {snippets_text or ''}"
    return any(k in text for k in _CONFIRM_KEYWORDS)


def _apply_confirmation(fara_result, rag_snippets) -> None:
    """⑤ 标记是否需要人工确认（仅依据 Fara 引导语，避免 RAG 切片标题误触发）。"""
    if fara_result.needs_confirmation or not fara_result.reasoning:
        return
    if _needs_confirmation(fara_result.reasoning, ""):
        fara_result.needs_confirmation = True


# ============================================================
# ④ 失败降级：无有效坐标时降级为文字指导
# ============================================================

def _make_degraded(fara_result) -> FaraResult:
    """将无有效坐标的结果降级为文字指导（degraded=True）。"""
    return FaraResult(
        success=True,
        coordinate=[-1, -1],
        reasoning=(fara_result.reasoning
                   or "模型未能确定下一步，请根据页面情况自行操作。"),
        raw_response=fara_result.raw_response,
        prompt_tokens=fara_result.prompt_tokens,
        completion_tokens=fara_result.completion_tokens,
        degraded=True,
    )


# ============================================================
# 关键词质量检测（plan_rag_query 输出可能被 Fara 复述为摘要）
# ============================================================

def _is_summary_like(text: str) -> bool:
    """检测文本是否像说明书摘要复述而非搜索关键词。"""
    if not text or len(text) < 3:
        return True
    # 含文档结构标记
    for marker in ['版本', '说明书', '系统入口', '使用说明', 'Document', 'Chunks', '流程 -']:
        if marker in text:
            return True
    # 含句号/分号 = 是句子不是关键词
    if '。' in text or '；' in text:
        return True
    # 超过80字且空格少于3个 = 大概率是段落不是关键词
    if len(text) > 80 and text.count(' ') < 3:
        return True
    return False


def _build_fallback_keywords(rag_retriever, user_intent: str, step_context: str = "") -> str:
    """从索引中提取所有编号步骤标题，拼接为兜底关键词。"""
    all_steps = []
    for i in range(len(rag_retriever._index)):
        ch = rag_retriever._index.get_chunk(i)
        # 只收集编号步骤（如 "2.1 步骤一：..."），跳过概述/章标题/按钮速查表
        if ch.step_title and ch.step_title[0].isdigit():
            all_steps.append(ch.step_title)
    prefix = step_context if step_context else user_intent
    return prefix + ' ' + ' '.join(all_steps)


def _fill_step_gaps(rag_retriever, snippets: List[Dict], max_n: int) -> List[Dict]:
    """激进补全：以得分最高的切片为锚点，返回其所属 section 内的连续步骤区间。

    优先包含锚点后续步骤（向后扩展），不足再向前补前序步骤，凑满 max_n。
    结果按文档顺序排序——保证 Fara/Checker 拿到的是连续的步骤上下文，
    而非零散的、可能跨流程的检索命中。
    """
    if not snippets:
        return snippets
    best = max(snippets, key=lambda s: s.get("score", 0))
    target_section = best.get("section", "")
    if not target_section or target_section == "概述":
        return snippets
    # 该 section 内所有 chunk 索引（按文档顺序）
    section_indices = [
        i for i in range(len(rag_retriever._index))
        if rag_retriever._index.get_chunk(i).section_title == target_section
    ]
    if len(section_indices) < 2:
        return snippets
    anchor = rag_retriever._index._id_to_idx.get(best["chunk_id"])
    if anchor not in section_indices:
        return snippets
    pos = section_indices.index(anchor)
    # 以锚点为中心扩展，优先向后（后续步骤）
    chosen = [anchor]
    forward = section_indices[pos + 1:]
    backward = list(reversed(section_indices[:pos]))
    fi, bi = 0, 0
    while len(chosen) < max_n and (fi < len(forward) or bi < len(backward)):
        if fi < len(forward):
            chosen.append(forward[fi])
            fi += 1
        elif bi < len(backward):
            chosen.append(backward[bi])
            bi += 1
    filled = []
    for i in sorted(chosen):
        ch = rag_retriever._index.get_chunk(i)
        filled.append({
            "chunk_id": ch.chunk_id,
            "section": ch.section_title,
            "step": ch.step_title,
            "text": ch.text,
            "hint": ch.metadata.get("hint", ""),
            "score": 0.0,
            "context_text": "",
            "neighbors": [],
            "page": ch.page_start,
            "image_path": ch.page_image_path,
        })
    return filled[:max_n]


class FaraCheckerLoop:
    """
    Fara + Checker 反馈迭代编排器。
    将 RAG 检索、Fara 推理、Checker 验证串联为闭环。

    Usage:
        from rag import RAGConfig, RAGRetriever
        from fara import FaraConfig, FaraActor
        from checker import CheckerConfig, Checker
        from loop import FaraCheckerLoop

        loop = FaraCheckerLoop(rag_retriever, fara_actor, checker)
        final_coord, history = loop.run(
            screenshot_b64="...",
            user_intent="在线挂号选骨科",
        )
    """

    def __init__(
        self,
        rag: RAGRetriever,
        fara: FaraActor,
        checker: Checker = None,
        max_retries: int = 3,
    ):
        self._rag = rag
        self._fara = fara
        self._checker = checker
        self._max_retries = max_retries
        self._step_history: List[str] = []  # P0-1: 已完成微动作
        self._query_cache: Dict[Tuple[str, str], str] = {}  # P2-6: (intent, step_context)→keywords
        self._last_snippets: List[Dict] = []  # 最近一次实际喂给模型的 RAG 切片（展示用）

    def reset_context(self):
        """重置步骤上下文（新任务开始前调用）。"""
        self._step_history = []

    # ==================== 主循环 ====================

    def run(
        self,
        screenshot_b64: str,
        user_intent: str = "",
        step_context: str = "",
        top_k_rag: int = 3,
    ) -> Tuple[FaraResult, List[Dict]]:
        """
        执行 纯文本推演→RAG→Fara/Checker 反馈迭代，返回最终坐标。

        流程:
          Step 0: Fara 纯文本推演 → 推理需要哪些说明书章节
          Step 1: Fara 的推演关键词 → RAG 检索相关切片
          Step 2-N: Fara 看图+切片 → 坐标 → Checker 验证 → 反馈迭代

        Args:
            screenshot_b64: 截图 base64（仅坐标推理阶段使用）
            user_intent: 用户意图
            step_context: 当前步骤状态
            top_k_rag: RAG 检索返回的切片数

        Returns:
            (final_result, history): 最终 FaraResult 和每轮迭代记录列表
        """
        # ============================================================
        # Step 0: RAG 自动生成说明书结构摘要 + 关键词提取
        # ============================================================
        manual_summary = self._rag.get_summary()
        # P2-6: 缓存 key = 意图 + 当前进度（进度变化 → 重新检索下一步相关步骤）
        cache_key = (user_intent, step_context)
        if cache_key in self._query_cache:
            rag_keywords = self._query_cache[cache_key]
            logger.info(f"查询缓存命中: {rag_keywords[:60]}...")
        else:
            rag_keywords = self._fara.plan_rag_query(user_intent, manual_summary, step_context)
            # 验证: 检测摘要复述（含文档结构标记、句号、或过长）
            if _is_summary_like(rag_keywords):
                logger.warning("plan_rag_query 输出疑似摘要复述，使用 intent + 步标题兜底")
                rag_keywords = _build_fallback_keywords(self._rag, user_intent, step_context)
            else:
                self._query_cache[cache_key] = rag_keywords  # 仅缓存有效关键词
            logger.info(f"查询关键词: {rag_keywords[:80]}...")

        # ============================================================
        # Step 1: 关键词 → RAG 检索 + 步骤连续性补全
        # ============================================================
        rag_snippets = self._rag.query(rag_keywords, top_k=top_k_rag) if rag_keywords else []
        rag_snippets = _fill_step_gaps(self._rag, rag_snippets, top_k_rag)
        self._last_snippets = rag_snippets  # 保存实际喂给模型的切片（供展示层读取）
        structured_snippets = self._format_snippets_structured(rag_snippets) if rag_snippets else ""  # P1-3
        logger.info(f"RAG 检索 → {len(rag_snippets)} 个切片: "
                     f"{[s.get('step', '?') for s in rag_snippets]}")

        history: List[Dict] = []
        checker_feedback = ""

        # P0-1: 构建已完成步骤文本
        if step_context:
            # 外部传入的初始上下文（如 "用户已打开挂号页面"）
            if step_context not in self._step_history:
                self._step_history.append(step_context)

        for attempt in range(self._max_retries + 1):
            logger.info(f"=== 第 {attempt + 1}/{self._max_retries + 1} 轮 ===")

            # 格式化已完成微动作列表
            done_text = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(self._step_history)) if self._step_history else "尚未执行任何操作"
            # P1-3: 注入结构化 RAG 信息
            context_with_rag = f"已完成的操作:\n{done_text}\n\n当前步骤说明书:\n{structured_snippets}"

            # ---- Fara 推理 ----
            fara_result = self._fara.act(
                screenshot_b64=screenshot_b64,
                user_intent=user_intent,
                rag_snippets=rag_snippets,
                step_context=context_with_rag,
                checker_feedback=checker_feedback,
            )

            entry = {
                "attempt": attempt + 1,
                "fara": fara_result,
                "checker": None,
            }
            history.append(entry)

            if not fara_result.success:
                logger.error(f"Fara 调用失败: {fara_result.error}")
                return fara_result, history

            if not fara_result.is_valid:
                logger.warning("Fara 未返回有效坐标，降级为文字指导")
                degraded = _make_degraded(fara_result)
                return degraded, history

            # ---- Checker 验证 ----
            if self._checker is None or not self._checker.enabled:
                logger.info("Checker 未启用，直接返回 Fara 结果")
                self._record_action(fara_result)
                _apply_confirmation(fara_result, rag_snippets)
                return fara_result, history

            # 多候选坐标不支持 Checker 验证，直接返回
            if fara_result.single_coord is None:
                logger.info("多候选坐标，跳过 Checker 验证，直接返回")
                self._record_action(fara_result)
                _apply_confirmation(fara_result, rag_snippets)
                return fara_result, history

            checker_result = self._checker.verify(
                screenshot_b64=screenshot_b64,
                fara_coord=fara_result.single_coord,
                fara_reasoning=fara_result.reasoning,
                rag_snippets=rag_snippets,
                user_intent=user_intent,
            )
            entry["checker"] = checker_result

            # ---- 判断 ----
            if checker_result.is_correct:
                _apply_confirmation(fara_result, rag_snippets)
                logger.info(f"[OK] Checker 确认正确，返回坐标 {fara_result.coordinate}"
                            + ("（需人工确认）" if fara_result.needs_confirmation else ""))
                self._record_action(fara_result)
                return fara_result, history

            if checker_result.needs_retry and attempt < self._max_retries:
                logger.info(f"[失败] Checker 拒绝 (attempt {attempt+1}): {checker_result.reason}")
                checker_feedback = checker_result.reason
                continue

            # 达到最大重试：Checker(0.8B) 视觉不可靠，不否决 Fara 坐标，
            # 返回坐标 + Checker 意见，由用户自行判断（人在回路）
            logger.warning(
                f"Checker {'拒绝' if checker_result.needs_retry else '无法判断'}"
                f"，已达最大重试，返回 Fara 坐标（请人工判断）"
            )
            self._record_action(fara_result)
            return fara_result, history

    def _record_action(self, fara_result: FaraResult):
        """P0-1: 记录已完成的操作到历史列表。"""
        if fara_result.success and fara_result.reasoning:
            desc = fara_result.reasoning.split("。")[0].strip()
            if len(desc) > 5:
                self._step_history.append(desc)
                logger.info(f"[记录] 记录操作: {desc}")

    def _format_snippets_structured(self, snippets: List[Dict]) -> str:
        """P1-3: 将 RAG 切片转为结构化任务描述。"""
        if not snippets:
            return "(无相关说明书内容)"

        lines = []
        for i, s in enumerate(snippets, 1):
            step = s.get("step", "?")
            text = s.get("text", "").replace("\n", " ")
            hint = s.get("hint", "")
            # 提取操作关键词
            actions = []
            if "点击" in text:
                actions.append("点击")
            if "选择" in text:
                actions.append("选择")
            if "填写" in text:
                actions.append("填写")
            if "确认" in text:
                actions.append("确认")
            action_hint = f" [操作: {', '.join(actions)}]" if actions else ""
            lines.append(f"  Step {i}: {step}{action_hint}\n    {text[:150]}")
            if hint:
                lines.append(f"    → {hint}")

            # 附加上下文邻块
            ctx = s.get("context_text", "")
            if ctx:
                lines.append(f"    前后步骤: {ctx[:120]}")

        return "\n".join(lines)

    # ==================== 便捷方法 ====================

    def run_simple(
        self,
        screenshot_b64: str,
        user_intent: str = "",
    ) -> Optional[List[int]]:
        """
        简化版：只返回最终坐标（单点），适合直接用于点击。

        Returns:
            [x, y] 或 None
        """
        result, _ = self.run(screenshot_b64, user_intent)
        return result.single_coord if result.is_valid else None
