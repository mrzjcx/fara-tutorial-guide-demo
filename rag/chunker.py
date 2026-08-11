"""
文本分块器：按章节/步骤将 PDF 文本切分为语义独立的 Chunk。

P0 重构:
  - ① 步骤正则去掉 \\s+ 硬性要求
  - ② chunk_id 顺序生成 (sec_0001, sec_0002...)
  - ③ 标题自动探测（全文扫描统计）
  - ④ 多级层级支持
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from .pdf_parser import PDFData
from .config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """一个语义切片"""
    chunk_id: str               # 唯一 ID，顺序生成如 "sec_0001"
    section_title: str          # 所属章节标题
    step_title: str             # 步骤标题
    text: str                   # 正文内容
    page_start: int             # 起始页码（1-indexed）
    page_end: int               # 结束页码
    page_image_path: str = ""   # 关联的页面截图路径
    metadata: dict = field(default_factory=dict)

    @property
    def index(self) -> int:
        """从 chunk_id 提取顺序索引。"""
        try:
            return int(self.chunk_id.split("_")[1])
        except (IndexError, ValueError):
            return -1


# ============================================================
# 通用步骤正则（P0-①: 去掉 \\s+ 硬性要求）
# ============================================================
# 匹配: "2.1标题" / "2.1 标题" / "2.1、标题" / "2.1.3 标题"
_STEP_REGEX_RAW = r"\d+(?:\.\d+)+[、.．]?\s*[^\n]+"


def detect_title_style(full_text: str, candidates: tuple) -> str:
    """(P0-③) 自动探测说明书所用的章节编号风格。返回命中行数最多的模式。"""
    best = ""
    best_cnt = 0
    for pat in candidates:
        try:
            cnt = len(re.findall(pat, full_text, re.MULTILINE))
            if cnt > best_cnt:
                best_cnt = cnt
                best = pat
        except re.error:
            continue
    if best and best_cnt >= 2:
        logger.info(f"自动探测标题风格: {best!r} (命中{best_cnt}行)")
        return best
    logger.info("自动探测失败，回退默认")
    return ""


def _build_section_regex(full_text: str, config: RAGConfig) -> str:
    """构建章节切分正则。"""
    if config.title_auto_detect:
        detected = detect_title_style(full_text, config.title_candidate_patterns)
        if detected:
            return detected
    patterns = [re.escape(p) for p in config.section_patterns if p]
    if patterns:
        return "|".join(patterns)
    return r"^[一二三四五六七八九十]+[、]"


def chunk_pdf(pdf_data: PDFData, config: RAGConfig) -> List[Chunk]:
    """将 PDF 数据切分为 Chunk 列表。"""
    if not pdf_data.pages:
        return []

    full_text = "\n".join(p.text for p in pdf_data.pages if p.text)
    if not full_text.strip():
        return []

    section_regex = _build_section_regex(full_text, config)
    logger.info(f"章节正则: {section_regex!r}")

    sections = _split_by_regex(full_text, section_regex)

    # 多级层级 (P1-④)
    if config.level_patterns:
        fragments = []
        for sec_title, sec_body in sections:
            fragments.extend(_multi_split(sec_title, sec_body, config.level_patterns))
    else:
        fragments = sections

    # 二级切分：步骤
    chunks: List[Chunk] = []
    idx = 0

    for title, body in fragments:
        steps = _split_by_regex(body, _STEP_REGEX_RAW)
        if len(steps) <= 1:
            text = body.strip()
            if len(text) >= config.chunk_min_chars:
                chunks.append(_make_chunk(idx, title, title, text, pdf_data))
                idx += 1
        else:
            for st, sb in steps:
                text = sb.strip()
                if len(text) < config.chunk_min_chars:
                    continue
                chunks.append(_make_chunk(idx, title, st, text, pdf_data))
                idx += 1

    # 去重 (按chunk_id)
    seen = set()
    unique = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)

    logger.info(f"分块完成: {len(unique)} chunks")
    return unique


def _multi_split(title: str, body: str, level_patterns: tuple) -> List[Tuple[str, str]]:
    """按多级正则逐级切分。"""
    frags = [(title, body)]
    for pat in level_patterns:
        new_frags = []
        for t, b in frags:
            parts = _split_by_regex(b, pat)
            if len(parts) <= 1:
                new_frags.append((t, b))
            else:
                for st, sb in parts:
                    new_frags.append((st or t, sb))
        frags = new_frags
    return frags


def _split_by_regex(text: str, pattern: str) -> List[Tuple[str, str]]:
    """按正则切分，返回 [(标题, 正文), ...]。首段之前标为'概述'。"""
    try:
        regex = re.compile(f"({pattern})", re.MULTILINE)
    except re.error:
        return [(text[:50].strip(), text)] if text.strip() else []

    parts = regex.split(text)
    if len(parts) <= 1:
        return [(text[:50].strip(), text)] if text.strip() else []

    result = []
    preamble = parts[0].strip()
    if preamble:
        result.append(("概述", preamble))

    i = 1
    while i + 1 < len(parts):
        title = parts[i].strip()
        body = parts[i + 1]
        if title:
            result.append((title, body))
        i += 2
    return result


def _make_chunk(idx: int, section: str, step: str,
                text: str, pdf_data: PDFData) -> Chunk:
    """创建 Chunk，ID 为 sec_{idx:04d}。P2-5: 自动提取 action_type。"""
    page = _find_page(text, pdf_data)
    actions = []
    if "点击" in text: actions.append("click")
    if "选择" in text: actions.append("select")
    if "填写" in text: actions.append("input")
    if "确认" in text: actions.append("confirm")
    action_type = ",".join(actions) if actions else "info"
    hint = _generate_interaction_hint(text, step, action_type)
    return Chunk(
        chunk_id=f"sec_{idx:04d}",
        section_title=section,
        step_title=step,
        text=text,
        page_start=page,
        page_end=page,
        page_image_path=_page_image(page, pdf_data),
        metadata={"section": section, "action_type": action_type, "hint": hint},
    )


def _generate_interaction_hint(text: str, step_title: str, action_type: str) -> str:
    """生成通用 GUI 交互提示。不绑定具体 CSS/颜色/按钮文案。"""
    parts = []
    actions = set(action_type.split(","))

    if "select" in actions or ("click" in actions and ("选择" in text or "选" in text or "选择" in step_title)):
        parts.append("先检查目标项是否已处于选中态(高亮/变色/勾选) → 已选中则跳过此项 → 未选中才点击它")

    if "input" in actions:
        parts.append("从上到下扫描输入框，仅点击第一个空白的输入框；已填文字的跳过")

    if "confirm" in actions or "确认" in text:
        parts.append("确保本步所有必选项已选完、所有必填框已填完后，再点确认按钮(灰色=disabled=还有未完成项)")

    if "下一步" in text:
        parts.append("只有当本步所有项目都已经选中时才点「下一步」，否则继续选择未选中的项目")

    if not parts:
        return ""
    return " | ".join(parts)


def _find_page(text: str, pdf_data: PDFData) -> int:
    for page in pdf_data.pages:
        if text[:30] in page.text:
            return page.page_num
    return 1


def _page_image(page_num: int, pdf_data: PDFData) -> str:
    for p in pdf_data.pages:
        if p.page_num == page_num and p.page_image_path:
            return p.page_image_path
    return ""

