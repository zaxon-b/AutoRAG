"""答案生成：把检索证据组织成 grounded prompt 并调用 LLM。

没有配置 API Key 或模型调用失败时，用规则从证据中抽取片段兜底，
保证接口仍能返回带页码的回答。
"""

import json
import os
import re
from collections.abc import Generator

from openai import OpenAI


def _llm_api_key() -> str | None:
    return os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def _llm_base_url() -> str | None:
    return os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")


def _llm_model() -> str:
    return os.getenv("LLM_MODEL", "deepseek-r1")


def build_grounded_prompt(question: str, evidence: list[dict]) -> str:
    """把证据编号后拼进 prompt，并要求模型只依据证据回答。"""

    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        role = item.get("evidence_role")
        if not role:
            role = "visual" if item.get("modality") in {
                "icon",
                "operation_figure",
                "page_snapshot",
            } else "text"
        blocks.append(
            f"[{index}] role={role} page={item.get('page')} modality={item.get('modality')} "
            f"section={item.get('section_title') or ''} object={item.get('object_name') or ''}\n"
            f"{item.get('content_text') or ''}"
        )
    evidence_text = "\n".join(blocks)
    has_visual_evidence = any("role=visual" in block for block in blocks)
    # 只有存在视觉证据时才追加视觉/正文的角色约束，纯文本问答不受影响。
    role_instructions = (
        "\n- role=visual 的唯一职责是确认用户图片对应手册中的什么视觉对象。"
        "\n- role=text 的职责是说明该对象的含义、使用方法、触发条件和处理建议。"
        "\n- 不得仅凭视觉相似度补写正文没有支持的确定结论。"
        if has_visual_evidence
        else ""
    )
    return f"""你是文档证据问答助手。
你只能基于给定文档证据回答。
如果证据不足，明确说明无法从当前文档确认。
不要因为文档不是车主手册、合同、研报或其他特定类型而拒答；只要证据能支持，就应直接回答。
如果用户要求推断、同比、环比、占比等简单计算，可以基于证据中的数值进行算术推导，并写出公式或计算过程。

用户问题：
{question}

候选证据：
{evidence_text}

回答要求：
- 先直接回答用户问题。
- 引用证据编号和页码。
- 涉及计算时，说明使用的原始数值和公式。
- 不编造文档外信息。
{role_instructions}
"""


def fallback_answer(question: str, evidence: list[dict]) -> str:
    """无 LLM 时的兜底回答：优先返回与问题相关的证据片段。"""

    if not evidence:
        return "当前证据不足，无法可靠回答这个问题。请确认已上传正确文档，或换一个更具体的问题。"

    snippets = _extract_snippets(question, evidence)
    if snippets:
        joined = "；".join(snippets)
        return f"根据当前检索到的证据，{joined}。"

    first = evidence[0]
    page = first.get("page")
    text = _clean_text(first.get("content_text") or "")
    return f"根据当前检索到的证据，最相关内容出现在第 {page} 页：{text[:360]}。"


def _extract_snippets(question: str, evidence: list[dict]) -> list[str]:
    """从 Top3 证据里截取问题关键词附近的文字。

    财务指标相关的正则分支来自原项目遗留场景；在车主手册语料上通常
    不会命中，命不中时走通用关键词窗口逻辑。
    """

    question = _clean_text(question)
    revenue_question = bool(re.search(r"营收|营业收入|收入", question))
    snippets: list[str] = []
    fallback_snippets: list[str] = []

    for index, item in enumerate(evidence[:3], start=1):
        page = item.get("page")
        text = _clean_text(item.get("content_text") or "")
        if not text:
            continue

        revenue = _extract_revenue_answer(question, text)
        if revenue:
            snippets.append(f"{revenue}（证据 {index}，第 {page} 页）")
            continue

        # 问的是营收时，不再用普通关键词窗口凑答案，避免答非所问。
        if revenue_question:
            continue

        focused = _window_around_keywords(question, text)
        if focused:
            fallback_snippets.append(f"{focused}（证据 {index}，第 {page} 页）")

    if snippets:
        return _dedupe(snippets)[:3]
    return _dedupe(fallback_snippets)[:3]


def _extract_revenue_answer(question: str, text: str) -> str | None:
    """遗留的财务指标提取分支；仅当问题提到营收时生效。"""

    if not re.search(r"营收|营业收入|收入", question):
        return None

    patterns = [
        r"(?:公司发布\s*)?(?:2023年中报，?)?(?:上半年)?[^。；]{0,30}实现营业收入\s*[\d.]+\s*亿元\s*，?\s*同比(?:下降|增长|降低|提升)\s*[\d.]+%",
        r"(?:其中)?(?:第二季度|Q2)[^。；]{0,30}营业收入\s*[\d.]+\s*亿元\s*，?\s*同比(?:下降|增长|降低|提升)\s*[\d.]+%",
        r"营业收入\s*[\d.]+\s*亿元",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    if not matches:
        return None

    return "；".join(_dedupe(matches)[:2])


def _window_around_keywords(question: str, text: str) -> str | None:
    """截取第一个问题关键词前后的一段文字；无关键词时退回开头。"""

    keywords = _question_keywords(question)
    if not keywords:
        return text[:260]

    best_pos = -1
    for keyword in keywords:
        pos = text.find(keyword)
        if pos >= 0:
            best_pos = pos
            break

    if best_pos < 0:
        return None

    start = max(0, best_pos - 80)
    end = min(len(text), best_pos + 280)
    return text[start:end]


def _question_keywords(question: str) -> list[str]:
    """与遗留财务/研报场景匹配的关键词表。"""

    candidates = [
        "营业收入",
        "营收",
        "上半年",
        "归母净利润",
        "净利润",
        "毛利率",
        "客户",
        "PCB",
        "投资建议",
        "风险提示",
    ]
    return [item for item in candidates if item in question]


def _clean_text(text: str) -> str:
    """压缩空白、去掉 PDF 装饰符，并修复数字之间被拆开的空格。"""

    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"＃[^＃]{0,40}＃", "", text)
    text = re.sub(r"([0-9])\s+([0-9])", r"\1\2", text)
    return text


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def stream_answer(question: str, evidence: list[dict]) -> Generator[str, None, None]:
    """以 SSE 片段流式返回文本问答结果，最后补发证据和结束标记。"""

    if not evidence:
        message = fallback_answer(question, evidence)
        yield f"data: {json.dumps({'type': 'answer_delta', 'content': message}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'evidence', 'items': []}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        return

    if not _llm_api_key():
        message = fallback_answer(question, evidence)
        yield f"data: {json.dumps({'type': 'answer_delta', 'content': message}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'evidence', 'items': evidence}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        return

    client = OpenAI(api_key=_llm_api_key(), base_url=_llm_base_url())
    prompt = build_grounded_prompt(question, evidence)
    completion = client.chat.completions.create(
        model=_llm_model(),
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in completion:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield f"data: {json.dumps({'type': 'answer_delta', 'content': content}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'evidence', 'items': evidence}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


def complete_answer(question: str, evidence: list[dict]) -> str:
    """一次性返回完整回答，供图片问答等 JSON 接口使用。"""

    if not _llm_api_key():
        return fallback_answer(question, evidence)

    client = OpenAI(api_key=_llm_api_key(), base_url=_llm_base_url())
    prompt = build_grounded_prompt(question, evidence)
    completion = client.chat.completions.create(
        model=_llm_model(),
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return completion.choices[0].message.content or fallback_answer(question, evidence)
