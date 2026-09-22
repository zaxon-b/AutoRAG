# 模块说明：答案生成服务：把 evidence 组织成 grounded prompt，调用 LLM，或在无 key 时用规则兜底回答。
# 调用关系：由 qa_rt 文本/图片问答调用。
# 返回/产出：返回流式 SSE 片段或完整回答文本。
# 代码说明：导入模块，json：读写 JSON/JSONL 数据，例如 VLM 输出、评测样本和 evidence。
import json
# 代码说明：导入模块，os：读取环境变量，例如模型 key、阈值、数据库地址和存储目录。
import os
# 代码说明：导入模块，re：用正则表达式从 VLM 输出或 caption 中提取图标名称。
import re
# 代码说明：从 collections.abc 导入：Generator：本文件后续逻辑依赖的对象。
from collections.abc import Generator

# 代码说明：从 openai 导入：OpenAI：本文件后续逻辑依赖的对象。
from openai import OpenAI


# 函数说明：_llm_api_key：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _llm_api_key：封装当前模块的一段业务逻辑，供其他流程复用。
def _llm_api_key() -> str | None:
    # 代码说明：返回调用结果：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里，调用方会拿这个结果继续下一步。
    return os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


# 函数说明：_llm_base_url：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _llm_base_url：封装当前模块的一段业务逻辑，供其他流程复用。
def _llm_base_url() -> str | None:
    # 代码说明：返回调用结果：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里，调用方会拿这个结果继续下一步。
    return os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")


# 函数说明：_llm_model：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _llm_model：封装当前模块的一段业务逻辑，供其他流程复用。
def _llm_model() -> str:
    # 代码说明：返回调用结果：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里，调用方会拿这个结果继续下一步。
    return os.getenv("LLM_MODEL", "deepseek-r1")


# 函数说明：build_grounded_prompt：把用户问题和 evidence 拼成要求“只基于证据回答”的 LLM prompt。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 build_grounded_prompt：把用户问题和 evidence 拼成要求“只基于证据回答”的 LLM prompt。
def build_grounded_prompt(question: str, evidence: list[dict]) -> str:
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
    role_instructions = (
        "\n- role=visual 的唯一职责是确认用户图片对应手册中的什么视觉对象。"
        "\n- role=text 的职责是说明该对象的含义、使用方法、触发条件和处理建议。"
        "\n- 不得仅凭视觉相似度补写正文没有支持的确定结论。"
        if has_visual_evidence
        else ""
    )
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
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


# 函数说明：fallback_answer：没有 LLM API 或调用失败时，根据证据文本用规则生成兜底回答。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 fallback_answer：没有 LLM API 或调用失败时，根据证据文本用规则生成兜底回答。
def fallback_answer(question: str, evidence: list[dict]) -> str:
    # 代码说明：判断条件 `not evidence` 是否成立，成立时执行下面缩进代码。
    if not evidence:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return "当前证据不足，无法可靠回答这个问题。请确认已上传正确文档，或换一个更具体的问题。"

    # 代码说明：snippets 表示snippets；本行右侧表达式负责得到这个值。
    snippets = _extract_snippets(question, evidence)
    # 代码说明：判断条件 `snippets` 是否成立，成立时执行下面缩进代码。
    if snippets:
        # 代码说明：joined：joined 是把多个文字片段拼接后的完整标签文本。
        joined = "；".join(snippets)
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return f"根据当前检索到的证据，{joined}。"

    # 代码说明：first 表示first；本行右侧表达式负责得到这个值。
    first = evidence[0]
    # 代码说明：page：候选证据所在手册页码；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    page = first.get("page")
    # 代码说明：text：text 保存当前正在清洗、匹配或写入 chunk 的文本内容。
    text = _clean_text(first.get("content_text") or "")
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return f"根据当前检索到的证据，最相关内容出现在第 {page} 页：{text[:360]}。"


# 函数说明：_extract_snippets：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _extract_snippets：封装当前模块的一段业务逻辑，供其他流程复用。
def _extract_snippets(question: str, evidence: list[dict]) -> list[str]:
    # 代码说明：question：用户输入的问题文本；图片问答中它是可选补充问题；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    question = _clean_text(question)
    # 代码说明：revenue_question 表示revenue / question；这里调用 `search` 得到它，调用目的：用正则在 VLM reason 或 caption 中提取图标名称。
    revenue_question = bool(re.search(r"营收|营业收入|收入", question))
    # 代码说明：snippets 表示snippets；本行右侧表达式负责得到这个值。
    snippets: list[str] = []
    # 代码说明：fallback_snippets 表示fallback / snippets；本行右侧表达式负责得到这个值。
    fallback_snippets: list[str] = []

    # 代码说明：循环遍历 enumerate(evidence[，每轮把当前元素放到 index, item 中处理。
    for index, item in enumerate(evidence[:3], start=1):
        # 代码说明：page：候选证据所在手册页码；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
        page = item.get("page")
        # 代码说明：text：text 保存当前正在清洗、匹配或写入 chunk 的文本内容。
        text = _clean_text(item.get("content_text") or "")
        # 代码说明：判断条件 `not text` 是否成立，成立时执行下面缩进代码。
        if not text:
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue

        # 代码说明：revenue 表示revenue；本行右侧表达式负责得到这个值。
        revenue = _extract_revenue_answer(question, text)
        # 代码说明：判断条件 `revenue` 是否成立，成立时执行下面缩进代码。
        if revenue:
            # 代码说明：把当前结果追加到 snippets 列表中；这表示它会参与后续排序、融合、返回或统计。
            snippets.append(f"{revenue}（证据 {index}，第 {page} 页）")
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue

        # 代码说明：判断条件 `revenue_question` 是否成立，成立时执行下面缩进代码。
        if revenue_question:
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue

        # 代码说明：focused 表示focused；本行右侧表达式负责得到这个值。
        focused = _window_around_keywords(question, text)
        # 代码说明：判断条件 `focused` 是否成立，成立时执行下面缩进代码。
        if focused:
            # 代码说明：把当前结果追加到 fallback_snippets 列表中；这表示它会参与后续排序、融合、返回或统计。
            fallback_snippets.append(f"{focused}（证据 {index}，第 {page} 页）")

    # 代码说明：判断条件 `snippets` 是否成立，成立时执行下面缩进代码。
    if snippets:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return _dedupe(snippets)[:3]
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return _dedupe(fallback_snippets)[:3]


# 函数说明：_extract_revenue_answer：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _extract_revenue_answer：封装当前模块的一段业务逻辑，供其他流程复用。
def _extract_revenue_answer(question: str, text: str) -> str | None:
    # 代码说明：判断条件：用正则在 VLM reason 或 caption 中提取图标名称；只有判断通过才执行下面分支。
    if not re.search(r"营收|营业收入|收入", question):
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None

    # 代码说明：patterns 表示patterns；本行右侧表达式负责得到这个值。
    patterns = [
        # 代码说明：这是上一行多行表达式的一部分，用来继续补充筛选条件、字段值或函数参数。
        r"(?:公司发布\s*)?(?:2023年中报，?)?(?:上半年)?[^。；]{0,30}实现营业收入\s*[\d.]+\s*亿元\s*，?\s*同比(?:下降|增长|降低|提升)\s*[\d.]+%",
        # 代码说明：这是上一行多行表达式的一部分，用来继续补充筛选条件、字段值或函数参数。
        r"(?:其中)?(?:第二季度|Q2)[^。；]{0,30}营业收入\s*[\d.]+\s*亿元\s*，?\s*同比(?:下降|增长|降低|提升)\s*[\d.]+%",
        # 代码说明：这是上一行多行表达式的一部分，用来继续补充筛选条件、字段值或函数参数。
        r"营业收入\s*[\d.]+\s*亿元",
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    ]
    # 代码说明：matches 表示matches；本行右侧表达式负责得到这个值。
    matches: list[str] = []
    # 代码说明：循环遍历 patterns，每轮把当前元素放到 pattern 中处理。
    for pattern in patterns:
        # 代码说明：把一组结果批量追加到 matches 中，用来合并多个来源的数据。
        matches.extend(re.findall(pattern, text))

    # 代码说明：判断条件 `not matches` 是否成立，成立时执行下面缩进代码。
    if not matches:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None

    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return "；".join(_dedupe(matches)[:2])


# 函数说明：_window_around_keywords：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _window_around_keywords：封装当前模块的一段业务逻辑，供其他流程复用。
def _window_around_keywords(question: str, text: str) -> str | None:
    # 代码说明：keywords 表示关键词集合；本行右侧表达式负责得到这个值。
    keywords = _question_keywords(question)
    # 代码说明：判断条件 `not keywords` 是否成立，成立时执行下面缩进代码。
    if not keywords:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return text[:260]

    # 代码说明：best_pos 表示best / pos；本行右侧表达式负责得到这个值。
    best_pos = -1
    # 代码说明：循环遍历 keywords，每轮把当前元素放到 keyword 中处理。
    for keyword in keywords:
        # 代码说明：pos 表示pos；本行右侧表达式负责得到这个值。
        pos = text.find(keyword)
        # 代码说明：判断条件 `pos >= 0` 是否成立，成立时执行下面缩进代码。
        if pos >= 0:
            # 代码说明：best_pos 表示best / pos；本行右侧表达式负责得到这个值。
            best_pos = pos
            # 代码说明：导入 break：供本模块后续代码直接调用。
            break

    # 代码说明：判断条件 `best_pos < 0` 是否成立，成立时执行下面缩进代码。
    if best_pos < 0:
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return None

    # 代码说明：start 表示start；本行右侧表达式负责得到这个值。
    start = max(0, best_pos - 80)
    # 代码说明：end 表示end；本行右侧表达式负责得到这个值。
    end = min(len(text), best_pos + 280)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return text[start:end]


# 函数说明：_question_keywords：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _question_keywords：封装当前模块的一段业务逻辑，供其他流程复用。
def _question_keywords(question: str) -> list[str]:
    # 代码说明：candidates：拼接后的 VLM 候选名称文本，用来写入 LLM 内部问题；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    candidates = [
        # 代码说明：`"营业收入"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "营业收入",
        # 代码说明：`"营收"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "营收",
        # 代码说明：`"上半年"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "上半年",
        # 代码说明：`"归母净利润"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "归母净利润",
        # 代码说明：`"净利润"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "净利润",
        # 代码说明：`"毛利率"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "毛利率",
        # 代码说明：`"客户"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "客户",
        # 代码说明：`"PCB"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "PCB",
        # 代码说明：`"投资建议"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "投资建议",
        # 代码说明：`"风险提示"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "风险提示",
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    ]
    # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
    return [item for item in candidates if item in question]


# 函数说明：_clean_text：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _clean_text：封装当前模块的一段业务逻辑，供其他流程复用。
def _clean_text(text: str) -> str:
    # 代码说明：text：text 保存当前正在清洗、匹配或写入 chunk 的文本内容。
    text = re.sub(r"\s+", " ", text or "").strip()
    # 代码说明：text：text 保存当前正在清洗、匹配或写入 chunk 的文本内容。
    text = re.sub(r"＃[^＃]{0,40}＃", "", text)
    # 代码说明：text：text 保存当前正在清洗、匹配或写入 chunk 的文本内容。
    text = re.sub(r"([0-9])\s+([0-9])", r"\1\2", text)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return text


# 函数说明：_dedupe：封装当前模块的一段业务逻辑，供其他流程调用。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _dedupe：封装当前模块的一段业务逻辑，供其他流程复用。
def _dedupe(items: list[str]) -> list[str]:
    # 代码说明：result：当前步骤的处理结果，具体含义由调用函数决定；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    result: list[str] = []
    # 代码说明：seen 表示seen；本行右侧表达式负责得到这个值。
    seen: set[str] = set()
    # 代码说明：循环遍历 items，每轮把当前元素放到 item 中处理。
    for item in items:
        # 代码说明：normalized：VLM 生成的标准化检索句，通常比用户原问题更适合搜索手册；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
        normalized = item.strip()
        # 代码说明：判断条件 `not normalized or normalized in seen` 是否成立，成立时执行下面缩进代码。
        if not normalized or normalized in seen:
            # 代码说明：导入 continue：供本模块后续代码直接调用。
            continue
        # 代码说明：把当前 ID/名称加入 seen 集合，用于去重或快速判断是否已经处理过。
        seen.add(normalized)
        # 代码说明：把当前结果追加到 result 列表中；这表示它会参与后续排序、融合、返回或统计。
        result.append(normalized)
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return result


# 函数说明：stream_answer：流式生成文本问答答案，并在最后通过 SSE 返回 evidence。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 stream_answer：流式生成文本问答答案，并在最后通过 SSE 返回 evidence。
def stream_answer(question: str, evidence: list[dict]) -> Generator[str, None, None]:
    # 代码说明：判断条件 `not evidence` 是否成立，成立时执行下面缩进代码。
    if not evidence:
        # 代码说明：message 表示消息；本行右侧表达式负责得到这个值。
        message = fallback_answer(question, evidence)
        # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
        yield f"data: {json.dumps({'type': 'answer_delta', 'content': message}, ensure_ascii=False)}\n\n"
        # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
        yield f"data: {json.dumps({'type': 'evidence', 'items': []}, ensure_ascii=False)}\n\n"
        # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        # 代码说明：导入 return：供本模块后续代码直接调用。
        return

    # 代码说明：判断条件 `not _llm_api_key()` 是否成立，成立时执行下面缩进代码。
    if not _llm_api_key():
        # 代码说明：message 表示消息；本行右侧表达式负责得到这个值。
        message = fallback_answer(question, evidence)
        # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
        yield f"data: {json.dumps({'type': 'answer_delta', 'content': message}, ensure_ascii=False)}\n\n"
        # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
        yield f"data: {json.dumps({'type': 'evidence', 'items': evidence}, ensure_ascii=False)}\n\n"
        # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        # 代码说明：导入 return：供本模块后续代码直接调用。
        return

    # 代码说明：client：外部服务客户端，例如 Elasticsearch 或 OpenAI 兼容 API 客户端；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    client = OpenAI(api_key=_llm_api_key(), base_url=_llm_base_url())
    # 代码说明：prompt 表示提示词；本行右侧表达式负责得到这个值。
    prompt = build_grounded_prompt(question, evidence)
    # 代码说明：completion 表示模型补全结果；本行右侧表达式负责得到这个值。
    completion = client.chat.completions.create(
        # 代码说明：model 表示模型名；本行右侧表达式负责得到这个值。
        model=_llm_model(),
        # 代码说明：messages 表示消息列表；本行右侧表达式负责得到这个值。
        messages=[{"role": "user", "content": prompt}],
        # 代码说明：stream 表示stream；本行右侧表达式负责得到这个值。
        stream=True,
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    )

    # 代码说明：循环遍历 completion，每轮把当前元素放到 chunk 中处理。
    for chunk in completion:
        # 代码说明：delta 表示delta；本行右侧表达式负责得到这个值。
        delta = chunk.choices[0].delta
        # 代码说明：content：用户上传图片的二进制内容，保存到本地后才能被 VLM、embedding 和 OpenCV 读取；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
        content = getattr(delta, "content", None)
        # 代码说明：判断条件 `content` 是否成立，成立时执行下面缩进代码。
        if content:
            # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
            yield f"data: {json.dumps({'type': 'answer_delta', 'content': content}, ensure_ascii=False)}\n\n"
    # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
    yield f"data: {json.dumps({'type': 'evidence', 'items': evidence}, ensure_ascii=False)}\n\n"
    # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


# 函数说明：complete_answer：一次性生成完整回答，图片问答接口用它返回 JSON。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 complete_answer：一次性生成完整回答，图片问答接口用它返回 JSON。
def complete_answer(question: str, evidence: list[dict]) -> str:
    # 代码说明：判断条件 `not _llm_api_key()` 是否成立，成立时执行下面缩进代码。
    if not _llm_api_key():
        # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
        return fallback_answer(question, evidence)

    # 代码说明：client：外部服务客户端，例如 Elasticsearch 或 OpenAI 兼容 API 客户端；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    client = OpenAI(api_key=_llm_api_key(), base_url=_llm_base_url())
    # 代码说明：prompt 表示提示词；本行右侧表达式负责得到这个值。
    prompt = build_grounded_prompt(question, evidence)
    # 代码说明：completion 表示模型补全结果；本行右侧表达式负责得到这个值。
    completion = client.chat.completions.create(
        # 代码说明：model 表示模型名；本行右侧表达式负责得到这个值。
        model=_llm_model(),
        # 代码说明：messages 表示消息列表；本行右侧表达式负责得到这个值。
        messages=[{"role": "user", "content": prompt}],
        # 代码说明：stream 表示stream；本行右侧表达式负责得到这个值。
        stream=False,
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    )
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return completion.choices[0].message.content or fallback_answer(question, evidence)
