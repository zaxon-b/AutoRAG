# 模块说明：文本 embedding 服务：把中文文本转成向量，供 ES 做语义相似度检索。
# 调用关系：由建库和检索流程调用。
# 返回/产出：返回固定维度文本向量；没有 API key 时返回确定性兜底向量。
# 代码说明：导入模块，hashlib：用哈希生成确定性兜底向量或稳定 ID。
import hashlib
# 代码说明：导入模块，os：读取环境变量，例如模型 key、阈值、数据库地址和存储目录。
import os
# 代码说明：导入模块，random：提供本模块需要的标准库/第三方能力。
import random

# 代码说明：从 openai 导入：OpenAI：本文件后续逻辑依赖的对象。
from openai import OpenAI


# 代码说明：EMBEDDING_DIM 表示文本 embedding 向量维度，必须和 ES dense_vector mapping 保持一致；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
# 代码说明：TEXT_EMBEDDING_MODEL 表示文本 embedding 模型名称，例如 DashScope 的 text-embedding-v3；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-v3")
# 代码说明：TEXT_EMBEDDING_BATCH_SIZE 表示文本向量批处理大小，控制一次 API 请求送多少条文本；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
TEXT_EMBEDDING_BATCH_SIZE = int(os.getenv("TEXT_EMBEDDING_BATCH_SIZE", "10"))


# 函数说明：deterministic_embedding：用哈希方式生成固定伪向量，在没有 embedding API key 时保证检索流程不崩。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 deterministic_embedding：用哈希方式生成固定伪向量，在没有 embedding API key 时保证检索流程不崩。
def deterministic_embedding(text: str, dimensions: int = EMBEDDING_DIM) -> list[float]:
    # 代码说明：seed 表示seed；本行右侧表达式负责得到这个值。
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    # 代码说明：rng 表示rng；本行右侧表达式负责得到这个值。
    rng = random.Random(seed)
    # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
    return [rng.uniform(-0.01, 0.01) for _ in range(dimensions)]


# 函数说明：embed_texts：批量生成文本 embedding，优先调用 DashScope/OpenAI 兼容接口，失败时使用确定性兜底向量。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 embed_texts：批量生成文本 embedding，优先调用 DashScope/OpenAI 兼容接口，失败时使用确定性兜底向量。
def embed_texts(texts: list[str]) -> list[list[float]]:
    # 代码说明：判断条件 `not texts` 是否成立，成立时执行下面缩进代码。
    if not texts:
        # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
        return []
    # 代码说明：判断条件：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里；只有判断通过才执行下面分支。
    if not os.getenv("DASHSCOPE_API_KEY"):
        # 代码说明：返回列表结果：调用方会把这些元素作为候选、chunk 或指标继续处理。
        return [deterministic_embedding(text) for text in texts]

    # 代码说明：vectors：embedding 模型输出的向量列表；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    vectors = _generate_dashscope_embeddings(texts)
    # 代码说明：result：当前步骤的处理结果，具体含义由调用函数决定；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    result: list[list[float]] = []
    # 代码说明：循环遍历 zip(texts, vectors or [])，每轮把当前元素放到 text, vector 中处理。
    for text, vector in zip(texts, vectors or []):
        # 代码说明：判断条件 `vector` 是否成立，成立时执行下面缩进代码。
        if vector:
            # 代码说明：把当前结果追加到 result 列表中；这表示它会参与后续排序、融合、返回或统计。
            result.append(vector)
        # 代码说明：兜底分支：前面的条件都不满足时执行这里，通常用于默认结果或失败处理。
        else:
            # 代码说明：把当前结果追加到 result 列表中；这表示它会参与后续排序、融合、返回或统计。
            result.append(deterministic_embedding(text))

    # 代码说明：条件循环：只要 `len(result) < len(texts)` 成立，就继续重复执行下面逻辑。
    while len(result) < len(texts):
        # 代码说明：把当前结果追加到 result 列表中；这表示它会参与后续排序、融合、返回或统计。
        result.append(deterministic_embedding(texts[len(result)]))
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return result


# 函数说明：embed_text：为单条文本生成 embedding，是 embed_texts 的便捷封装。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 embed_text：为单条文本生成 embedding，是 embed_texts 的便捷封装。
def embed_text(text: str) -> list[float]:
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return embed_texts([text])[0]


# 函数说明：_generate_dashscope_embeddings：调用 DashScope/OpenAI 兼容 embedding API 生成真正语义向量。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 _generate_dashscope_embeddings：调用 DashScope/OpenAI 兼容 embedding API 生成真正语义向量。
def _generate_dashscope_embeddings(texts: list[str]) -> list[list[float] | None]:
    # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
    """Call DashScope through the OpenAI-compatible embeddings API.

    Keeping this small wrapper local lets the demo delete the old generic RAG
    core while preserving the same text-embedding-v3 behavior.
    """
    # 代码说明：client：外部服务客户端，例如 Elasticsearch 或 OpenAI 兼容 API 客户端；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    client = OpenAI(
        # 代码说明：api_key 表示API / 密钥；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 代码说明：base_url 表示基础 / 地址；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    )
    # 代码说明：embeddings 表示向量集合；本行右侧表达式负责得到这个值。
    embeddings: list[list[float] | None] = []
    # 代码说明：循环遍历 range(0, len(texts), TEXT_EMBEDDING_BATCH_SIZE)，每轮把当前元素放到 start 中处理。
    for start in range(0, len(texts), TEXT_EMBEDDING_BATCH_SIZE):
        # 代码说明：batch 表示batch；本行右侧表达式负责得到这个值。
        batch = texts[start : start + TEXT_EMBEDDING_BATCH_SIZE]
        # 代码说明：进入异常保护块：下面的外部接口、文件、数据库或模型调用可能失败，需要捕获错误。
        try:
            # 代码说明：response：HTTP/API/模型调用返回对象，需要进一步解析状态码和响应体；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
            response = client.embeddings.create(
                # 代码说明：model 表示模型名；本行右侧表达式负责得到这个值。
                model=TEXT_EMBEDDING_MODEL,
                # 代码说明：input 表示input；本行右侧表达式负责得到这个值。
                input=batch,
                # 代码说明：dimensions 表示dimensions；本行右侧表达式负责得到这个值。
                dimensions=EMBEDDING_DIM,
                # 代码说明：encoding_format 表示encoding / format；本行右侧表达式负责得到这个值。
                encoding_format="float",
            # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
            )
            # 代码说明：把一组结果批量追加到 embeddings 中，用来合并多个来源的数据。
            embeddings.extend([item.embedding for item in response.data])
        # 代码说明：捕获异常 `Exception`，把底层错误转换成当前模块可处理的返回或提示。
        except Exception:
            # 代码说明：把一组结果批量追加到 embeddings 中，用来合并多个来源的数据。
            embeddings.extend([None for _ in batch])
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return embeddings
