/**
 * 模块：文本问答和图片问答 API
 *
 * 谁调用它：App.tsx 中的 askText 和 askImage。
 * 为什么这里使用 fetch：文本接口返回 POST SSE 流，需要直接读取浏览器的
 * ReadableStream；图片接口与它属于同一业务模块，也沿用 fetch。
 *
 * 两个接口的响应方式不同：
 * - POST /qa/text：连续返回 SSE 事件，本模块解析一条就调用一次 onEvent；
 * - POST /qa/image：整条视觉链路结束后一次性返回 ImageQaResult JSON。
 *
 * 重要边界：fetch 遇到 HTTP 400/500 时不会自动抛异常，所以两个函数都必须
 * 主动检查 response.ok。网络断开和 JSON/SSE 解析失败则会自然抛给页面 catch。
 */

import { ImageQaResult, TextQaEvent } from '@/types/manual'

// Vite 在启动/构建时注入后端地址，例如 http://localhost:8000。
// 未配置时使用空字符串，表示请求当前页面同源下的 /qa/... 路径。
const API_BASE = import.meta.env.VITE_API_BASE || ''

type TextQaParams = {
  /** 在哪本已建库手册中检索。 */
  manual_id: string
  /** 用户输入并已去掉首尾空格的问题。 */
  question: string
  /** 每解析出一条服务端事件就回调页面，由页面决定怎样更新 state。 */
  onEvent: (event: TextQaEvent) => void
}

/**
 * 发起文本 RAG 问答，并消费后端返回的 SSE 字节流。
 *
 * 函数本身不保存 answer；它只负责通信和协议解析。这样 API 层不依赖 React，
 * 页面可以在 onEvent 中用 setAnswer 和 setEvidence 更新自己的状态。
 */
export async function textQa(params: TextQaParams) {
  const response = await fetch(`${API_BASE}/qa/text`, {
    method: 'POST',
    headers: {
      // 文本请求不含文件，用 JSON 与 FastAPI 的 TextQaRequest 对应。
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      manual_id: params.manual_id,
      question: params.question,
    }),
  })

  // fetch 只有网络层失败才自动 reject；HTTP 错误必须由业务代码检查。
  if (!response.ok) {
    throw new Error(await response.text())
  }

  // response.body 是浏览器的可读字节流。极少数环境可能没有 body，此时无法流式读取。
  const reader = response.body?.getReader()
  if (!reader) return

  // 网络传来 Uint8Array，需要按 UTF-8 解码后才能处理中文和 JSON。
  const decoder = new TextDecoder('utf-8')

  // 一次 read() 不保证恰好是一条 SSE：事件可能被拆成多块，也可能多条合在一块。
  // buffer 保存“已经收到、但还没有形成完整空行分隔事件”的尾部文字。
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()

    // done=true 表示服务端已经关闭响应流，不会再有新字节。
    if (done) break

    // stream=true 让解码器保留可能被切到两块中的多字节 UTF-8 字符。
    buffer += decoder.decode(value, { stream: true })

    // SSE 用空行分隔事件。最后一段可能不完整，先留在 buffer 等下一批字节。
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''

    for (const event of events) {
      // 当前后端把每条业务 JSON 放在 data: 行；其他 SSE 字段暂不参与页面逻辑。
      const dataLine = event
        .split('\n')
        .find((line) => line.startsWith('data: '))

      if (!dataLine) continue

      // 去掉 SSE 协议前缀，留下 JSON 字符串或结束标记。
      const payload = dataLine.replace(/^data:\s*/, '')

      // [DONE] 只表示流结束，不是 JSON，不能交给 JSON.parse。
      if (!payload || payload === '[DONE]') continue

      // JSON 解析失败会抛到 App.tsx 的 catch，避免把损坏事件当成正常回答。
      params.onEvent(JSON.parse(payload) as TextQaEvent)
    }
  }
}

type ImageQaParams = {
  /** 在哪本手册中寻找视觉锚点和正文证据。 */
  manual_id: string
  /** 用户选择的浏览器原生图片 File。 */
  image: File
  /** 用户可选的补充问题；为空时后端根据图片构造默认问题。 */
  question?: string
}

/**
 * 上传图片并等待完整图片问答结果。
 *
 * 图片是二进制文件，所以使用 FormData/multipart，而不是 JSON。后端完成 VLM
 * 初判、图片/多模态向量双路召回、等权 RRF、正文检索和回答后才返回 JSON。
 * 候选改选式 VLM 二验默认关闭，只用于离线实验。
 */
export async function imageQa(params: ImageQaParams) {
  const form = new FormData()

  // 这三个名字必须与 FastAPI image_qa 的 Form/File 参数完全一致。
  form.append('manual_id', params.manual_id)
  form.append('image', params.image)

  // 可选问题为空时不添加字段，让后端使用 question=None 的默认值。
  if (params.question) {
    form.append('question', params.question)
  }

  const response = await fetch(`${API_BASE}/qa/image`, {
    method: 'POST',
    body: form,
    // 不手写 Content-Type：浏览器会为 FormData 自动添加正确的 multipart boundary。
  })

  if (!response.ok) {
    throw new Error(await response.text())
  }

  // `as` 只提供编译期类型提示，不会在运行时校验响应字段。
  return (await response.json()) as ImageQaResult
}
