/**
 * 模块：后端业务状态拦截器
 *
 * HTTP 200 只说明请求成功到达并返回；有些后端还会在 JSON 中用 status 表达
 * 业务成功或失败。本插件识别 `{ status: 'error', message: '...' }`，把它转换成
 * ResponseError，使页面和错误 toast 可以走统一的 Promise catch 流程。
 *
 * 当前项目的手册响应也用 status 表示建库状态（例如 ready）。除 success/error
 * 外的取值会原样通过，因此这个字段同时承担两种语义，是后续应拆分的契约风险。
 */

import { AxiosResponse } from 'axios'
import { ResponseError } from '../error'
import { IRequestPlugin } from './plugin'

export const CODE_KEY = 'status'
export const MESSAGE_KEY = 'message'

type ApiRecord = Record<string, unknown>

export const servicePlugin: IRequestPlugin = {
  install(instance) {
    instance.interceptors.response.use(
      (response) => {
        const data: unknown = response.data

        // 文件、字符串或数组响应没有统一 status 对象，直接交给业务调用方。
        if (!isRecord(data) || !(CODE_KEY in data)) return response

        const status = data[CODE_KEY]

        // 明确成功时保持 AxiosResponse 结构，业务数据仍通过 response.data 读取。
        if (status === 'success') return response

        if (status === 'error') {
          return Promise.reject(
            new ResponseError(readErrorMessage(data), response),
          )
        }

        // ready、empty 等手册状态不是通用业务错误，不能在这里擅自拒绝。
        return response
      },
      (error: unknown) => {
        // HTTP 4xx/5xx 会进入失败拦截器；只有拿得到对象响应体时才检查业务 status。
        const response = responseFromError(error)
        const data: unknown = response?.data

        if (!response || !isRecord(data)) return Promise.reject(error)

        if (data[CODE_KEY] === 'error') {
          return Promise.reject(
            new ResponseError(readErrorMessage(data), response),
          )
        }

        // 不认识的错误保持原样，留给后面的 errorToastPlugin 或页面处理。
        return Promise.reject(error)
      },
    )
  },
}

/** unknown 只有通过这个类型守卫后，才能安全按键读取字段。 */
function isRecord(value: unknown): value is ApiRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** 按后端常见字段顺序提取可展示的字符串错误。 */
function readErrorMessage(data: ApiRecord) {
  const message = data[MESSAGE_KEY] ?? data.detail
  return typeof message === 'string' && message ? message : '接口数据异常'
}

/**
 * Axios 的失败对象通常带 response，但网络断开时没有。先做形状判断，避免为了
 * 读取 response 而把整个 error 写成 any。
 */
function responseFromError(error: unknown) {
  if (typeof error !== 'object' || error === null || !('response' in error)) {
    return undefined
  }

  return error.response as AxiosResponse<unknown> | undefined
}
