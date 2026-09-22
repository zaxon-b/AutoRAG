/**
 * 模块：Axios 全局错误提示
 *
 * 这是请求流水线最后的用户提示层。前面的 servicePlugin 可能把业务失败转换成
 * ResponseError；HTTP 4xx/5xx 和网络错误也会进入这里。本插件选择最有用的错误
 * 文字，并通过 Ant Design message 在页面顶部短暂显示。
 *
 * 调用方可设置 `errorToast: false` 关闭全局提示，再由页面 Alert 单独展示，避免
 * 同一个错误同时弹 toast 和页面错误条。
 */

import { AxiosRequestConfig, AxiosResponse, CanceledError } from 'axios'
import { ResponseError } from '../error'
import { IRequestPlugin } from './plugin'

// 这里只覆盖需要友好中文提示的特殊状态；其他状态优先使用后端 detail/message。
const NETWORK_ERROR_MAP: Partial<Record<number, string>> = {
  429: '请求过于频繁，请稍后再试',
}

type ApiRecord = Record<string, unknown>

export const errorToastPlugin: IRequestPlugin = {
  postinstall(instance) {
    instance.interceptors.response.use(
      // 成功响应无需处理，保持原 AxiosResponse 返回给业务层。
      (response) => response,
      (error: unknown) => {
        const response = responseFromError(error)
        const config = configFromError(error, response)

        // 明确关闭 toast 的请求，由调用页面自己决定怎样展示错误。
        if (config && !config.errorToast) return Promise.reject(error)

        // repeatPlugin 主动取消旧请求是正常控制流程，不应该吓到用户。
        if (error instanceof CanceledError) return Promise.reject(error)

        const data = asRecord(response?.data)
        const message =
          (error instanceof ResponseError ? error.message : undefined) ||
          (response ? NETWORK_ERROR_MAP[response.status] : undefined) ||
          readString(data, 'message') ||
          readString(data, 'detail') ||
          readString(data, 'error') ||
          (error instanceof Error ? error.message : undefined) ||
          '请求错误'

        // window.$app 由 App.tsx 的 MountApi 在 React 上下文中挂载。
        window.$app.message.error(message)

        // 显示提示后仍继续 reject，让调用方 catch 有机会更新页面状态或日志。
        return Promise.reject(error)
      },
    )
  },
}

function asRecord(value: unknown): ApiRecord | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as ApiRecord)
    : undefined
}

/** 只接受真实字符串，避免把对象直接渲染成 [object Object]。 */
function readString(data: ApiRecord | undefined, key: string) {
  const value = data?.[key]
  return typeof value === 'string' && value ? value : undefined
}

function responseFromError(error: unknown) {
  if (typeof error !== 'object' || error === null || !('response' in error)) {
    return undefined
  }
  return error.response as AxiosResponse<unknown> | undefined
}

function configFromError(
  error: unknown,
  response: AxiosResponse<unknown> | undefined,
): AxiosRequestConfig | undefined {
  if (response?.config) return response.config

  if (typeof error === 'object' && error !== null && 'config' in error) {
    return error.config as AxiosRequestConfig | undefined
  }

  return undefined
}
