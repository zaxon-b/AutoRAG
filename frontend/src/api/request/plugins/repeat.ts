/**
 * 模块：Axios 重复请求取消
 *
 * 当 `cancelRepeat` 开启时，如果相同 key 的新请求到来，本插件会取消旧请求，
 * 避免用户快速连点导致旧响应晚到并覆盖新结果。
 *
 * 当前 key 默认只包含 method、url 和可选 repeatKey，不自动比较 query 或请求体。
 * 因此同一路径的不同业务参数若允许并发，调用方必须提供不同 repeatKey。
 */

import { AxiosRequestConfig } from 'axios'
import { IRequestPlugin } from './plugin'

/**
 * signal 只能负责传递取消状态，真正执行 abort 的方法在 controller 上。
 * WeakMap 建立 signal -> controller 的反向关系，并允许不再使用的 signal 被回收。
 */
const abortControllerMap = new WeakMap<AbortSignal, AbortController>()

/** 创建一对 AbortController/signal，并保存反向关系。 */
export function createAbortController() {
  const controller = new AbortController()
  abortControllerMap.set(controller.signal, controller)
  return controller
}

/** 保存每个重复 key 当前最后一个未清理的请求配置。 */
const pendingRequests = new Map<string, AxiosRequestConfig>()

function setPending(config: AxiosRequestConfig) {
  const key = getRepeatKey(config)
  // Axios 为兼容不同运行环境使用 GenericAbortSignal；浏览器端实际保存的是
  // createAbortController 产生的标准 AbortSignal，所以在查 WeakMap 前收窄类型。
  const previousSignal = pendingRequests.get(key)?.signal as
    | AbortSignal
    | undefined

  if (previousSignal) {
    // 新请求替代旧请求。Axios 会把 abort 转换成 CanceledError，错误插件不会弹 toast。
    abortControllerMap.get(previousSignal)?.abort('取消重复请求')
  }

  pendingRequests.set(key, config)
}

/** 请求完成后删除当前 key，避免后续正常请求误认为它仍在进行。 */
export function remove(config: AxiosRequestConfig) {
  pendingRequests.delete(getRepeatKey(config))
}

/**
 * 生成重复请求标识。
 * 例：`get-/manuals/status-`。注意它没有包含 data/params，这是当前实现的边界。
 */
export function getRepeatKey(config: AxiosRequestConfig) {
  return `${config.method}-${config.url}-${config.repeatKey ?? ''}`
}

export const repeatPlugin: IRequestPlugin = {
  // 成功响应到达后清理对应 key。
  preinstall(instance) {
    instance.interceptors.response.use((response) => {
      const config = response.config as AxiosRequestConfig
      if (!config.cancelRepeat) return response

      remove(config)
      return response
    })
  },

  // 请求发出前准备 signal，并取消同 key 的旧请求。
  postinstall(instance) {
    instance.interceptors.request.use((config) => {
      if (!config.cancelRepeat) return config

      // 尊重调用方已有 signal；只有没提供时才创建项目自己的 controller。
      config.signal = config.signal ?? createAbortController().signal
      setPending(config)
      return config
    })
  },
}
