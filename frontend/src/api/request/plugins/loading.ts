/**
 * 模块：Axios 全屏 loading 拦截器
 *
 * 请求配置 `loading: true` 时，本插件在发请求前打开全屏 Spin，在收到成功或
 * 失败响应后关闭。真正的 React state 位于 App.tsx 的 MountApi；这里通过
 * window.$showLoading / $hideLoading 与组件连接。
 *
 * 注意：这套机制只覆盖自定义 Axios 实例。qa.ts 使用 fetch，它的按钮 loading
 * 由 DemoWorkspace 自己管理。
 */

import { AxiosRequestConfig, AxiosResponse } from 'axios'
import { IRequestPlugin } from './plugin'

function show() {
  window.$showLoading({ title: '加载中...' })
}

function hide() {
  window.$hideLoading()
}

export const loadingPlugin: IRequestPlugin = {
  // 先注册响应清理：无论请求成功还是失败，都要对称地结束 loading。
  preinstall(instance) {
    instance.interceptors.response.use(
      (response) => {
        const config = response.config as AxiosRequestConfig
        if (config.loading) hide()
        return response
      },
      (error: unknown) => {
        // 网络错误可能没有 response，所以同时尝试 error.config。
        const config = configFromError(error)
        if (config?.loading) hide()
        return Promise.reject(error)
      },
    )
  },

  // 后注册请求准备：只有显式或默认开启 loading 的请求才显示遮罩。
  postinstall(instance) {
    instance.interceptors.request.use((config) => {
      if (config.loading) show()
      return config
    })
  },
}

/** 从不同 Axios 错误形状中安全取出原请求配置。 */
function configFromError(error: unknown): AxiosRequestConfig | undefined {
  if (typeof error !== 'object' || error === null) return undefined

  if ('response' in error && error.response) {
    const response = error.response as AxiosResponse<unknown>
    if (response.config) return response.config
  }

  if ('config' in error) {
    return error.config as AxiosRequestConfig | undefined
  }

  return undefined
}
