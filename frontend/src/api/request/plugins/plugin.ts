/**
 * 模块：Axios 插件安装器
 *
 * Axios 官方提供 interceptor，但没有这里的 preinstall/install/postinstall 概念。
 * 这三个阶段是本项目为了控制拦截器注册顺序设计的轻量约定。
 *
 * 每个插件可以选择实现一个或多个阶段；安装器只负责依次调用，不负责具体的
 * loading、错误或取消逻辑。理解时可以把它看成“组装请求流水线”的地方。
 */

import { AxiosInstance } from 'axios'

/** 一个请求插件可以参与的三个安装阶段。 */
export type IRequestPlugin = {
  /** 最早注册，主要用于响应结束后的清理逻辑。 */
  preinstall?: (instance: AxiosInstance) => void
  /** 常规业务响应处理阶段。 */
  install?: (instance: AxiosInstance) => void
  /** 最后注册，主要用于请求发出前的准备逻辑和最终错误提示。 */
  postinstall?: (instance: AxiosInstance) => void
}

/** 把一组插件安装到同一个 Axios 实例。 */
export function installPlugins(
  instance: AxiosInstance,
  plugins: IRequestPlugin[],
) {
  // preinstall 使用反向插件顺序注册。Axios 的请求和响应拦截器执行规则不同，
  // 显式分阶段能让“结束 loading / 清理重复请求”等逻辑处在预期位置。
  plugins
    .slice() // 先复制数组，避免 reverse 修改调用方传入的原列表。
    .reverse()
    .forEach(({ preinstall }) => preinstall?.(instance))

  // 常规处理按业务列表顺序安装，例如先识别后端业务错误。
  plugins.forEach(({ install }) => install?.(instance))

  // 请求发出前和最终错误提示也按列表顺序安装。
  plugins.forEach(({ postinstall }) => postinstall?.(instance))
}
