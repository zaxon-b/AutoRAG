/**
 * 模块：Axios 实例工厂
 *
 * `axios.create` 根据传入配置创建一个相互隔离的客户端实例；随后本模块按固定
 * 顺序安装四类插件。业务模块最终拿到的是已经具备统一行为的 AxiosInstance。
 *
 * 为什么使用工厂函数：未来若有另一个域名或不需要全局 toast 的服务，可以再次
 * 调用 createRequest 创建独立实例，而不必修改全局 axios 默认值。
 */

import axios, { AxiosRequestConfig } from 'axios'
import { errorToastPlugin } from './plugins/error-toast'
import { loadingPlugin } from './plugins/loading'
import { installPlugins } from './plugins/plugin'
import { repeatPlugin } from './plugins/repeat'
import { servicePlugin } from './plugins/service'

/** 创建并返回一个安装好项目拦截器的 Axios 实例。 */
export function createRequest(configs: AxiosRequestConfig = {}) {
  // configs 包含 baseURL 和项目自定义开关，之后每个请求还可以覆盖这些默认值。
  const instance = axios.create(configs)

  // 插件顺序会影响拦截器的注册和执行顺序，不能把它当成无序依赖列表。
  installPlugins(instance, [
    // 把后端业务错误转换成统一的 ResponseError。
    servicePlugin,
    // 在配置要求时显示和关闭全屏加载层。
    loadingPlugin,
    // 为重复请求创建 AbortController，并取消前一个同 key 请求。
    repeatPlugin,
    // 把未被页面关闭的错误显示为 Ant Design message。
    errorToastPlugin,
  ])

  return instance
}
