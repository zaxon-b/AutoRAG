/**
 * 模块：项目默认 Axios 实例
 *
 * 业务 API 不直接使用全局 axios，而是统一导入这里的 request。这样后端地址、
 * loading、错误提示和重复请求策略只配置一次，manuals.ts 的每个函数无需重复。
 *
 * 当前只有手册状态和 PDF 上传使用这个实例；qa.ts 为读取 SSE 使用原生 fetch，
 * 因此下面的插件不会自动作用到文本问答和图片问答。
 */

import { createRequest } from './request'

export const request = createRequest({
  // 所有相对接口路径都会拼到这个后端地址后面。
  baseURL: import.meta.env.VITE_API_BASE,

  // 以下四项是本项目扩展的 AxiosRequestConfig 字段。
  loading: true,
  errorToast: true,
  cancelRepeat: true,

  // 当前插件尚未实现响应展开；保留该值只反映原配置意图。
  unwrap: true,
})
