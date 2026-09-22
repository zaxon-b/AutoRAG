/**
 * 模块：统一业务响应错误
 *
 * 普通 Error 只有 message；接口调试时还经常需要 HTTP 状态码和后端响应体。
 * ResponseError 在保留标准 Error 行为的同时，额外保存 AxiosResponse，供错误
 * toast、页面或日志按需读取。
 */

import { AxiosResponse } from 'axios'

export class ResponseError extends Error {
  /** 产生业务错误的原始 Axios 响应；纯网络错误时可能没有响应。 */
  response: AxiosResponse<unknown> | undefined

  constructor(message: string, response?: AxiosResponse<unknown>) {
    // 调用父类构造函数后，本对象才拥有标准的 message 和 stack。
    super(message)

    // 让 instanceof ResponseError 判断稳定，也方便浏览器调试时识别错误类型。
    this.name = 'ResponseError'
    this.response = response
  }
}
