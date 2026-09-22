/**
 * 模块：手册生命周期 API
 *
 * 谁调用它：DemoWorkspace 首次加载时调用 status；用户点击“上传并建库”时
 * 调用 upload。
 * 请求路线：本模块 -> 自定义 Axios 实例 request -> FastAPI /manuals 路由。
 * 返回特点：Axios 返回的是 AxiosResponse，业务 JSON 位于 response.data。
 */

import { ManualUploadResult } from '@/types/manual'
import { request } from './request'

/**
 * 查询后端当前默认手册的建库状态。
 *
 * GET 请求没有请求体。页面刷新后 React state 会丢失，因此 DemoWorkspace
 * 通过这个接口重新读取服务端保存的文件名、页数和 chunk 统计。
 */
export function status() {
  // 空库时后端仍返回状态对象；联合类型保留了早期只返回 { status } 的兼容性。
  return request.get<ManualUploadResult | { status: string }>('/manuals/status')
}

/**
 * 上传一个 PDF，并等待后端完成整本手册建库。
 *
 * `file` 是浏览器原生 File，来自 Ant Design UploadFile.originFileObj。这里把
 * 二进制文件放入 FormData，因为 JSON 不能直接承载浏览器文件。
 *
 * 当前后端是同步长任务：Promise 只有在保存 PDF、解析、切块、向量化和写库
 * 全部结束后才完成。`loading: true` 还会通知 Axios 插件显示全屏遮罩。
 */
export function upload(file: File) {
  const form = new FormData()

  // 字段名必须是 file，才能匹配 FastAPI 的 file: UploadFile = File(...) 参数。
  form.append('file', file)

  return request.post<ManualUploadResult>('/manuals/upload', form, {
    // 告诉后端这是包含文件的 multipart 请求；boundary 由 Axios/浏览器补齐。
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    // 这是项目扩展的 Axios 配置，不是标准 HTTP 字段。
    loading: true,
  })
}
