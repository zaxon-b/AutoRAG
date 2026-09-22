/**
 * 模块：Axios 类型扩展
 *
 * 这是 `.d.ts` 类型声明文件，不会生成浏览器 JavaScript，也不会自己显示 loading
 * 或取消请求。它只告诉 TypeScript：“本项目允许在 Axios 配置中使用下面这些
 * 自定义字段”。真正的运行行为分别由 plugins/ 下的拦截器完成。
 *
 * `import 'axios'` 先加载原模块类型；`declare module 'axios'` 再通过 TypeScript
 * 的模块合并机制，把项目字段追加到 Axios 官方接口，而不是复制整份接口定义。
 */

import 'axios'

declare module 'axios' {
  export interface AxiosRequestConfig {
    /**
     * 是否显示全屏 loading。
     * 请求前由 loadingPlugin 调用 window.$showLoading，响应后再隐藏。
     */
    loading?: boolean

    /**
     * 请求失败时是否弹出 Ant Design message。
     * 设为 false 可让页面自行处理错误，避免同时出现 toast 和 Alert。
     */
    errorToast?: boolean

    /** 是否让 repeatPlugin 取消相同 key 的上一个未完成请求。 */
    cancelRepeat?: boolean

    /**
     * 重复请求 key 的补充部分。默认 key 只含 method 和 url；当同一 URL 的不同
     * 参数不应互相取消时，调用方需要提供能区分业务请求的 repeatKey。
     */
    repeatKey?: string

    /**
     * 计划用于展开统一响应包裹的数据开关。
     * 注意：当前 servicePlugin 并没有实现 unwrap，这只是保留的类型配置，不能
     * 因为这里有字段就认为 response.data.data 已经自动提升。
     */
    unwrap?: boolean
  }

  // AxiosResponse 的第二个泛型 D 是 Axios 原接口用于描述请求体的类型。模块合并
  // 必须保留相同的泛型列表，即使当前新增的 _data 字段暂时只使用 T。
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  export interface AxiosResponse<T, D> {
    /**
     * 预留的“展开前原始响应”字段。
     * 当前 servicePlugin 没有给它赋值，因此业务代码不应依赖它存在。
     */
    _data?: {
      code: number
      msg: string
      data: T
    }
  }
}
