/**
 * 模块：Vite 与浏览器全局变量类型
 *
 * 这是类型声明文件，不会生成运行时代码。第一行让 TypeScript 认识
 * `import.meta.env` 等 Vite API；Window 扩展则描述 MountApi 在浏览器全局对象
 * 上实际挂载的三个能力，避免其他模块访问 window.$app 时被判定为未知属性。
 */

/// <reference types="vite/client" />

interface Window {
  /** Ant Design App 上下文，错误插件用它调用 message.error。 */
  $app: import('antd/es/app/context').useAppProps

  /** 打开全屏 Spin；由 Axios loadingPlugin 在请求开始时调用。 */
  $showLoading: (options?: { title?: string }) => void

  /** 结束一个全屏 loading 计数；所有并发请求结束后才真正关闭 Spin。 */
  $hideLoading: () => void
}
