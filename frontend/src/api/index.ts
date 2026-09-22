/**
 * 模块：业务 API 统一出口
 *
 * 这个文件不发送网络请求，只负责整理导入路径。App.tsx 只需导入一次
 * `@/api`，再通过 `api.manuals.status()` 或 `api.qa.textQa()` 调用对应模块。
 *
 * 使用命名空间而不是把所有函数平铺导出，可以让调用位置直接看出接口属于
 * “手册生命周期”还是“问答”业务，也避免未来出现同名函数。
 */

// 手册状态与 PDF 上传接口，调用时写作 api.manuals.status / upload。
export * as manuals from './manuals'

// 文本问答与图片问答接口，调用时写作 api.qa.textQa / imageQa。
export * as qa from './qa'
