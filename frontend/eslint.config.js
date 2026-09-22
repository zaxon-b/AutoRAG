/**
 * 模块：ESLint 静态代码检查配置
 *
 * 谁读取它：执行 `npm run lint` 或编辑器启用 ESLint 时自动加载。
 * 它和 TypeScript 的区别：TypeScript 重点检查类型，ESLint 还检查未使用变量、
 * React Hook 使用规则和可能破坏热更新的导出方式。它不运行浏览器或接口测试。
 */

import js from '@eslint/js'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  // dist 是 Vite 生成的压缩产物，不应按手写源码重复检查。
  { ignores: ['dist'] },
  {
    // 同时启用 JavaScript 和 TypeScript 官方推荐规则。
    extends: [js.configs.recommended, ...tseslint.configs.recommended],

    // 下面的语言、插件和规则只作用于 TypeScript/TSX 源码。
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      // 声明 window、document、File 等浏览器变量是合法全局对象。
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      // 检查 Hook 是否只在组件顶层调用，以及依赖数组是否合理。
      ...reactHooks.configs.recommended.rules,

      // 组件文件导出非组件值时给出警告，避免开发热更新状态异常。
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
)
