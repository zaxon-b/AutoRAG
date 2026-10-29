/**
 * 模块：Vite 开发服务器与构建配置
 *
 * 谁读取它：执行 `npm run dev`、`npm run build` 或 `npm run preview` 时由 Vite
 * 自动加载，不需要在 React 代码中 import。
 *
 * 它负责开发端口、源码路径解析和 React TSX 转换；它不是 FastAPI 后端，也不
 * 配置 PostgreSQL、Elasticsearch 或模型服务。
 */

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// defineConfig 主要提供类型提示，让配置字段拼错时能在构建前被发现。
export default defineConfig(() => {
  return {
    server: {
      // 本地前端地址是 http://localhost:5181；FastAPI 仍运行在 8000 端口。
      port: 5181,

      // 监听所有网络接口，方便 Docker、局域网设备或其他容器访问开发服务器。
      // 这不等于生产部署已经安全开放，仍需防火墙和反向代理配置。
      host: '0.0.0.0',
    },

    resolve: {
      alias: [
        {
          // 把 import '@/api' 中的 @/ 映射到 frontend/src/。
          find: /^@\//,
          replacement: '/src/',
        },
      ],
    },

    // React 插件负责转换 JSX/TSX，并在开发环境提供组件热更新。
    plugins: [react()],
  }
})
