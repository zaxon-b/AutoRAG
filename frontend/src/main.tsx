/**
 * 模块：React 浏览器入口
 *
 * 谁调用它：index.html 通过 <script type="module" src="/src/main.tsx"> 加载。
 * 它做什么：准备全局样式，找到 HTML 中的 #root，然后把最外层 App 组件
 * 挂载到浏览器页面。
 * 它不做什么：这里不保存问题、答案或文件，也不直接调用后端接口；这些业务
 * 逻辑从 App.tsx 的 DemoWorkspace 开始。
 *
 * 完整调用链：index.html -> main.tsx -> App -> DemoWorkspace。
 */

// 统一不同浏览器的部分默认样式，减少按钮、标题和 body 边距的显示差异。
import 'normalize.css'

// createRoot 是 React 18 的浏览器挂载 API，负责把 React 组件树接入真实 DOM。
import { createRoot } from 'react-dom/client'

// 覆盖少量 Ant Design 默认样式。这是全局样式，所以在入口处只导入一次。
import './antd.scss'

// App 是应用最外层组件，内部再挂载主题、业务工作台和全屏 loading。
import App from './App.tsx'

// 设置整个页面的字体、字号、背景等基础样式，同样只需要在入口处导入一次。
import './index.css'

// 从 index.html 中找到 <div id="root"></div>。
const rootElement = document.getElementById('root')

// 如果 root 被误删，与其在后面产生难懂的空指针错误，不如直接说明启动失败原因。
if (!rootElement) {
  throw new Error('前端启动失败：index.html 中缺少 id="root" 的挂载节点。')
}

// 创建 React 根节点并首次渲染 App。之后的页面更新由 React state 驱动，
// 不需要再次调用 createRoot，也不需要手动修改 root.innerHTML。
createRoot(rootElement).render(<App />)
