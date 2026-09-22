/**
 * 模块：多模态手册问答单页应用
 *
 * 这是当前前端最重要的业务文件。main.tsx 只负责启动 React，真正的页面状态、
 * 按钮事件和结果展示都从这里开始。
 *
 * 本文件包含四个自定义组件：
 * - App：应用外壳，提供 Ant Design 中文语言、主题和全局上下文；
 * - DemoWorkspace：业务中心，管理上传、问答、答案、证据和错误状态；
 * - EvidencePanel：纯展示组件，把 evidence 数组转换成证据卡片；
 * - MountApi：把 Ant Design 的 message/loading 能力提供给 Axios 插件。
 *
 * 三条主要用户链路：
 * - 上传 PDF：Upload -> uploadManual -> api.manuals.upload -> POST /manuals/upload；
 * - 文本问答：TextArea -> askText -> api.qa.textQa -> POST /qa/text SSE；
 * - 图片问答：Upload -> askImage -> api.qa.imageQa -> POST /qa/image JSON。
 *
 * 设计边界：当前是单页联调 Demo，没有路由和全局业务状态库。DemoWorkspace
 * 集中了较多职责，方便追踪完整链路，但继续产品化时应按业务域拆分组件。
 */
// `@/` 由 Vite 和 TypeScript 映射到 frontend/src。使用 `* as api` 后，调用处会
// 保留业务分组，例如 api.manuals.upload 和 api.qa.textQa，比平铺函数名更直观。
import * as api from '@/api'

// 这三个名称都是 TypeScript 类型：它们描述 state 和 props 的数据形状，构建为
// JavaScript 后不会成为运行时代码。EvidenceItem 同时服务文本和视觉证据。
import { EvidenceItem, ImageQaResult, ManualUploadResult } from '@/types/manual'

// Ant Design 也导出一个名为 App 的组件。为了不与本文件自定义的 App 冲突，
// 导入时重命名为 AntdApp。ConfigProvider 提供主题，Spin 提供全屏加载动画。
import { App as AntdApp, ConfigProvider, Spin } from 'antd'

// 这些是通用 UI 组件，只负责浏览器交互和外观；PDF 解析、检索和模型调用仍在
// FastAPI 后端。比如 Upload 只选择文件，不会自动理解或解析文件内容。
import {
  Alert,
  Button,
  Image,
  Input,
  Space,
  Tag,
  Typography,
  Upload,
} from 'antd'

// 让 Upload、Image 等 AntD 组件自带的按钮和提示使用中文。
import zhCN from 'antd/es/locale/zh_CN'

// `import type` 明确表示只导入类型。UploadFile 是 AntD 对文件列表项的包装，
// 真正上传给后端的浏览器 File 位于 UploadFile.originFileObj。
import type { UploadFile } from 'antd/es/upload/interface'

// 本文件使用的 Hook：useState 保存页面数据，useEffect 在首次挂载后查状态，
// useMemo 计算选中文件，useRef 保存并发计数，useCallback稳定全局函数引用。
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

// 这是普通全局 CSS 导入。JSX 中的 className 会与 App.css 的选择器匹配。
import './App.css'

// TextArea 是 Input 的子组件。提前解构后，JSX 可以写 <TextArea>，不必每次写
// <Input.TextArea>；文本问题和图片补充问题各使用一个独立实例。
const { TextArea } = Input

/**
 * 应用外壳组件。
 *
 * main.tsx 只渲染一个 <App />。App 自己不保存业务状态，而是从外到内建立两个
 * Provider，再放入真正的工作台：
 *
 * ConfigProvider（中文和主题）
 *   -> AntdApp（message/modal/notification 上下文）
 *      -> DemoWorkspace（业务页面）+ MountApi（请求插件桥接）
 *
 * Provider 的配置会沿 React 组件树向下生效，因此 DemoWorkspace 不需要逐个给
 * Button、Upload 传主题色和语言包。
 */
function App() {
  return (
    <ConfigProvider
      // locale 只影响 AntD 内置文案；业务标题仍是 App.tsx 中直接写的中文。
      locale={zhCN}
      theme={{
        // 以 CSS 变量形式注入 token，浏览器开发工具中可以看到最终主题值。
        cssVar: true,
        token: {
          // 主要影响 type="primary" 的按钮和使用主色的 AntD 交互状态。
          colorPrimary: '#1F70FE',
        },
      }}
    >
      <AntdApp>
        {/* 两个组件是兄弟节点：工作台负责业务，MountApi 负责全局请求反馈。 */}
        <DemoWorkspace />
        <MountApi />
      </AntdApp>
    </ConfigProvider>
  )
}

/**
 * 单页工作台，也是当前业务状态的所有者。
 *
 * React 函数组件会在 state 更新后重新执行。下面的 useState 并不是普通局部变量：
 * setter 更新值后，React 会再次运行 DemoWorkspace，并用新状态计算新的 JSX。
 *
 * 状态按职责分为四组：
 * 1. 手册上下文：manual；
 * 2. 用户输入：两个 Upload 文件列表和两个问题字符串；
 * 3. 后端结果：answer、evidence、imageResult；
 * 4. 交互反馈：loading、error。
 *
 * 当前所有业务状态集中在这里，所以文本与图片问答能复用回答/证据面板；代价是
 * 组件较大，而且一个 loading 同时代表三种任务。
 */
function DemoWorkspace() {
  // ===== 1. 当前手册 =====
  // null 表示页面尚未从后端恢复到有效手册；它不一定表示数据库中绝对没有数据，
  // 也可能是状态接口还没完成或查询失败。成功查询/上传后保存完整建库统计。
  const [manual, setManual] = useState<ManualUploadResult | null>(null)

  // ===== 2. 用户选择的文件 =====
  // AntD Upload 是受控组件：fileList 由 React state 提供，选择或删除文件后再通过
  // onChange 写回。maxCount=1 限制 UI 只保留一个元素，但类型仍然是数组。
  const [manualFileList, setManualFileList] = useState<UploadFile[]>([])
  const [imageFileList, setImageFileList] = useState<UploadFile[]>([])

  // ===== 3. 两个受控输入框 =====
  // 输入框显示 value，用户输入时 onChange 调 setter；因此页面状态始终是问题文本
  // 的唯一可信来源。两类问题分开保存，切换问答方式不会互相覆盖。
  const [textQuestion, setTextQuestion] = useState('')
  const [imageQuestion, setImageQuestion] = useState('')

  // ===== 4. 两种问答共用的结果区 =====
  // 文本问答每收到一条 answer_delta 就追加；图片问答收到完整 JSON 后一次替换。
  const [answer, setAnswer] = useState('')

  // 数组交给 EvidencePanel。新问题开始前先清空，防止旧证据短暂挂在新答案下方。
  const [evidence, setEvidence] = useState<EvidenceItem[]>([])

  // 只有图片问答拥有视觉描述、候选名和视觉匹配摘要；null 时整个识别面板不渲染。
  const [imageResult, setImageResult] = useState<ImageQaResult | null>(null)

  // ===== 5. 页面反馈 =====
  // 当前三个按钮共享一个 boolean。任一任务运行时，三个按钮都会显示 loading；
  // 这是 Demo 的简化设计，不等价于独立的 upload/text/image 请求状态机。
  const [loading, setLoading] = useState(false)

  // 空字符串表示不显示顶部 Alert；有值时 Alert 展示并允许用户手动关闭。
  const [error, setError] = useState('')

  // manualId 可以完全从 manual 推导，所以不再额外 useState，避免两份手册 ID 不一致。
  // 首次加载、空库或状态查询失败时使用后端同样约定的默认 ID。
  const manualId = manual?.manual_id || 'demo_manual'

  /**
   * 页面挂载后的状态恢复。
   *
   * React state 只在当前页面内存中存在，刷新后 manual 会重新变为 null；数据库和
   * ES 中的手册并没有随浏览器刷新删除。因此首次渲染结束后，请求后端恢复统计。
   */
  useEffect(() => {
    api.manuals
      .status()
      .then((response) => {
        // Axios 返回外层 AxiosResponse，真正的后端 JSON 位于 response.data。
        const data = response.data as ManualUploadResult

        // 后端空库状态可能返回 manual_id=null。只在 ID 有效时更新，状态卡片否则
        // 继续显示“未上传”和 0。setManual 会触发组件重新渲染。
        if (data?.manual_id) {
          setManual(data)
        }
      })
      .catch(() => {
        // 当前本地 Demo 把状态恢复当成 best-effort：失败不阻断页面，用户仍可上传。
        // 代价是 UI 无法区分“数据库为空”和“状态接口故障”，生产版应单独保存状态错误。
      })

    // 空依赖数组表示只在本次组件挂载后执行；后续普通 state 更新不会重新查状态。
  }, [])

  // UploadFile 还包含显示名称、状态等 UI 信息，接口真正需要的是 originFileObj 中
  // 的浏览器 File。useMemo 明确它只依赖 PDF 列表，列表不变时复用同一结果。
  const selectedManual = useMemo(() => {
    return manualFileList[0]?.originFileObj as File | undefined
  }, [manualFileList])

  // 图片使用同样的派生过程。没有选择文件时可选链返回 undefined，按钮据此禁用，
  // 事件函数里仍会再次检查，形成 UI 与业务函数两层保护。
  const selectedImage = useMemo(() => {
    return imageFileList[0]?.originFileObj as File | undefined
  }, [imageFileList])

  /**
   * “上传并建库”按钮的点击处理函数。
   *
   * 调用前，Upload 组件只是把浏览器 File 保存到 state，并未发送网络请求。这里
   * 才调用 POST /manuals/upload。当前后端同步执行保存、PDF 解析、chunk 构建、
   * embedding 和写库，所以 await 可能持续较长时间，不只是等待文件传输。
   *
   * 函数没有 return 业务数据；它通过 setManual 等 setter 把结果写入页面状态。
   */
  async function uploadManual() {
    // 按钮已经通过 disabled 阻止空文件点击，但函数自身仍防御性检查，避免未来
    // 被键盘快捷键或其他组件直接调用时提交空 multipart 请求。
    if (!selectedManual) return

    // 新操作开始时先清除上一次错误，否则成功请求期间仍会显示旧的红色 Alert。
    setError('')

    // 触发重新渲染，Button 的 loading 属性开始显示转圈。Axios 上传还配置了
    // 全屏 loading，因此当前上传过程可能同时看到按钮和全屏两层反馈。
    setLoading(true)

    try {
      // AxiosResponse 的 data 是 ManualUploadResult，包括页数和两类 chunk 统计。
      const response = await api.manuals.upload(selectedManual)

      // 保存新手册后，manualId 派生值也会随下一次渲染切换到新 manual_id。
      setManual(response.data)

      // 旧 answer/evidence/imageResult 属于上传前的手册。必须一起清空，否则页面
      // 会让用户误以为旧证据来自刚建库的新手册。
      setAnswer('')
      setEvidence([])
      setImageResult(null)
    } catch (error: unknown) {
      // 可能失败在网络、PDF 解析、数据库、ES 或模型调用。errorMessage 先验证
      // unknown 的形状，再选择具体 message 或“手册上传失败”兜底文案。
      setError(errorMessage(error, '手册上传失败'))
    } finally {
      // finally 无论 try 成功还是 catch 失败都会执行，保证按钮不会永久转圈。
      setLoading(false)
    }
  }

  /**
   * “发送问题”按钮的文本 RAG 处理函数。
   *
   * 与图片问答不同，POST /qa/text 返回 SSE 流。api.qa.textQa 负责把网络字节
   * 解析为事件，本函数通过 onEvent 决定每种事件如何更新 React state。
   */
  async function askText() {
    // trim 去掉首尾空格。按钮虽已按同一条件禁用，函数仍保留输入边界检查。
    if (!textQuestion.trim()) return

    // 请求开始前建立干净的结果区。尤其要清空 imageResult，否则右侧可能同时
    // 显示上一次图片识别摘要和这一次纯文本答案，造成结果来源混淆。
    setError('')
    setLoading(true)
    setAnswer('')
    setEvidence([])
    setImageResult(null)

    try {
      // await 等到服务端关闭整个 SSE 流才完成，但 onEvent 会在等待期间被多次调用，
      // 所以用户无需等待完整答案就能逐段看到内容。
      await api.qa.textQa({
        // 后端按 manual_id 过滤 ES/数据库证据，防止不同手册内容混用。
        manual_id: manualId,
        question: textQuestion.trim(),

        // TextQaEvent 使用 type 作为判别字段。检查 type 后，TypeScript 才知道
        // 当前事件安全地拥有 content 或 items。
        onEvent(event) {
          if (event.type === 'answer_delta') {
            // 使用函数式 setter，而不是 setAnswer(answer + content)。SSE 事件可能
            // 快速连续到达，函数参数 value 始终是 React 提供的最新累计答案。
            setAnswer((value) => `${value}${event.content}`)
          }

          if (event.type === 'evidence') {
            // evidence 通常在回答流后部一次返回。替换整个数组后，EvidencePanel
            // 会按 chunk_id 重新计算需要新增或移除的证据卡片。
            setEvidence(event.items as EvidenceItem[])
          }

          // done 或未来新增的未知事件在当前 UI 中无需更新状态；流真正结束后
          // textQa Promise 返回，finally 再关闭 loading。
        },
      })
    } catch (error: unknown) {
      // HTTP 非 2xx、网络中断、TextDecoder/JSON.parse 异常都会进入这里。
      setError(errorMessage(error, '文本问答失败'))
    } finally {
      // SSE 正常结束或中途失败后都恢复按钮状态。
      setLoading(false)
    }
  }

  /**
   * “图片查询”按钮的多模态问答处理函数。
   *
   * api.qa.imageQa 会把 manual_id、图片 File 和可选问题组装成 FormData。后端要
   * 先完成图片理解、三路召回、门控融合、VLM 二验、正文检索和回答，最后才
   * 一次性返回 JSON，因此这里不像文本问答那样处理中间流事件。
   */
  async function askImage() {
    // 与 Upload 按钮的 disabled 形成双重保护，确保 FormData 中一定有 image。
    if (!selectedImage) return

    // 新图片开始查询时清空旧答案和证据。imageResult 暂不提前清空，因此请求
    // 期间旧识别面板仍会保留；若希望严格避免旧结果残留，可在此 setImageResult(null)。
    setError('')
    setLoading(true)
    setAnswer('')
    setEvidence([])

    try {
      const result = await api.qa.imageQa({
        // 手册 ID 决定候选和正文检索范围。
        manual_id: manualId,
        // selectedImage 是浏览器 File，不是 base64 字符串或服务器路径。
        image: selectedImage,

        // 空白输入转为 undefined。qa.ts 检查后不会 append question，FastAPI 的
        // question 参数便使用 None，由后端构造默认问法。
        question: imageQuestion.trim() || undefined,
      })

      // 同一个后端响应被拆给三个页面区域：
      // imageResult -> 图片识别摘要；answer -> 共用回答区；evidence -> 证据卡片。
      setImageResult(result)
      setAnswer(result.answer)

      // 类型声明要求 evidence 为数组，`|| []` 仍为旧后端或异常空值提供兼容兜底。
      // 业务拒答也可能是 HTTP 200 且 evidence=[]，不能仅凭请求成功判断“已匹配”。
      setEvidence(result.evidence || [])
    } catch (error: unknown) {
      // 这里只处理 HTTP/网络/解析层失败；后端有意返回的“证据不足”属于正常 JSON，
      // 会显示在 answer/suggestion，而不会进入 catch。
      setError(errorMessage(error, '图片问答失败'))
    } finally {
      // 完整 JSON 收到或请求失败后，恢复按钮。
      setLoading(false)
    }
  }

  /**
   * 下面返回的是 JSX 页面描述，不是拼接出来的 HTML 字符串。
   *
   * 每次 state 更新，React 会重新执行到这里，把新 JSX 与上一次结果比较，只把
   * 真正变化的 DOM 更新到浏览器。className 对应 App.css；花括号 `{}` 内可以
   * 写 JavaScript 表达式，把 state 转成文字、属性或条件节点。
   */
  return (
    // 最外层容器至少占满 100vh，并提供整个工作台的背景和默认文字颜色。
    <div className="demo-shell">
      {/* header 是语义化页头；它不保存状态，只展示项目名称和定位。 */}
      <header className="demo-header">
        <div className="demo-header__inner">
          <h1 className="demo-title">AutoRAG</h1>
          <p className="demo-subtitle">
            基于多模态 RAG 的汽车手册问答系统
          </p>
        </div>
      </header>

      {/* main 表示当前文档的主要内容，内部先放全局错误，再放两栏工作区。 */}
      <main className="demo-main">
        {/*
          这是条件渲染：error 是非空字符串时创建 Alert；空字符串时返回 null，
          React 不会在 DOM 中保留一个隐藏 Alert。closable 允许用户手动清空错误。
        */}
        {error ? (
          <Alert
            // 少量只作用当前实例的样式直接使用 style；通用布局仍放 App.css。
            style={{ marginBottom: 16 }}
            type="error"
            message={error}
            showIcon
            closable
            // onClose 接收函数而非 setError('') 的执行结果；点击关闭时才真正调用。
            onClose={() => setError('')}
          />
        ) : null}

        {/*
          demo-grid 在宽屏下生成“320～380px 左栏 + 剩余宽度右栏”；小于 900px
          时 App.css 媒体查询会改成单栏。React 组件树不需要为响应式布局分叉。
        */}
        <div className="demo-grid">
          {/* aside 表示辅助操作区：手册上下文、上传入口和图片查询都放在这里。 */}
          <aside>
            {/*
              手册状态直接读取 manual state。状态接口尚未完成或返回空库时，manual
              为 null，因此所有字段都必须提供默认展示，不能直接写 manual.page_count。
            */}
            <section className="demo-panel">
              <h2 className="demo-panel__title">手册状态</h2>
              <div className="demo-status">
                <div className="demo-status__row">
                  <span>manual_id</span>
                  {/* `?.` 遇到 null 就停止读取；`||` 再提供默认手册 ID。 */}
                  <strong>{manual?.manual_id || 'demo_manual'}</strong>
                </div>
                <div className="demo-status__row">
                  <span>文件</span>
                  <strong>{manual?.file_name || '未上传'}</strong>
                </div>
                <div className="demo-status__row">
                  <span>页数</span>
                  {/*
                    数值使用 `??` 而不是 `||`：0 是合法统计值，不应被误判为空；
                    只有 null/undefined 才退回到右侧的 0。
                  */}
                  <strong>{manual?.page_count ?? 0}</strong>
                </div>
                <div className="demo-status__row">
                  <span>文本 chunk</span>
                  <strong>{manual?.text_chunk_count ?? 0}</strong>
                </div>
                <div>
                  <span>文本父节点</span>
                  <strong>{manual?.text_parent_count ?? 0}</strong>
                </div>
                <div className="demo-status__row">
                  <span>视觉 chunk</span>
                  <strong>{manual?.visual_chunk_count ?? 0}</strong>
                </div>
                <div className="demo-status__row">
                  <span>图标组</span>
                  <strong>{manual?.icon_group_count ?? 0}</strong>
                </div>
              </div>
            </section>

            {/*
              PDF 上传分成两个动作：Upload 选择并保存本地 File；下面的 Button 才
              调用 uploadManual 发请求。这样用户可以先检查文件名再主动开始长建库。
            */}
            <section className="demo-panel">
              <h2 className="demo-panel__title">上传手册</h2>
              <Upload
                // 让 Upload 根节点占满面板宽度；按钮自己的 block 再占满内部宽度。
                className="demo-upload"
                // accept 改善系统文件选择器筛选体验，但不能替代后端文件校验。
                accept=".pdf,application/pdf"
                // 当前只支持一本默认手册，因此 UI 最多保留一个候选 PDF。
                maxCount={1}
                // 返回 false 阻止 AntD 自动上传；项目必须调用自己的 /manuals/upload。
                beforeUpload={() => false}
                // fileList + onChange 构成受控 Upload，真实列表保存在 React state。
                fileList={manualFileList}
                onChange={({ fileList }) => setManualFileList(fileList)}
              >
                <Button block>选择 PDF</Button>
              </Upload>
              <div className="demo-actions">
                <Button
                  type="primary"
                  block
                  // 共享 loading 为 true 时显示转圈并阻止按钮的普通点击反馈。
                  loading={loading}
                  // 没有从 originFileObj 得到真实 File 时不允许提交。
                  disabled={!selectedManual}
                  // 这里传函数引用；React 会在用户点击后执行 uploadManual。
                  onClick={uploadManual}
                >
                  上传并建库
                </Button>
              </div>
            </section>

            {/* 图片问答同样采用“先选择、再提交”，但会额外收集一个可选问题。 */}
            <section className="demo-panel">
              <h2 className="demo-panel__title">图片问答</h2>
              <Upload
                // 浏览器选择器只展示常见支持格式；后端仍会再次检查文件扩展名。
                accept="image/png,image/jpeg,image/webp"
                maxCount={1}
                beforeUpload={() => false}
                fileList={imageFileList}
                onChange={({ fileList }) => setImageFileList(fileList)}
              >
                <Button block>选择图片</Button>
              </Upload>
              <TextArea
                style={{ marginTop: 12 }}
                // value 来自 state，所以它是受控输入框。
                value={imageQuestion}
                // 每次键入都取事件目标当前值写回 state，随后 React 重新渲染 value。
                onChange={(event) => setImageQuestion(event.target.value)}
                placeholder="可选补充问题"
                // 随文字在 2～4 行间自动增高，超过 4 行后在输入框内部滚动。
                autoSize={{ minRows: 2, maxRows: 4 }}
              />
              <div className="demo-actions">
                <Button
                  type="primary"
                  block
                  loading={loading}
                  // 问题可以为空，但图片是必需输入。
                  disabled={!selectedImage}
                  onClick={askImage}
                >
                  图片查询
                </Button>
              </div>
            </section>
          </aside>

          {/*
            右栏是统一结果区：文本问答和图片问答共用 answer/evidence；只有图片
            问答额外产生 imageResult，因此图片识别面板采用条件渲染。
          */}
          <section>
            {/* 文本问题输入和发送入口。TextArea 不直接调用后端，按钮才触发 askText。 */}
            <section className="demo-panel">
              <h2 className="demo-panel__title">文本问答</h2>
              <TextArea
                // 受控输入：页面显示内容始终与 textQuestion state 一致。
                value={textQuestion}
                onChange={(event) => setTextQuestion(event.target.value)}
                placeholder="输入车主手册相关问题"
                autoSize={{ minRows: 3, maxRows: 8 }}
              />
              <div className="demo-actions">
                <Button
                  type="primary"
                  loading={loading}
                  // 只有空格也视为空问题；askText 内部会用相同条件再次校验。
                  disabled={!textQuestion.trim()}
                  onClick={askText}
                >
                  发送问题
                </Button>
              </div>
            </section>

            {/*
              imageResult 初始为 null，纯文本问答开始时也会被清空，所以该面板只在
              收到图片接口 JSON 后存在。`condition ? node : null` 是 React 条件渲染。
            */}
            {imageResult ? (
              <section className="demo-panel">
                <h2 className="demo-panel__title">图片识别</h2>
                <div className="demo-image-result">
                  {/* 第一行展示最终对象、风险级别和可用的总体置信度。 */}
                  <Space wrap>
                    <Tag color="blue">{imageResult.recognized_object}</Tag>
                    <Tag color={riskColor(imageResult.risk_level)}>
                      {imageResult.risk_level}
                    </Tag>
                    {/*
                      不能写 `imageResult.confidence ? ...`：合法的 0 会被当成 false。
                      显式检查 number 后，0 也能显示，undefined 才隐藏标签。
                    */}
                    {typeof imageResult.confidence === 'number' ? (
                      <Tag color="green">
                        confidence {imageResult.confidence.toFixed(2)}
                      </Tag>
                    ) : null}
                  </Space>

                  {/* visual_match 不存在时，说明后端没有可展示的最终候选摘要。 */}
                  {imageResult.visual_match ? (
                    <Space wrap>
                      <Tag
                        color={
                          imageResult.visual_match.matched ? 'green' : 'orange'
                        }
                      >
                        {/* matched 是后端门控结论，不应由前端根据 score 自行推断。 */}
                        {imageResult.visual_match.matched
                          ? '视觉匹配'
                          : '视觉候选'}
                      </Tag>
                      {imageResult.visual_match.object_name ? (
                        <Tag>{imageResult.visual_match.object_name}</Tag>
                      ) : null}
                      {imageResult.visual_match.section_title ? (
                        <Tag>{imageResult.visual_match.section_title}</Tag>
                      ) : null}
                      {typeof imageResult.visual_match.score === 'number' ? (
                        <Tag color="green">
                          match {imageResult.visual_match.score.toFixed(2)}
                        </Tag>
                      ) : null}
                      {imageResult.visual_match.match_method ? (
                        <Tag color="purple">
                          method {imageResult.visual_match.match_method}
                        </Tag>
                      ) : null}
                      {typeof imageResult.visual_match.vlm_verified ===
                      'boolean' ? (
                        <Tag
                          color={
                            imageResult.visual_match.vlm_verified
                              ? 'green'
                              : 'orange'
                          }
                        >
                          {/*
                            vlm_verified=false 也是有意义的结果，必须检查 boolean 类型，
                            不能只在 true 时显示。
                          */}
                          VLM{' '}
                          {imageResult.visual_match.vlm_verified
                            ? 'verified'
                            : 'unverified'}
                        </Tag>
                      ) : null}
                      {typeof imageResult.visual_match.vlm_confidence ===
                      'number' ? (
                        <Tag color="cyan">
                          rerank{' '}
                          {imageResult.visual_match.vlm_confidence.toFixed(2)}
                        </Tag>
                      ) : null}
                      {imageResult.visual_match.page ? (
                        // 页码来自手册视觉证据，便于用户回到 PDF 核验。
                        <Tag>page {imageResult.visual_match.page}</Tag>
                      ) : null}
                    </Space>
                  ) : null}

                  {/* VLM 对用户上传图的自然语言描述，用于人工理解初判依据。 */}
                  {imageResult.visual_description ? (
                    <Typography.Text>
                      {imageResult.visual_description}
                    </Typography.Text>
                  ) : null}

                  {/*
                    map 把每个候选名称变成 Tag。key 帮 React 稳定识别列表元素；
                    如果后端可能返回重复名称，应先去重，否则会出现重复 key 警告。
                  */}
                  {imageResult.candidate_names?.length ? (
                    <Space wrap>
                      {imageResult.candidate_names.map((name) => (
                        <Tag key={name}>{name}</Tag>
                      ))}
                    </Space>
                  ) : null}

                  {/* 数组 join 只负责展示，不会改变后端关键词数组。 */}
                  {imageResult.visual_keywords?.length ? (
                    <Typography.Text type="secondary">
                      keywords: {imageResult.visual_keywords.join(' / ')}
                    </Typography.Text>
                  ) : null}

                  {/* 这是后端最终用于正文检索的 query，主要用于调试和解释链路。 */}
                  {imageResult.normalized_query ? (
                    <Typography.Text type="secondary">
                      query: {imageResult.normalized_query}
                    </Typography.Text>
                  ) : null}

                  {/*
                    vision_error 是图片接口正常 JSON 内的降级字段，不等同于页面 error。
                    因此使用 warning Alert，并继续允许展示后端给出的拒答和建议。
                  */}
                  {imageResult.vision_error ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="VLM 调用失败"
                      description={imageResult.vision_error}
                    />
                  ) : null}

                  {/* suggestion 是后端返回的用户下一步操作或安全提示。 */}
                  <Typography.Text>{imageResult.suggestion}</Typography.Text>
                </div>
              </section>
            ) : null}

            {/*
              统一回答面板：文本 SSE 会连续追加 answer，图片 JSON 会一次设置 answer。
              CSS 的 white-space: pre-wrap 保留后端换行；空字符串时显示占位文案。
            */}
            <section className="demo-panel">
              <h2 className="demo-panel__title">回答</h2>
              <div className="demo-answer">{answer || '等待问题输入。'}</div>
            </section>

            {/*
              父组件通过 props 把 evidence state 传给展示组件。数据流是单向的：
              DemoWorkspace -> EvidencePanel；子组件不会请求后端或修改父状态。
            */}
            <section className="demo-panel">
              <h2 className="demo-panel__title">证据</h2>
              <EvidencePanel items={evidence} />
            </section>
          </section>
        </div>
      </main>
    </div>
  )
}

/**
 * 证据列表展示组件。
 *
 * props 是父组件传入的数据入口。这里用内联类型声明规定 items 必须是
 * EvidenceItem[]，因此调用者漏传、传错类型时，TypeScript 会在构建前提示。
 *
 * 这是“纯展示组件”：它不保存 state、不调用接口，也不修改 items，只把后端证据
 * 转换成卡片。这样检索逻辑仍集中在后端，请求状态仍由 DemoWorkspace 管理。
 */
function EvidencePanel(props: { items: EvidenceItem[] }) {
  // 提前返回能把“空列表”和“有数据”两个分支分开。拒答、尚未提问或接口返回
  // evidence=[] 时都进入这里；该组件本身不猜测究竟是哪一种业务原因。
  if (!props.items.length) {
    // secondary 只改变视觉强调程度，不会禁用文本或改变数据状态。
    return <Typography.Text type="secondary">暂无证据。</Typography.Text>
  }

  // 有证据时返回一个列表容器。map 不会修改原数组，而是为每个 item 创建一张
  // 独立卡片，最后得到 React 节点数组。
  return (
    <div className="demo-evidence">
      {props.items.map((item) => {
        // 缩略图体积较小，列表展示时优先使用；旧数据或文本证据可能没有缩略图，
        // 再回退到 image_path。两者都为空时 src 为 undefined，下面不渲染 Image。
        const src = item.thumbnail_path || item.image_path

        return (
          // article 是语义化的独立内容单元。chunk_id 是后端持久化标识，适合作为
          // key；React 用它判断新旧列表中的卡片是复用、移动、增加还是删除。
          <article className="demo-evidence__item" key={item.chunk_id}>
            {/*
              元信息来自后端，不参与前端重新打分：page 便于回查手册，modality 区分
              text/visual，score 与 match_method 用于解释该证据如何被选中。
            */}
            <div className="demo-evidence__meta">
              <Tag>page {item.page}</Tag>
              <Tag color="geekblue">{item.modality}</Tag>
              {/* 0 是合法分数，所以检查 number 类型，不能直接用 item.score 作条件。 */}
              {typeof item.score === 'number' ? (
                <Tag color="green">{item.score.toFixed(2)}</Tag>
              ) : null}
              {item.match_method ? (
                <Tag color="purple">{item.match_method}</Tag>
              ) : null}
            </div>

            {/*
              后端保存的是静态资源路径，assetUrl 负责补齐 API 域名。Image 只负责
              浏览器展示和预览，不会在这里读取本地文件或再次执行视觉检索。
            */}
            {src ? (
              <Image
                src={assetUrl(src)}
                width={120}
                style={{ marginBottom: 8, borderRadius: 6 }}
              />
            ) : null}

            {item.object_name ? (
              <Typography.Text strong>{item.object_name}</Typography.Text>
            ) : null}
            {item.section_title ? (
              <Typography.Text type="secondary">
                {item.section_title}
              </Typography.Text>
            ) : null}
            <p className="demo-evidence__text">
              {item.content_text ||
                (item.modality === 'page_snapshot' ? '整页视觉证据' : '')}
            </p>
          </article>
        )
      })}
    </div>
  )
}

/**
 * 把 catch 捕获到的未知值转换成用户可读文字。
 *
 * JavaScript 允许 throw 任意内容，例如 Error、字符串甚至普通对象，所以
 * TypeScript 会把安全的 catch 参数视为 unknown。这里先识别标准 Error，再
 * 兼容少量第三方库可能抛出的 { message } 对象，最后才使用调用方提供的兜底文案。
 */
function errorMessage(error: unknown, fallback: string) {
  // 标准 Error（包括大多数 Axios/运行时异常）可以安全读取 message。instanceof
  // 同时完成运行时检查和 TypeScript 类型收窄。
  if (error instanceof Error && error.message) return error.message

  // 某些库会 reject 一个普通对象而不是 Error。先确认它不是 null、确实含有
  // message，并且 message 是非空字符串，避免对 unknown 直接取属性导致二次异常。
  if (
    typeof error === 'object' &&
    error !== null &&
    'message' in error &&
    typeof error.message === 'string' &&
    error.message
  ) {
    return error.message
  }

  // 字符串、空对象等无法可靠提取信息时，用调用点提供的业务文案。例如上传、
  // 文本问答和图片问答分别传入不同 fallback，用户能知道失败发生在哪条链路。
  return fallback
}

/**
 * 把证据中的资源路径转换为浏览器可请求的图片 URL。
 *
 * 开发环境中，前端和后端通常运行在不同端口：页面可能是 localhost:5173，图片
 * 实际由 localhost:8000 提供。如果后端只返回 `/storage/...`，浏览器直接使用它会
 * 错误请求前端端口，所以需要补上与 API 相同的基础地址。
 */
function assetUrl(path: string) {
  // 已经是绝对 HTTP(S) 地址时原样返回，避免重复拼成
  // `http://apihttp://cdn/...`。正则中的 i 表示忽略大小写。
  if (/^https?:\/\//i.test(path)) return path

  // VITE_API_BASE 在 Vite 构建时从环境变量注入，例如 http://localhost:8000。
  // 浏览器不能读取任意服务器环境变量，只有 VITE_ 前缀会暴露给前端代码。
  const base = import.meta.env.VITE_API_BASE || ''

  // 当前约定 base 不以 `/` 结尾、path 以 `/` 开头。若接口以后返回 `storage/x`
  // 或配置带尾斜杠，应先统一斜杠，否则可能得到缺少或重复 `/` 的 URL。
  return `${base}${path}`
}

/**
 * 把后端风险枚举映射为 Ant Design Tag 的颜色名称。
 *
 * 该函数只负责视觉提示，不改变风险等级，也不参与安全判断。真正的风险判定来自
 * 后端；即使前端颜色映射缺失，也不能把未知等级自动当成 low。
 */
function riskColor(riskLevel: string) {
  // 高风险用红色强化警示。
  if (riskLevel === 'high') return 'red'
  // 中风险用橙色表示需要关注但不等同于最高等级。
  if (riskLevel === 'medium') return 'orange'
  // low 来自明确的后端值时才显示绿色。
  if (riskLevel === 'low') return 'green'
  // 新增、缺失或拼写异常的等级保持默认颜色，避免错误呈现为“安全”。
  return 'default'
}

/**
 * React UI 与普通 TypeScript 请求模块之间的桥接组件。
 *
 * loading.ts 和 error-toast.ts 不是 React 组件，不能在 Axios 拦截器里直接调用 Hook，
 * 也拿不到 AntdApp Provider 的 Context。MountApi 位于 Provider 内部，可以合法取得
 * 这些能力，再挂到浏览器 window，供请求插件调用：
 *
 * 请求开始 -> window.$showLoading -> 打开全屏 Spin
 * 请求结束 -> window.$hideLoading -> 所有并发请求结束后关闭 Spin
 * 请求失败 -> window.$app.message.error -> 显示 AntD 错误提示
 *
 * 这种全局桥接便于小型 Demo 联调，但属于隐式依赖；大型项目通常会使用 React
 * Context、请求状态库或显式服务封装，减少“必须先挂载 MountApi”的时序约束。
 */
function MountApi() {
  // useApp 必须在 <AntdApp> 的子树中调用。它返回 message、modal、notification
  // 实例；直接从 antd 静态调用可能无法继承当前主题/语言 Context。
  window.$app = AntdApp.useApp()

  // loading 决定 Spin 是否可见；setter 会触发 MountApi 重新渲染。
  const [loading, setLoading] = useState(false)

  // loadingText 是全屏层提示文字。多个并发请求先后显示时，当前实现展示最后一次
  // showLoading 传入的标题，而不是为每个请求分别保留文案。
  const [loadingText, setLoadingText] = useState('')

  // useRef 保存一个可变的并发计数。修改 current 不会触发渲染，正适合内部记账；
  // 如果用普通局部变量，每次组件重渲染都会重新变成 0。
  const loadingCount = useRef(0)

  // useCallback 的空依赖数组让函数在后续渲染中保持同一引用，普通请求模块可以
  // 持续调用它。每次组件渲染仍会把该稳定引用重新赋给 window 对应属性。
  window.$showLoading = useCallback(({ title }: { title?: string } = {}) => {
    // 例如 A、B 两个请求依次开始，计数会从 0 -> 1 -> 2。
    loadingCount.current++

    // React 会安排重渲染，使下面 Spin 收到 spinning=true。
    setLoading(true)

    // `??` 只在 title 为 null/undefined 时回退，显式空字符串仍保持为空。
    setLoadingText(title ?? '')
  }, [])

  // 每个已开启全局 loading 的请求在结束时都应恰好调用一次 hide；请求插件通过
  // Axios 的响应/异常钩子保证成功和失败两种路径都会进入这里。
  window.$hideLoading = useCallback(() => {
    // 若 A 先结束，2 -> 1，加载层仍保留；B 结束后 1 -> 0，才允许关闭。
    loadingCount.current--

    // 延迟 100ms 可以减少极短请求造成的闪烁。定时器回调读取的是 ref 的最新值，
    // 因此等待期间若又开始了请求，计数大于 0 就不会关闭加载层。
    setTimeout(() => {
      // `<= 0` 兼容计数意外低于 0，但也暴露当前实现没有严格配对校验；如果某处
      // 多调用 hide，计数会失真。更完善的实现可把计数限制为 Math.max(0, n - 1)。
      if (loadingCount.current <= 0) {
        // 更新两个 state 后，React 通常会批量执行一次重渲染。
        setLoading(false)
        setLoadingText('')
      }
    }, 100)
  }, [])

  // MountApi 不展示普通页面内容，只渲染一个全屏 Spin。fullscreen 让 AntD 创建
  // 覆盖视口的遮罩，较高 zIndex 确保它位于当前页面面板之上。
  return (
    // Fragment 不产生额外 DOM；这里虽然只有一个子节点，保留它也不会改变布局。
    <>
      <Spin
        spinning={loading}
        tip={loadingText}
        fullscreen
        style={{
          zIndex: 9999999,
        }}
      ></Spin>
    </>
  )
}

// 默认导出与 main.tsx 中的 `import App from './App'` 对应。模块内部的
// DemoWorkspace、EvidencePanel 和 MountApi 不导出，只由本文件组合使用。
export default App
