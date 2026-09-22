/**
 * 模块：手册问答接口的前端类型契约
 *
 * 谁使用它：App.tsx、api/manuals.ts 和 api/qa.ts。
 * 它解决什么问题：明确后端响应应该包含哪些字段，让编辑器和 TypeScript 在
 * 开发、构建阶段发现拼写错误和明显的类型误用。
 *
 * 重要边界：这些 type 在浏览器运行时会被删除。`as ImageQaResult` 只是告诉
 * 编译器“按这个形状理解数据”，不会真的验证后端 JSON。生产系统若要防止
 * 接口漂移，还应使用运行时 Schema 校验或从 FastAPI OpenAPI 生成客户端。
 */

/** 手册状态查询和 PDF 上传成功后共用的统计结果。 */
export type ManualUploadResult = {
  /** 手册的数据隔离标识；后续问答会把它原样传回后端。 */
  manual_id: string
  /** 用户上传 PDF 时的原始文件名，用于状态卡片展示。 */
  file_name: string
  /** PDF 总页数。 */
  page_count: number
  /** 已建立、可直接参与 BM25/Dense 检索的 text 叶子数量。 */
  text_chunk_count: number
  /** 仅存于 PostgreSQL、需要由 Auto-Merging 取得的正文父节点数量。 */
  text_parent_count: number
  /** 已建立的图标、页面截图、操作图等视觉证据数量。 */
  visual_chunk_count: number
  /** 去重归并后的标准图标组数量；兼容旧响应，所以前端允许它缺省。 */
  icon_group_count?: number
  /** 后端建库状态，例如 ready；它不是 HTTP 状态码。 */
  status: string
}

/**
 * 回答所引用的一条手册证据。
 *
 * 同一个结构同时承载文本证据和视觉证据，因此很多字段是可选的：文本 chunk
 * 通常没有图片路径，视觉 chunk 则可能没有 content_text。页面渲染前应先判断
 * 字段是否存在，不能用非空断言强行读取。
 */
export type ChunkModality = 'text' | 'icon' | 'operation_figure' | 'page_snapshot'

export type EvidenceItem = {
  /** 每条 chunk 的稳定唯一标识，也用作 React 列表 key。 */
  chunk_id: string
  /** 证据所在的手册页码，方便用户回到原文核对。 */
  page: number
  /** 系统只允许存在的四种 chunk 类型。 */
  modality: ChunkModality

  // 下列字段描述证据的文字信息。
  section_title?: string
  content_text?: string
  object_name?: string
  parent_chunk_id?: string
  is_leaf?: boolean
  token_count?: number
  risk_level?: string

  // 下列路径由后端 /storage 静态目录提供，不是用户电脑上的本地文件路径。
  image_path?: string
  thumbnail_path?: string

  // icon 直接保存持久化分组 ID；operation 和 page snapshot 不参与分组。
  icon_group_id?: string

  /** 候选的检索或融合分数；不同算法产生的原始分数不一定能直接横向比较。 */
  score?: number
  /** 证据经历的匹配路线，例如 image_embedding、opencv 或 vlm_visual_anchor。 */
  match_method?: string
}

/**
 * 图片问答最终视觉锚点的摘要。
 *
 * 它回答“用户图片是否被可靠地对应到手册中的某个视觉对象”，不是最终的
 * 自然语言回答。只有视觉锚点通过门控和 VLM 二验后，后端才继续检索正文。
 */
export type VisualMatch = {
  /** true 表示当前候选满足后端的可靠匹配条件。 */
  matched: boolean
  /** 命中的视觉对象名称；page snapshot 可以为空。 */
  object_name?: string
  /** 视觉对象在 PDF 中所属的完整书签路径。 */
  section_title?: string
  modality?: Exclude<ChunkModality, 'text'>
  /** 视觉候选的融合分数。 */
  score?: number
  /** 命中的手册页码。 */
  page?: number
  risk_level?: string
  image_path?: string
  thumbnail_path?: string

  // icon 命中对象所属的持久化图标组。
  icon_group_id?: string

  /** 参与最终匹配的召回、融合和验证方法。 */
  match_method?: string
  /** VLM 是否确认用户图与候选图指向同一视觉对象。 */
  vlm_verified?: boolean
  /** VLM 二验置信度，通常是 0～1；缺省表示没有可用二验结果。 */
  vlm_confidence?: number
  /** VLM 给出的匹配或不匹配理由，当前页面暂未直接展示。 */
  vlm_reason?: string
}

/** `POST /qa/image` 一次性返回的图片问答结果。 */
export type ImageQaResult = {
  /** 最终识别对象；验证成功时优先使用手册视觉证据的标准名称。 */
  recognized_object: string
  /** VLM 对用户上传图片的外观描述。 */
  visual_description?: string
  /** 颜色、形状、符号等视觉关键词。 */
  visual_keywords?: string[]
  /** VLM 初步给出的可能名称，并不等于最终匹配已通过。 */
  candidate_names?: string[]
  /** 后端结合问题、视觉描述和锚点构造出的正文检索 query。 */
  normalized_query?: string
  /** 风险等级使用开放 string 兼容后端新增取值。 */
  risk_level: 'low' | 'medium' | 'high' | 'unknown' | string
  /** 识别或二验置信度；具体来源取决于后端是否获得可靠视觉匹配。 */
  confidence?: number
  /** VLM 首次理解失败时的错误摘要；有该字段不一定代表 HTTP 请求失败。 */
  vision_error?: string
  /** 最终视觉匹配摘要；拒答或无候选时可能不存在。 */
  visual_match?: VisualMatch
  /** 后端基于证据生成的答案，或者证据不足时的拒答说明。 */
  answer: string
  /** 面向用户的安全提醒或下一步建议。 */
  suggestion: string
  /** 最终证据列表：生产链路通常是视觉证据在前，正文证据随后。 */
  evidence: EvidenceItem[]
}

/**
 * 文本问答 SSE 中的单个事件。
 *
 * `type` 是判别字段：页面先检查 type，再安全读取该事件专属的 content 或
 * items。最后一个开放分支用于兼容后端未来新增事件，但未知事件不会自动显示。
 */
export type TextQaEvent =
  | {
      /** 一小段新生成的回答文字，需要追加到旧 answer 后面。 */
      type: 'answer_delta'
      content: string
    }
  | {
      /** 最终用于生成回答的手册证据。 */
      type: 'evidence'
      items: EvidenceItem[]
    }
  | {
      /** 服务端明确通知事件发送完毕。 */
      type: 'done'
    }
  | {
      /** 允许接收尚未加入前端联合类型的新事件。 */
      type: string
      [key: string]: unknown
    }
