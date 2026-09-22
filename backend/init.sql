-- 模块说明：init.sql 是 PostgreSQL 容器第一次初始化数据库时自动执行的 SQL 脚本。
-- 调用关系：docker-compose.yml 把它挂载到 /docker-entrypoint-initdb.d/init.sql，由 postgres 镜像启动时执行。
-- 返回/产出：创建项目需要的数据库扩展、历史遗留表和当前多模态 RAG Demo 表。
-- 逐行注释：启用 pgcrypto 扩展，让数据库可以使用 gen_random_uuid() 等加密/UUID 函数。
CREATE EXTENSION IF NOT EXISTS pgcrypto;
-- 创建 users 表
-- 逐行注释：创建历史遗留 users 表；当前精简 Demo 不做登录注册，通常不会使用它。
CREATE TABLE IF NOT EXISTS users (
-- 逐行注释：id 是 users 表的自增主键。
    id SERIAL PRIMARY KEY,
-- 逐行注释：username 保存用户名，并要求唯一且不能为空。
    username VARCHAR(50) UNIQUE NOT NULL,
-- 逐行注释：password_hash 保存密码哈希，不保存明文密码。
    password_hash VARCHAR(100) NOT NULL,
-- 逐行注释：created_at 保存记录创建时间，默认当前时间。
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
-- 逐行注释：updated_at 保存记录更新时间，默认当前时间。
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP  -- 更新时间
);

-- 创建会话表
-- 逐行注释：创建历史遗留 sessions 表；当前精简 Demo 不保留历史会话，通常不会使用它。
CREATE TABLE IF NOT EXISTS sessions (
-- 逐行注释：session_id 是会话主键。
    session_id VARCHAR(16) PRIMARY KEY,
-- 逐行注释：session_name 保存会话名称。
    session_name VARCHAR(255) NOT NULL,  
-- 逐行注释：user_id 保存所属用户 ID。
    user_id VARCHAR(255) NOT NULL,
-- 逐行注释：created_at 保存会话创建时间。
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
-- 逐行注释：updated_at 保存会话更新时间。
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP  -- 更新时间
);

-- 创建索引
-- 逐行注释：给 sessions.user_id 建索引，加快按用户查询会话。
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
-- 逐行注释：给 sessions.created_at 建索引，加快按创建时间排序或筛选。
CREATE INDEX IF NOT EXISTS idx_sessions_created_at  ON sessions(created_at);

-- 创建 messages 表
-- 逐行注释：创建历史遗留 messages 表；当前精简 Demo 不保存聊天历史，通常不会使用它。
CREATE TABLE IF NOT EXISTS messages (
-- 逐行注释：message_id 是消息主键，默认自动生成 UUID。
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
-- 逐行注释：session_id 表示消息属于哪个会话。
    session_id VARCHAR(16) NOT NULL,
-- 逐行注释：user_question 保存用户问题。
    user_question TEXT NOT NULL,
-- 逐行注释：model_answer 保存模型回答。
    model_answer TEXT NOT NULL,
-- 逐行注释：documents 保存相关文档信息；这里仍是历史字段。
    documents  TEXT,  -- 修改为 jsonb 类型
-- 逐行注释：recommended_questions 保存推荐问题；当前 Demo 不使用。
    recommended_questions TEXT,  
-- 逐行注释：think 保存模型思考或调试信息；当前 Demo 不使用。
    think TEXT,
-- 逐行注释：created_at 保存消息创建时间。
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
-- 逐行注释：updated_at 保存消息更新时间。
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP  -- 更新时间
);

-- 创建索引
-- 逐行注释：给 messages.session_id 建索引，加快按会话查消息。
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
-- 逐行注释：给 messages.created_at 建索引，加快按时间排序或筛选。
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- 创建知识库表
-- 逐行注释：创建历史遗留 knowledgebases 表；当前精简 Demo 只维护单手册，不做复杂知识库管理。
CREATE TABLE IF NOT EXISTS knowledgebases (
-- 逐行注释：id 是知识库记录自增主键。
    id SERIAL PRIMARY KEY,  -- 主键，自增
-- 逐行注释：user_id 保存所属用户 ID。
    user_id VARCHAR(255) NOT NULL,       -- 用户 ID
-- 逐行注释：file_name 保存上传文件名。
    file_name VARCHAR(255) NOT NULL,     -- 文件名称
-- 逐行注释：created_at 保存记录创建时间。
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
-- 逐行注释：updated_at 保存记录更新时间。
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP  -- 更新时间
);

-- 创建索引
-- 逐行注释：给 knowledgebases.user_id 建索引，加快按用户查知识库。
CREATE INDEX IF NOT EXISTS idx_knowledgebases_user_id ON knowledgebases(user_id);
-- 逐行注释：给 knowledgebases.created_at 建索引，加快按时间排序或筛选。
CREATE INDEX IF NOT EXISTS idx_knowledgebases_created_at ON knowledgebases(created_at);

-- 精简版车主手册多模态 RAG 数据表
-- 逐行注释：创建 manuals 表，这是当前 Demo 真正使用的手册元数据表。
CREATE TABLE IF NOT EXISTS manuals (
-- 逐行注释：id 是数据库内部自增主键。
    id SERIAL PRIMARY KEY,
-- 逐行注释：manual_id 是业务手册 ID，后端和前端用它指定查询哪本手册。
    manual_id VARCHAR(64) UNIQUE NOT NULL,
-- 逐行注释：vehicle_model 保存车型名称，可为空。
    vehicle_model VARCHAR(128),
-- 逐行注释：manual_version 保存手册版本，可为空。
    manual_version VARCHAR(128),
-- 逐行注释：file_name 保存原始 PDF 文件名。
    file_name VARCHAR(255) NOT NULL,
-- 逐行注释：file_path 保存 PDF 在本地存储中的路径。
    file_path VARCHAR(512) NOT NULL,
-- 逐行注释：page_count 保存 PDF 页数。
    page_count INTEGER NOT NULL DEFAULT 0,
-- 逐行注释：status 保存手册处理状态，例如 ready。
    status VARCHAR(32) NOT NULL DEFAULT 'ready',
-- 逐行注释：created_at 保存手册记录创建时间。
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
-- 逐行注释：updated_at 保存手册记录更新时间。
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 逐行注释：给 manuals.manual_id 建索引，加快按手册 ID 查询。
CREATE INDEX IF NOT EXISTS idx_manuals_manual_id ON manuals(manual_id);
-- 逐行注释：给 manuals.status 建索引，加快按处理状态筛选。
CREATE INDEX IF NOT EXISTS idx_manuals_status ON manuals(status);

-- 逐行注释：创建 manual_pages 表，保存每页 PDF 渲染截图和页面级文本。
CREATE TABLE IF NOT EXISTS manual_pages (
-- 逐行注释：id 是页面记录自增主键。
    id SERIAL PRIMARY KEY,
-- 逐行注释：manual_id 关联 manuals.manual_id，手册删除时页面记录也级联删除。
    manual_id VARCHAR(64) NOT NULL REFERENCES manuals(manual_id) ON DELETE CASCADE,
-- 逐行注释：page_number 保存页码。
    page_number INTEGER NOT NULL,
-- 逐行注释：image_path 保存页面截图路径。
    image_path VARCHAR(512),
-- 逐行注释：width 保存页面图宽度。
    width INTEGER,
-- 逐行注释：height 保存页面图高度。
    height INTEGER,
-- 逐行注释：ocr_text 保存页面 OCR 或文本抽取结果。
    ocr_text TEXT,
-- 逐行注释：created_at 保存页面记录创建时间。
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 逐行注释：给 manual_pages.manual_id 建索引，加快按手册查页面。
CREATE INDEX IF NOT EXISTS idx_manual_pages_manual_id ON manual_pages(manual_id);
-- 逐行注释：给 manual_pages.page_number 建索引，加快按页码查询。
CREATE INDEX IF NOT EXISTS idx_manual_pages_page_number ON manual_pages(page_number);

-- 图标组只保存组级信息；成员关系直接写在 manual_chunks.icon_group_id。
CREATE TABLE IF NOT EXISTS manual_icon_groups (
    id SERIAL PRIMARY KEY,
    icon_group_id VARCHAR(128) UNIQUE NOT NULL,
    manual_id VARCHAR(64) NOT NULL REFERENCES manuals(manual_id) ON DELETE CASCADE,
    canonical_icon_name VARCHAR(255),
    representative_chunk_id VARCHAR(128),
    member_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_manual_icon_groups_manual_id
    ON manual_icon_groups(manual_id);

-- 只允许 text、icon、operation_figure、page_snapshot 四种 chunk。
CREATE TABLE IF NOT EXISTS manual_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(128) UNIQUE NOT NULL,
    manual_id VARCHAR(64) NOT NULL REFERENCES manuals(manual_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    modality VARCHAR(64) NOT NULL DEFAULT 'text',
    section_title TEXT,
    content_text TEXT,
    object_name VARCHAR(255),
    icon_group_id VARCHAR(128) REFERENCES manual_icon_groups(icon_group_id) ON DELETE SET NULL,
    parent_chunk_id VARCHAR(128),
    is_leaf BOOLEAN,
    token_count INTEGER,
    risk_level VARCHAR(32),
    bbox JSONB,
    image_path VARCHAR(512),
    thumbnail_path VARCHAR(512),
    es_doc_id VARCHAR(128),
    score DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_manual_chunks_parent
        FOREIGN KEY (parent_chunk_id) REFERENCES manual_chunks(chunk_id)
        ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT ck_manual_chunks_modality
        CHECK (modality IN ('text', 'icon', 'operation_figure', 'page_snapshot')),
    CONSTRAINT ck_manual_chunks_parent_not_self
        CHECK (parent_chunk_id IS NULL OR parent_chunk_id <> chunk_id),
    CONSTRAINT ck_manual_chunks_fields_by_modality CHECK (
        (modality = 'text' AND is_leaf IS NOT NULL AND token_count IS NOT NULL
            AND object_name IS NULL AND icon_group_id IS NULL)
        OR (modality = 'icon' AND is_leaf IS NULL AND token_count IS NULL
            AND parent_chunk_id IS NULL)
        OR (modality IN ('operation_figure', 'page_snapshot')
            AND is_leaf IS NULL AND token_count IS NULL
            AND parent_chunk_id IS NULL AND icon_group_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_manual_chunks_chunk_id ON manual_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_manual_id ON manual_chunks(manual_id);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_page_number ON manual_chunks(page_number);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_modality ON manual_chunks(modality);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_parent_chunk_id ON manual_chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_icon_group_id ON manual_chunks(icon_group_id);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_manual_modality_leaf
    ON manual_chunks(manual_id, modality, is_leaf);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_manual_page_modality
    ON manual_chunks(manual_id, page_number, modality);

-- 逐行注释：创建 query_logs 表，可选保存用户查询、识别结果、证据和回答，方便后续分析。
CREATE TABLE IF NOT EXISTS query_logs (
-- 逐行注释：id 是查询日志自增主键。
    id SERIAL PRIMARY KEY,
-- 逐行注释：manual_id 保存查询发生在哪本手册上。
    manual_id VARCHAR(64) NOT NULL,
-- 逐行注释：query_type 保存查询类型，例如 text 或 image。
    query_type VARCHAR(32) NOT NULL,
-- 逐行注释：question 保存用户文本问题或图片补充问题。
    question TEXT,
-- 逐行注释：image_path 保存用户上传图片路径。
    image_path VARCHAR(512),
-- 逐行注释：recognized_object 保存 VLM 或视觉匹配识别出的对象名称。
    recognized_object VARCHAR(255),
-- 逐行注释：evidence_json 保存本次查询用到的证据 JSON。
    evidence_json JSONB,
-- 逐行注释：answer 保存最终回答。
    answer TEXT,
-- 逐行注释：created_at 保存日志创建时间。
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 逐行注释：给 query_logs.manual_id 建索引，加快按手册分析查询记录。
CREATE INDEX IF NOT EXISTS idx_query_logs_manual_id ON query_logs(manual_id);
-- 逐行注释：给 query_logs.query_type 建索引，加快按文本/图片查询类型筛选。
CREATE INDEX IF NOT EXISTS idx_query_logs_query_type ON query_logs(query_type);
