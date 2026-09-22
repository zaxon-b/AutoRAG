# 模块说明：数据库工具：创建 SQLAlchemy engine/session，并提供 FastAPI 依赖 get_db。
# 调用关系：由路由层 Depends(get_db) 和启动初始化逻辑调用。
# 返回/产出：返回数据库会话，或创建所有 ORM 表。
# 代码说明：从 sqlalchemy 导入：create_engine：本文件后续逻辑依赖的对象。
from sqlalchemy import create_engine
# 代码说明：从 sqlalchemy.orm 导入：sessionmaker：本文件后续逻辑依赖的对象。
from sqlalchemy.orm import sessionmaker
# 代码说明：从 models.base 导入：Base：本文件后续逻辑依赖的对象。
from models.base import Base
# 代码说明：导入模块，os：读取环境变量，例如模型 key、阈值、数据库地址和存储目录。
import os
# 代码说明：从 dotenv 导入：load_dotenv：本文件后续逻辑依赖的对象。
from dotenv import load_dotenv

# 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
load_dotenv()

# 代码说明：DATABASE_URL 表示PostgreSQL 连接串，告诉 SQLAlchemy 连接哪个数据库；这里调用 `getenv` 得到它，调用目的：从环境变量读取配置，让阈值、模型地址、数据库地址不写死在代码里。
DATABASE_URL = os.getenv("DATABASE_URL")
# 代码说明：engine 表示SQLAlchemy 数据库引擎，负责维护到底层 PostgreSQL 的连接能力；本行右侧表达式负责得到这个值。
engine = create_engine(DATABASE_URL)
# 代码说明：SessionLocal 表示数据库会话工厂，每个请求会用它创建一个独立 Session；本行右侧表达式负责得到这个值。
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 函数说明：get_db：FastAPI 数据库依赖：为每个请求创建 SQLAlchemy Session，并在请求结束后关闭。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 get_db：FastAPI 数据库依赖：为每个请求创建 SQLAlchemy Session，并在请求结束后关闭。
def get_db():
    # 代码说明：`"""获取数据库会话"""` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    """获取数据库会话"""
    # 代码说明：db：SQLAlchemy 数据库会话，用来读写 PostgreSQL 里的手册、页面和 chunk 元数据；右侧表达式给它赋具体值，后续流程会按这个业务含义使用。
    db = SessionLocal()
    # 代码说明：进入异常保护块：下面的外部接口、文件、数据库或模型调用可能失败，需要捕获错误。
    try:
        # 代码说明：产出一段流式数据，SSE/生成器会先把这一段发给调用方而不是等全部完成。
        yield db
    # 代码说明：最终收尾块：无论 try 成功还是失败，都执行这里释放资源或关闭 loading 状态。
    finally:
        # 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
        db.close()

# 函数说明：init_db：根据 ORM 模型创建数据库表。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 init_db：根据 ORM 模型创建数据库表。
def init_db():
    # 代码说明：`"""初始化数据库"""` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    """初始化数据库"""
    # 代码说明：Base 表示SQLAlchemy 声明式基类，所有 ORM 表模型都继承或注册到它上面；本行右侧表达式负责得到这个值。
    Base.metadata.create_all(bind=engine)
