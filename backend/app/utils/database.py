"""数据库连接与会话管理。"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.base import Base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI 依赖：每个请求一个 Session，请求结束自动关闭。"""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """按 ORM 模型建表；已存在的表不会被修改。"""

    Base.metadata.create_all(bind=engine)
