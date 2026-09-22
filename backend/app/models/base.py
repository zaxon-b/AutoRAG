# 模块说明：数据库模型模块：定义 SQLAlchemy ORM 类和数据库字段映射。
# 调用关系：由数据库初始化和业务查询调用。
# 返回/产出：产出 ORM 模型类。
# 代码说明：从 sqlalchemy.ext.declarative 导入：declarative_base：本文件后续逻辑依赖的对象。
from sqlalchemy.ext.declarative import declarative_base

# 代码说明：Base 表示SQLAlchemy 声明式基类，所有 ORM 表模型都继承或注册到它上面；本行右侧表达式负责得到这个值。
Base = declarative_base()
