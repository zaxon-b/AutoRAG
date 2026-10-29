"""SQLAlchemy 声明式基类。

所有 ORM 表模型都继承 ``Base``，建表时通过 ``Base.metadata`` 汇总。
"""

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
