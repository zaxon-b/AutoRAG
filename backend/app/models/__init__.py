# 模块说明：数据库模型模块：定义 SQLAlchemy ORM 类和数据库字段映射。
# 调用关系：由数据库初始化和业务查询调用。
# 返回/产出：产出 ORM 模型类。
# 代码说明：从 models.base 导入：Base：本文件后续逻辑依赖的对象。
from models.base import Base
# 代码说明：导出当前四类 chunk 方案使用的数据库模型。
from models.manual import Manual, ManualChunk, ManualIconGroup, ManualPage, QueryLog

# 代码说明：__all__ 表示模块公开导出列表，限制 from module import * 能导出的对象；本行右侧表达式负责得到这个值。
__all__ = [
    # 代码说明：`'Base'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    'Base',
    # 代码说明：`'Manual'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    'Manual',
    # 代码说明：`'ManualPage'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    'ManualPage',
    # 代码说明：`'ManualChunk'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    'ManualChunk',
    # 代码说明：`'ManualIconGroup'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    'ManualIconGroup',
    # 代码说明：`'QueryLog'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    'QueryLog',
# 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
]
