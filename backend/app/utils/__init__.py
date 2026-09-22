# 模块说明：通用工具模块：提供数据库、日志、密码等基础能力。
# 调用关系：由路由层和 service 层复用。
# 返回/产出：返回工具函数结果。
# set global timezone
# 代码说明：从 time 导入：tzset：本文件后续逻辑依赖的对象。
from time import tzset
# 代码说明：这一行是当前多行表达式或控制流程的一部分，负责补充具体条件、字段或计算结果。
tzset()

# 代码说明：从 .get_logger 导入：get_logger：创建统一格式的 Python logger。
from .get_logger import get_logger
# 代码说明：logger 表示logger；本行右侧表达式负责得到这个值。
logger = get_logger()
