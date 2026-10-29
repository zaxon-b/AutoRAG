"""通用工具：数据库会话、日志和密码校验。"""

# 容器与宿主机都依赖 TZ 环境变量；进程启动时同步一次，保证日志和评测时间一致。
from time import tzset

tzset()

from .get_logger import get_logger

logger = get_logger()
