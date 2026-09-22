# 模块说明：日志工具：统一创建带格式的 logger，避免每个模块重复配置 logging。
# 调用关系：由需要写日志的服务模块调用。
# 返回/产出：返回 Python logging.Logger 对象。
# 代码说明：导入模块，logging：提供本模块需要的标准库/第三方能力。
import logging
# 代码说明：导入模块，colorlog：提供本模块需要的标准库/第三方能力。
import colorlog
# 代码说明：导入模块，os：读取环境变量，例如模型 key、阈值、数据库地址和存储目录。
import os


# 函数说明：get_logger：创建统一格式的 Python logger。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 get_logger：创建统一格式的 Python logger。
def get_logger():

    # 代码说明：LOG_LEVEL 表示日志等级配置，决定后端输出 DEBUG/INFO/WARNING 等哪一级日志；本行右侧表达式负责得到这个值。
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEFAULT')

    #########################
    ## color logger define ##
    #########################

    # Handler for logging
    # 代码说明：handler 表示handler；本行右侧表达式负责得到这个值。
    handler = colorlog.StreamHandler()
    # 代码说明：formatter 表示formatter；本行右侧表达式负责得到这个值。
    formatter = colorlog.ColoredFormatter(
        # 代码说明：`"%(log_color)s%(asctime)s.%(msecs)03d - %(levelname)s - [%(funcName)s] - %(message)s"` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
        "%(log_color)s%(asctime)s.%(msecs)03d - %(levelname)s - [%(funcName)s] - %(message)s",
        # 代码说明：datefmt 表示datefmt；本行右侧表达式负责得到这个值。
        datefmt='%Y-%m-%d %H:%M:%S',
        # 代码说明：log_colors 表示日志 / colors；本行右侧表达式负责得到这个值。
        log_colors={
            # 代码说明：`'DEBUG': 'cyan'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
            'DEBUG': 'cyan',
            # 代码说明：`'INFO': 'green'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
            'INFO': 'green',
            # 代码说明：`'WARNING': 'yellow'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
            'WARNING': 'yellow',
            # 代码说明：`'ERROR': 'red'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
            'ERROR': 'red',
            # 代码说明：`'CRITICAL': 'red,bg_white'` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
            'CRITICAL': 'red,bg_white',
        # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
        }
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    )
    # 代码说明：调用 handler.setFormatter 执行当前步骤的副作用，例如写库、建索引、保存文件或更新状态。
    handler.setFormatter(formatter)

    # Logger instance
    # 代码说明：logger 表示logger；本行右侧表达式负责得到这个值。
    logger = colorlog.getLogger(__name__)
    # 代码说明：写入运行日志，方便排查建库、检索或模型调用过程中的问题。
    logger.addHandler(handler)

    # Logger level
    # 代码说明：LOG_LEVEL_OPTION 表示logging 模块实际使用的日志等级常量；本行右侧表达式负责得到这个值。
    LOG_LEVEL_OPTION = {
        # 代码说明：设置字典键 `DEBUG` 的值，用来表达接口字段、配置项或业务映射关系。
        'DEBUG': logging.DEBUG,
        # 代码说明：设置字典键 `INFO` 的值，用来表达接口字段、配置项或业务映射关系。
        'INFO': logging.INFO,
        # 代码说明：设置字典键 `WARNING` 的值，用来表达接口字段、配置项或业务映射关系。
        'WARNING': logging.WARNING,
        # 代码说明：设置字典键 `ERROR` 的值，用来表达接口字段、配置项或业务映射关系。
        'ERROR': logging.ERROR,
        # 代码说明：设置字典键 `CRITICAL` 的值，用来表达接口字段、配置项或业务映射关系。
        'CRITICAL': logging.CRITICAL,
        # 代码说明：设置字典键 `DEFAULT` 的值，用来表达接口字段、配置项或业务映射关系。
        'DEFAULT': logging.INFO
    # 代码说明：这里结束上一段多行结构，表示参数、字段或候选列表已经填写完毕。
    }
    # 代码说明：写入运行日志，方便排查建库、检索或模型调用过程中的问题。
    logger.setLevel(LOG_LEVEL_OPTION.get(LOG_LEVEL.upper(), 'DEFAULT'))

    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return logger
