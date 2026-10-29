"""统一日志格式与等级配置。"""

import logging
import colorlog
import os


def get_logger():
    """返回带颜色和调用函数名的 logger。

    等级由 LOG_LEVEL 环境变量控制，未配置时使用 INFO。
    """

    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEFAULT')

    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s.%(msecs)03d - %(levelname)s - [%(funcName)s] - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    handler.setFormatter(formatter)

    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)

    LOG_LEVEL_OPTION = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'DEFAULT': logging.INFO
    }
    logger.setLevel(LOG_LEVEL_OPTION.get(LOG_LEVEL.upper(), 'DEFAULT'))

    return logger
