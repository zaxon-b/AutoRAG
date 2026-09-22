# 模块说明：密码工具：保留历史登录系统的密码哈希和校验函数。
# 调用关系：当前精简 Demo 基本不调用，属于原项目遗留工具。
# 返回/产出：返回哈希字符串或密码是否匹配的布尔值。
# 代码说明：导入模块，bcrypt：提供本模块需要的标准库/第三方能力。
import bcrypt

# 函数说明：hash_password：对历史登录密码做 bcrypt 哈希。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 hash_password：对历史登录密码做 bcrypt 哈希。
def hash_password(password: str) -> str:
    # 代码说明：`"""对密码进行哈希处理"""` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    """对密码进行哈希处理"""
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# 函数说明：verify_password：验证明文密码是否匹配历史哈希。
# 被谁调用：本模块内部函数、其他 service、路由层或评测脚本按需调用。
# 返回内容：根据函数体 return/yield 返回接口响应、证据列表、向量、文件路径、模型结果或统计指标。
# 代码说明：定义函数 verify_password：验证明文密码是否匹配历史哈希。
def verify_password(plain_password: str, hashed_password: str) -> bool:
    # 代码说明：`"""验证密码是否匹配"""` 是当前集合/列表中的业务枚举值，会参与过滤、匹配或配置。
    """验证密码是否匹配"""
    # 代码说明：返回当前函数的最终结果，结束本函数执行并把数据交给调用方。
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
