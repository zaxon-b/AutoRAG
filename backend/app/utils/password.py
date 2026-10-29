"""历史登录系统的密码哈希工具；当前精简 Demo 基本不再调用。"""

import bcrypt


def hash_password(password: str) -> str:
    """用 bcrypt 生成加盐哈希。"""

    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码是否与已保存的哈希匹配。"""

    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
