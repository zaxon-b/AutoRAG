# 模块说明：Dockerfile 用来定义后端容器镜像怎么构建，Docker 会按顺序执行下面每一行。
# 调用关系：docker compose build/up 会读取这个文件，构建 autorag_api 后端服务镜像。
# 返回/产出：产出一个包含 Python 依赖和后端代码的 Docker 镜像。
# 逐行注释：选择 Python 3.11.7 的 slim 基础镜像，体积比完整系统镜像更小。
FROM python:3.11.7-slim

# 方案1：使用更稳定的镜像源（清华大学镜像源）
# 逐行注释：RUN 会在构建镜像时执行 shell 命令；下面三行连续命令把 Debian 软件源替换为清华镜像、更新源和安全源。
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# 或者使用官方源作为备用（如果镜像源有问题，请取消注释下面几行）
# RUN echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list && \
#     echo "deb http://security.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list && \
#     echo "deb http://deb.debian.org/debian bookworm-updates main" >> /etc/apt/sources.list

# 安装必要的构建工具和依赖，增加错误处理
# 逐行注释：先更新 apt 包索引；如果第一次因为网络抖动失败，就再尝试一次，然后安装编译、Python 头文件和 OpenCV 运行依赖。
RUN apt-get update --fix-missing || apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    python3-dev \
    make \
    libgl1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 更新 pip
# 逐行注释：升级 pip、setuptools、wheel，减少安装 Python 依赖时的兼容问题。
RUN pip install --upgrade pip setuptools wheel

# 设置工作目录
# 逐行注释：把容器里的工作目录切到 /app，后续 COPY/RUN/CMD 默认都在这里执行。
WORKDIR /app

# 拷贝依赖文件并安装
# 逐行注释：先只复制 requirements.txt，这样依赖安装层可以被 Docker 缓存复用。
COPY requirements.txt /app/
# 逐行注释：安装 requirements.txt 里的 Python 包，--no-cache-dir 可以减少镜像体积。
RUN pip install --index-url https://pypi.org/simple --no-cache-dir -r requirements.txt

# 拷贝代码
# 逐行注释：把 backend/app 目录中的后端源码复制进容器 /app。
COPY . /app

# 修改文件权限
# 逐行注释：给 /app 下文件设置可读可执行权限，避免容器启动时权限不足。
RUN chmod -R 755 /app
# 逐行注释：单独确保启动脚本 start.sh 可以执行。
RUN chmod +x /app/start.sh

# 设置启动脚本为入口点
# 逐行注释：容器启动时先执行 start.sh；compose 里的 command 会作为参数传给它。
ENTRYPOINT ["/app/start.sh"]
