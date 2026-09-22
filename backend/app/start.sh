#!/bin/bash
# 模块说明：start.sh 是后端容器启动脚本，负责先等待 PostgreSQL 可连接，再启动 FastAPI 服务。
# 调用关系：Dockerfile 的 ENTRYPOINT 和 docker-compose.yml 的 entrypoint 会调用这个脚本。
# 返回/产出：数据库可用时执行传入的 uvicorn 命令；数据库不可用时退出容器。
# 逐行注释：第一行 shebang 声明这个脚本用 bash 解释器执行，必须放在文件第一行。

# 逐行注释：在容器日志里打印启动脚本标题，方便开发者确认脚本已执行。
echo "=== 应用启动脚本 ==="

# 等待数据库就绪
# 逐行注释：提示当前正在等待数据库连接。
echo "等待数据库连接..."
# 逐行注释：运行一段内嵌 Python 代码，用 psycopg2 反复尝试连接 PostgreSQL。
python -c "
# 逐行注释：导入 time 模块，用于每次重试之间 sleep 等待。
import time
# 逐行注释：导入 os 模块，用于读取 DATABASE_URL 环境变量。
import os
# 逐行注释：导入 psycopg2，用于连接 PostgreSQL 数据库。
import psycopg2
# 逐行注释：导入连接失败异常类型，用于只捕获数据库连接错误。
from psycopg2 import OperationalError

# 逐行注释：最多重试 30 次，避免数据库一直不可用时无限等待。
max_retries = 30
# 逐行注释：记录当前已经重试了多少次。
retry_count = 0

# 逐行注释：只要重试次数还没到上限，就继续尝试连接数据库。
while retry_count < max_retries:
# 逐行注释：开始捕获本次连接尝试可能出现的异常。
    try:
# 逐行注释：从环境变量 DATABASE_URL 中读取连接串并尝试连接 PostgreSQL。
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
# 逐行注释：连接成功后立刻关闭，因为这里只是健康检查。
        conn.close()
# 逐行注释：打印连接成功日志。
        print('数据库连接成功!')
# 逐行注释：连接成功后跳出 while 循环。
        break
# 逐行注释：如果 PostgreSQL 还没启动好，就会进入这个异常分支。
    except OperationalError:
# 逐行注释：重试次数加一。
        retry_count += 1
# 逐行注释：打印当前等待进度，方便在 Docker 日志中观察。
        print(f'等待数据库... ({retry_count}/{max_retries})')
# 逐行注释：等待 2 秒再重试，避免疯狂请求数据库。
        time.sleep(2)
# 逐行注释：while 正常跑完但没有 break，说明数据库一直连接失败。
else:
# 逐行注释：打印数据库连接失败日志。
    print('数据库连接失败!')
# 逐行注释：用退出码 1 结束脚本，让容器启动失败并提示问题。
    exit(1)
"

# 逐行注释：数据库已经可连接，准备启动真正的 FastAPI 应用服务。
echo "启动应用服务..."
# 逐行注释：exec 会用 docker-compose 传进来的命令替换当前脚本进程，通常是 uvicorn 启动命令。
exec "$@" 
