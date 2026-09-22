# router 包只放 FastAPI 路由入口：这里的子模块负责接收 HTTP 请求、做轻量参数校验，并把真正的业务处理交给 service 层。
# app_main.py 会注册这些子路由；这个文件本身不定义接口，主要用于标记 router 是一个可导入的 Python 包。
