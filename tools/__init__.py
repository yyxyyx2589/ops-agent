


TOOL_REGISTRY = {}
def register_tool(cls):
    tool = cls()
    TOOL_REGISTRY[tool.name] = tool
    return cls

from tools import example    # 触发注册，没有这行注册表永远是空的
from tools import monitor    # 监控查询工具
from tools import diagnose   # 命令诊断工具（白名单沙箱）
from tools import rag_search # RAG 检索工具（路由信号源 + Agent 工具池）