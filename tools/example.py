"""运维工具：天气查询（练手）+ 日志查询（SQLite 版）。

QueryLogTool 改造：从 Python 常量改成 SQLite 查询（db/db.py 的 query_logs）。
保留 call 接口不变，Agent 和 Router 无感知。
"""
import json
import sys
from pathlib import Path

from tools import register_tool
from tools.base import BaseTool

# 让 tools 能找到 db 模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from db.db import query_logs


@register_tool
class WeatherTool(BaseTool):
    name = "get_current_weather"
    description = "当用户询问某地当前天气、气温时调用此工具"
    parameters = [{"name": "location", "type": "string",
                   "description": "城市名，如：北京", "required": True}]

    def call(self, params: dict) -> str:
        location = params["location"]
        result = {"location": location, "temperature": "26", "condition": "晴"}
        return json.dumps(result, ensure_ascii=False)


@register_tool
class QueryLogTool(BaseTool):
    name = "query_log"
    description = ("当用户询问服务报错、故障排查、异常分析等需要查看历史系统日志时调用，"
                   "按关键词过滤日志记录。"
                   "支持按关键词（服务名如 nginx/mysql，或错误类型如 upstream/timeout/memory）"
                   "查询，返回匹配的日志条目。"
                   "故障排查时与 get_monitor_metrics 配合：日志看现象，监控看资源。")
    parameters = [
        {"name": "keyword", "type": "string",
         "description": "日志关键词，如服务名（nginx/mysql/redis）、"
                        "错误类型（upstream/timeout/memory/killed/connections），"
                        "大小写不敏感", "required": True},
        {"name": "hours", "type": "integer",
         "description": "回溯时间范围（小时），默认 72", "required": False},
    ]

    def call(self, params: dict) -> str:
        keyword = params["keyword"]
        hours = params.get("hours", 72)  # 可选参数：模型可能不传，不能直接 []

        # 调 SQLite 查询（生产化换 ELK 时只改这里）
        logs = query_logs(keyword=keyword, hours=hours, limit=20)

        if not logs:
            return json.dumps(
                {"result": "未找到相关日志", "keyword": keyword, "hours": hours},
                ensure_ascii=False)

        # 格式化成易读的文本（LLM 看 observation 更容易理解）
        formatted = []
        for log in logs:
            formatted.append(
                f"{log['timestamp']} [{log['level']}] {log['service']} "
                f"({log['host']}): {log['message']}")

        return json.dumps(
            {"keyword": keyword, "hours": hours, "count": len(logs),
             "logs": formatted},
            ensure_ascii=False)
