"""监控指标查询工具（SQLite 版）。

改造：从 Python 常量 MOCK_METRICS 改成 SQLite 查询（db/db.py 的 query_metrics）。
保留 call 接口不变，Agent 无感知。

数据来源：db/ops.db 的 metrics 表，3 天每 10 分钟一个点的时序数据。
web-01 在 2026-09-04 23:00 附近出现异常峰值（CPU/内存/磁盘）。

第 4 周接真实 Prometheus API 时，只换 call 内部实现，接口不变。
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

from tools import register_tool
from tools.base import BaseTool

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.db import query_metrics


@register_tool
class MonitorTool(BaseTool):
    name = "get_monitor_metrics"
    description = ("当用户询问系统性能、资源使用、监控指标、负载情况时调用。"
                   "可查 CPU/内存/磁盘/网络四类指标，返回各主机的最新值。"
                   "故障排查时与 query_log 配合使用：日志看现象，监控看资源。"
                   "支持按主机过滤，如只查 web-01 的 CPU。")
    parameters = [
        {"name": "metric_type", "type": "string",
         "description": "指标类型：cpu / memory / disk / network", "required": True},
        {"name": "host", "type": "string",
         "description": "主机名过滤，如 web-01；不传则返回全部主机", "required": False},
    ]

    def call(self, params: dict) -> str:
        metric_type = params.get("metric_type", "").lower()
        host = params.get("host")

        valid_types = {"cpu", "memory", "disk", "network"}
        if metric_type not in valid_types:
            return json.dumps(
                {"error": f"未知指标类型: {metric_type}",
                 "available": list(valid_types)},
                ensure_ascii=False)

        # 查 SQLite：取每个主机的最新值
        rows = query_metrics(metric_type, host=host, limit=50)

        if not rows:
            return json.dumps(
                {"result": f"未找到指标数据",
                 "metric_type": metric_type, "host": host},
                ensure_ascii=False)

        # 按 host + metric_name 聚合，取每个组合的最新值
        latest = {}
        for row in rows:
            key = (row["host"], row["metric_name"])
            if key not in latest:
                latest[key] = {
                    "host": row["host"],
                    "metric_name": row["metric_name"],
                    "value": row["value"],
                    "timestamp": row["timestamp"],
                }

        # 按 host 分组组织返回
        by_host = defaultdict(list)
        for v in latest.values():
            by_host[v["host"]].append({
                "metric": v["metric_name"],
                "value": v["value"],
                "timestamp": v["timestamp"],
            })

        return json.dumps(
            {"metric_type": metric_type,
             "host_filter": host or "all",
             "hosts": {h: v for h, v in by_host.items()}},
            ensure_ascii=False)
