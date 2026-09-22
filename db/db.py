"""SQLite 数据库连接模块（共用）。

三张表：
  logs      运维日志（timestamp, level, service, host, message）
  metrics   监控指标时序（timestamp, host, metric_type, metric_name, value）
  sessions  会话记忆（session_id, role, content, created_at）

初始化：python db/init_db.py（建表 + 生成数据）
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "ops.db"


def get_conn() -> sqlite3.Connection:
    """获取数据库连接。Row 工厂让结果可按列名访问。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_cursor():
    """上下文管理器：自动提交 + 关闭。工具层用这个。"""
    conn = get_conn()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def query_logs(keyword: str = None, level: str = None, service: str = None,
               host: str = None, hours: int = 72, limit: int = 50) -> list[dict]:
    """查日志，支持关键词模糊匹配 + 多维过滤 + 时间回溯。

    面试讲点：
    1. keyword 同时匹配 service 和 message 字段——用户说"查 nginx 日志"
       时，nginx 是服务名不是消息内容，只搜 message 会漏。
    2. hours 基于 DB 最新时间回溯（预置数据时间窗口和真实当前时间可能错位，
       用 MAX(timestamp) 做基准保证故障数据不被误过滤）。
    生产化换 ELK 时，这个函数的接口不变，内部改成 ES multi_match 查询。
    """
    sql = ("SELECT timestamp, level, service, host, message FROM logs "
           "WHERE timestamp >= "
           "  (SELECT datetime(MAX(timestamp), '-' || ? || ' hours') FROM logs)")
    params = [hours]
    if keyword:
        # keyword 同时匹配 service 和 message（OR 逻辑），大小写不敏感
        kw_lower = f"%{keyword.lower()}%"
        sql += (" AND (LOWER(service) LIKE ? OR LOWER(message) LIKE ?)")
        params.append(kw_lower)
        params.append(kw_lower)
    if level:
        sql += " AND level = ?"
        params.append(level)
    if service:
        sql += " AND service = ?"
        params.append(service)
    if host:
        sql += " AND host = ?"
        params.append(host)
    # 排序：ERROR/WARN 优先（CASE 权重），同级按时间倒序
    sql += (" ORDER BY CASE level WHEN 'ERROR' THEN 0 WHEN 'WARN' THEN 1 ELSE 2 END, "
            "timestamp DESC LIMIT ?")
    params.append(limit)

    with db_cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def query_metrics(metric_type: str, host: str = None,
                  metric_name: str = None, limit: int = 10) -> list[dict]:
    """查监控指标，取最近 N 个点。

    面试讲点：时序数据按 timestamp DESC 取最新值；
    生产化换 Prometheus 时接口不变，内部改成 PromQL 查询。
    """
    sql = ("SELECT timestamp, host, metric_type, metric_name, value "
           "FROM metrics WHERE metric_type = ?")
    params = [metric_type]
    if host:
        sql += " AND host = ?"
        params.append(host)
    if metric_name:
        sql += " AND metric_name = ?"
        params.append(metric_name)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with db_cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def save_message(session_id: str, role: str, content: str):
    """保存一条会话消息。"""
    import datetime as dt
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, dt.datetime.now().isoformat()))


def load_history(session_id: str) -> list[dict]:
    """加载会话历史（按时间正序）。"""
    with db_cursor() as cur:
        cur.execute(
            "SELECT role, content FROM sessions WHERE session_id = ? ORDER BY id ASC",
            (session_id,))
        return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    # 快速自测
    print("=== query_log 测试 ===")
    nginx_err = query_logs(keyword="nginx", level="ERROR", hours=72)
    print(f"nginx ERROR 日志: {len(nginx_err)} 条")
    for r in nginx_err[:3]:
        print(f"  {r['timestamp']} {r['message'][:50]}")

    print("\n=== query_metrics 测试 ===")
    cpu = query_metrics("cpu", host="web-01", metric_name="usage_pct", limit=3)
    print(f"web-01 cpu 最新 3 个点: {cpu}")

    print("\n=== sessions 测试 ===")
    save_message("test-sid", "user", "你好")
    save_message("test-sid", "assistant", "你好，有什么可以帮你")
    hist = load_history("test-sid")
    print(f"会话历史: {hist}")
    # 清理测试数据
    with db_cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE session_id='test-sid'")
    print("测试数据已清理")
