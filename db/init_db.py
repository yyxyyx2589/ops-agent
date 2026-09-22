"""数据库初始化 + 数据生成脚本。

建 SQLite schema（日志/监控/会话三张表），生成真实格式的运维数据导入。

表结构：
  logs      (id, timestamp, level, service, host, message)
  metrics   (id, timestamp, host, metric_type, metric_name, value)
  sessions  (id, session_id, role, content, created_at)

运行：python db/init_db.py
"""
import json
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "ops.db"
LOGS_DIR = Path(__file__).parent.parent / "data" / "logs"

SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,          -- ISO8601: 2026-09-04 23:12:01
    level TEXT NOT NULL,              -- ERROR / WARN / INFO
    service TEXT NOT NULL,            -- nginx / mysql / kernel / disk / redis / java
    host TEXT NOT NULL,               -- web-01 / web-02 / db-01
    message TEXT NOT NULL             -- 日志正文
);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_service ON logs(service);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,          -- ISO8601
    host TEXT NOT NULL,               -- web-01 / web-02 / db-01
    metric_type TEXT NOT NULL,        -- cpu / memory / disk / network
    metric_name TEXT NOT NULL,        -- usage_pct / load_avg_1m / used_gb ...
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_host ON metrics(host);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,               -- user / assistant
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_sid ON sessions(session_id);
"""


def init_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def gen_logs():
    """生成真实格式的运维日志，围绕'web-01 昨晚变慢'的故障链故事。

    生成 3 天的日志（2026-09-04 ~ 2026-09-06），正常日志 + 故障链日志。
    故障集中在 2026-09-04 23:00 ~ 23:45。
    """
    random.seed(42)
    logs = []

    # 故障链核心日志（固定，保证 demo 效果）
    incident_logs = [
        ("2026-09-04 23:12:01", "ERROR", "nginx", "web-01",
         "upstream timed out while reading response header from 10.0.2.15:8080"),
        ("2026-09-04 23:12:03", "ERROR", "nginx", "web-01",
         "upstream server temporarily disabled while connecting to upstream"),
        ("2026-09-04 23:15:47", "WARN", "mysql", "db-01",
         "too many connections (max=151), rejecting 10.0.2.31"),
        ("2026-09-04 23:20:11", "ERROR", "kernel", "web-01",
         "Out of memory: Killed process 3321 (java) score 890"),
        ("2026-09-04 23:31:05", "WARN", "disk", "web-01",
         "/var/lib/docker usage 92% exceeds threshold 85%"),
        ("2026-09-04 23:40:22", "INFO", "redis", "db-01",
         "connection reset by peer from 10.0.2.44"),
    ]
    logs.extend(incident_logs)

    # 正常背景日志（随机生成，让数据库有量）
    services = [
        ("nginx", "web-01", ["GET /api/health 200", "POST /api/login 200",
                              "GET /index.html 200", "GET /static/app.js 304"]),
        ("nginx", "web-02", ["GET /api/health 200", "GET /index.html 200"]),
        ("mysql", "db-01", ["slow query: SELECT * FROM orders (1.2s)",
                            "connection from 10.0.2.15 accepted",
                            "connection from 10.0.2.31 accepted"]),
        ("redis", "db-01", ["SET key:session:abc OK", "GET key:cache:user:123"]),
        ("java", "web-01", ["app.jar started", "GC pause 50ms",
                            "request processed in 120ms"]),
    ]
    levels_weight = [("INFO", 0.85), ("WARN", 0.12), ("ERROR", 0.03)]

    base = datetime(2026, 9, 4, 0, 0, 0)
    for day_offset in range(3):
        for hour in range(24):
            for minute in range(0, 60, random.choice([5, 10, 15])):
                ts = (base + timedelta(days=day_offset, hours=hour, minutes=minute)
                      ).strftime("%Y-%m-%d %H:%M:%S")
                # 故障时段少生成正常日志，避免淹没故障
                if day_offset == 0 and 23 <= hour <= 23 and minute >= 10:
                    continue
                service, host, msgs = random.choice(services)
                msg = random.choice(msgs)
                r = random.random()
                level = "INFO"
                cum = 0
                for lv, p in levels_weight:
                    cum += p
                    if r < cum:
                        level = lv
                        break
                logs.append((ts, level, service, host, msg))

    # 偶发 WARN/ERROR 散落在正常时段（让 keyword 查询有结果）
    sporadic = [
        ("2026-09-04 10:23:11", "WARN", "nginx", "web-01",
         "upstream response slow (2.1s) from 10.0.2.15:8080"),
        ("2026-09-04 14:45:22", "WARN", "mysql", "db-01",
         "aborted connection from 10.0.2.31 (timeout)"),
        ("2026-09-05 02:15:03", "ERROR", "redis", "db-01",
         "connection reset by peer from 10.0.2.44"),
        ("2026-09-05 08:30:17", "WARN", "disk", "web-01",
         "/var/lib/docker usage 85% reaches threshold"),
        ("2026-09-05 19:12:44", "WARN", "java", "web-01",
         "GC pause 800ms, heap near limit"),
        ("2026-09-06 01:22:08", "ERROR", "mysql", "db-01",
         "too many connections (max=151), rejecting 10.0.2.15"),
    ]
    logs.extend(sporadic)

    return logs


def gen_metrics():
    """生成监控指标时序数据，3 天每 10 分钟一个点。

    web-01 在 2026-09-04 23:00 附近出现异常峰值。
    """
    random.seed(42)
    metrics = []
    base = datetime(2026, 9, 4, 0, 0, 0)
    hosts = {
        "web-01": {"cpu_base": 45, "mem_base": 5.0, "disk_base": 80, "anomaly": True},
        "web-02": {"cpu_base": 30, "mem_base": 3.0, "disk_base": 40, "anomaly": False},
        "db-01": {"cpu_base": 60, "mem_base": 12.0, "disk_base": 55, "anomaly": False},
    }

    for day_offset in range(3):
        for hour in range(24):
            for minute in range(0, 60, 10):
                ts = (base + timedelta(days=day_offset, hours=hour, minutes=minute)
                      ).strftime("%Y-%m-%d %H:%M:%S")
                for host, cfg in hosts.items():
                    # 故障时段 web-01 飙高
                    cpu_mult = 1.0
                    mem_mult = 1.0
                    if cfg["anomaly"] and day_offset == 0 and hour == 23:
                        cpu_mult = 2.0
                        mem_mult = 1.4

                    cpu = round(cfg["cpu_base"] * cpu_mult + random.uniform(-5, 5), 1)
                    load = round(cpu / 12 + random.uniform(-0.3, 0.3), 2)
                    metrics.append((ts, host, "cpu", "usage_pct", cpu))
                    metrics.append((ts, host, "cpu", "load_avg_1m", load))

                    mem = round(cfg["mem_base"] * mem_mult + random.uniform(-0.3, 0.3), 2)
                    metrics.append((ts, host, "memory", "used_gb", mem))
                    metrics.append((ts, host, "memory", "total_gb", 8.0))

                    disk = round(cfg["disk_base"] + random.uniform(-1, 1) +
                                 (day_offset * 0.5), 1)
                    metrics.append((ts, host, "disk", "usage_pct", disk))

                    net_in = round(random.uniform(50, 300), 1)
                    net_out = round(random.uniform(30, 150), 1)
                    metrics.append((ts, host, "network", "in_mbps", net_in))
                    metrics.append((ts, host, "network", "out_mbps", net_out))

    # 故障时段特殊指标：web-01 OOM kills
    oom_ts = ["2026-09-04 23:20:11", "2026-09-04 23:25:00", "2026-09-04 23:40:00"]
    for ts in oom_ts:
        metrics.append((ts, "web-01", "memory", "oom_kills_24h", 3.0))

    return metrics


def export_log_files(logs):
    """把日志按服务导出成真实格式的日志文件（面试可讲'日志解析'）。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # nginx error log 格式
    nginx_logs = [l for l in logs if l[2] == "nginx"]
    with open(LOGS_DIR / "nginx_error.log", "w", encoding="utf-8") as f:
        for ts, level, service, host, msg in nginx_logs:
            f.write(f'{ts} [{level}] {msg} (host={host})\n')

    # mysql error log 格式
    mysql_logs = [l for l in logs if l[2] == "mysql"]
    with open(LOGS_DIR / "mysql_error.log", "w", encoding="utf-8") as f:
        for ts, level, service, host, msg in mysql_logs:
            f.write(f'{ts} [{level}] [pid] {msg} (host={host})\n')

    # syslog 格式（kernel/disk/redis/java）
    sys_logs = [l for l in logs if l[2] in ("kernel", "disk", "redis", "java")]
    with open(LOGS_DIR / "syslog.log", "w", encoding="utf-8") as f:
        for ts, level, service, host, msg in sys_logs:
            f.write(f'{ts} {host} {service}[pid]: [{level}] {msg}\n')


def main():
    print(f"数据库路径: {DB_PATH}")
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("已删除旧数据库")

    conn = sqlite3.connect(str(DB_PATH))
    init_schema(conn)
    print("schema 建表完成")

    # 日志
    logs = gen_logs()
    conn.executemany(
        "INSERT INTO logs (timestamp, level, service, host, message) VALUES (?,?,?,?,?)",
        logs)
    print(f"日志导入: {len(logs)} 条")

    # 监控
    metrics = gen_metrics()
    conn.executemany(
        "INSERT INTO metrics (timestamp, host, metric_type, metric_name, value) VALUES (?,?,?,?,?)",
        metrics)
    print(f"监控指标导入: {len(metrics)} 条")

    # 导出日志文件
    export_log_files(logs)
    print(f"日志文件导出到: {LOGS_DIR}")

    conn.commit()

    # 验证
    print("\n=== 验证 ===")
    print(f"logs 总数: {conn.execute('SELECT COUNT(*) FROM logs').fetchone()[0]}")
    err_count = conn.execute("SELECT COUNT(*) FROM logs WHERE level='ERROR'").fetchone()[0]
    print(f"  ERROR: {err_count}")
    nginx_count = conn.execute("SELECT COUNT(*) FROM logs WHERE service='nginx'").fetchone()[0]
    print(f"  nginx: {nginx_count}")
    print(f"metrics 总数: {conn.execute('SELECT COUNT(*) FROM metrics').fetchone()[0]}")
    cpu_count = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE host='web-01' AND metric_type='cpu' AND metric_name='usage_pct'"
    ).fetchone()[0]
    print(f"  web-01 cpu 点数: {cpu_count}")

    print("\n=== 故障时段 web-01 cpu 抽样 ===")
    for row in conn.execute(
        "SELECT timestamp, value FROM metrics WHERE host='web-01' AND metric_type='cpu' AND metric_name='usage_pct' AND timestamp LIKE '2026-09-04 23%' ORDER BY timestamp"):
        print(f"  {row[0]}  cpu={row[1]}%")

    conn.close()
    print(f"\n数据库初始化完成: {DB_PATH}")


if __name__ == "__main__":
    main()
