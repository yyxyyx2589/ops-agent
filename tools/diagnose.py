"""命令诊断工具（Docker 容器真实执行版）。

改造：从白名单 mock 输出改成 Docker 容器真实执行。
通过 subprocess 调 `docker exec` 在诊断容器里执行 ps/df/free/netstat 等命令。

安全设计（保留）：
  1. 命令白名单——模型生成的命令必须在白名单内，拒绝任意命令注入
  2. 参数校验——只允许白名单内的精确命令，不做字符串拼接
  3. 容器隔离——在独立容器里执行，不碰宿主机

容器管理：
  - 首次调用时自动启动诊断容器（alpine + 常用工具）
  - 容器名 ops-diagnose-sandbox，以 sleep infinity 保持运行
  - 工具调用通过 docker exec 执行
"""
import json
import subprocess
import shutil

from tools import register_tool
from tools.base import BaseTool

CONTAINER_NAME = "ops-diagnose-sandbox"
IMAGE_NAME = "ops-diagnose:latest"

# 命令白名单：key = 用户/模型指定的命令，value = 容器内实际执行的命令
# （面试可讲：白名单是精确匹配不是前缀匹配，"ps; rm -rf /" 不会通过）
ALLOWED_COMMANDS = {
    "ps": "ps aux",
    "netstat": "netstat -tlnp 2>/dev/null || ss -tlnp",
    "df": "df -h",
    "free": "free -m",
    "top": "top -bn1 | head -20",
    "uptime": "uptime",
    "dmesg": "dmesg | tail -20 2>/dev/null || echo 'dmesg not available'",
    "cat_nginx_log": "cat /var/log/nginx_error.log 2>/dev/null || echo 'no nginx log'",
    "cat_messages": "cat /var/log/messages 2>/dev/null || echo 'no messages log'",
    "ls_var_log": "ls -la /var/log/",
}


def _ensure_container():
    """确保诊断容器在运行。首次调用时自动构建+启动。"""
    # 检查 docker 是否可用
    if not shutil.which("docker"):
        raise RuntimeError("docker 命令不可用，请确保 Docker Desktop 已启动")

    # 检查容器是否已在运行
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True, text=True, timeout=10)
    if r.stdout.strip() == "true":
        return  # 已在运行

    # 检查容器是否存在（停止状态）
    r = subprocess.run(
        ["docker", "inspect", CONTAINER_NAME],
        capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        # 容器存在但没运行，启动它
        subprocess.run(["docker", "start", CONTAINER_NAME],
                       capture_output=True, timeout=10)
        return

    # 容器不存在，检查镜像
    r = subprocess.run(
        ["docker", "images", "-q", IMAGE_NAME],
        capture_output=True, text=True, timeout=10)
    if not r.stdout.strip():
        # 镜像不存在，构建
        import os
        dockerfile_dir = os.path.join(os.path.dirname(__file__), "..", "docker")
        subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, "-f",
             "diagnose.Dockerfile", "."],
            capture_output=True, text=True, timeout=120,
            cwd=dockerfile_dir)

    # 创建并启动容器
    subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER_NAME,
         "--rm", IMAGE_NAME, "sleep", "infinity"],
        capture_output=True, text=True, timeout=30)


@register_tool
class DiagnoseTool(BaseTool):
    name = "run_diagnostic_command"
    description = ("在受控沙箱容器内执行诊断命令，用于进程、网络、磁盘、内存排查。"
                   "命令在独立 Docker 容器里真实执行，不碰宿主机。"
                   "仅允许白名单命令，禁止任意命令注入。"
                   "当需要看实时系统状态、定位资源瓶颈时调用。")
    parameters = [
        {"name": "command", "type": "string",
         "description": ("要执行的命令，必须是白名单之一。"
                         "可选值：ps / netstat / df / free / top / uptime / "
                         "dmesg / cat_nginx_log / cat_messages / ls_var_log"),
         "required": True},
    ]

    def call(self, params: dict) -> str:
        command = params.get("command", "").strip()

        # 防御 1：白名单校验（核心安全设计）
        if command not in ALLOWED_COMMANDS:
            return json.dumps(
                {"error": f"命令不在白名单内，拒绝执行: {command}",
                 "allowed": list(ALLOWED_COMMANDS.keys())},
                ensure_ascii=False)

        # 防御 2：确保容器运行
        try:
            _ensure_container()
        except Exception as e:
            return json.dumps(
                {"error": f"诊断容器启动失败: {e}",
                 "hint": "请确保 Docker Desktop 已启动"},
                ensure_ascii=False)

        # 防御 3：在容器内执行命令
        actual_cmd = ALLOWED_COMMANDS[command]
        try:
            r = subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "sh", "-c", actual_cmd],
                capture_output=True, text=True, timeout=15)
            output = r.stdout or r.stderr or "(无输出)"
        except subprocess.TimeoutExpired:
            output = "(命令执行超时，15s)"
        except Exception as e:
            output = f"(执行异常: {e})"

        return json.dumps(
            {"command": command, "actual_cmd": actual_cmd,
             "container": CONTAINER_NAME, "output": output},
            ensure_ascii=False)
