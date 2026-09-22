import os
from pathlib import Path


def _load_dotenv():
    """从 .env 文件加载环境变量（不覆盖已有值）。零依赖手动实现。

    用法：把 DEEPSEEK_API_KEY=sk-xxx 写进项目根目录的 .env 文件
    （已被 .gitignore 排除，不会误提交）。双击运行/换终端都不用重新 export。
    """
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # 若报 model not found，以 DeepSeek 文档页列出的模型名为准
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


def require_api_key() -> str:
    """运行时校验 API key（不在 import 时 raise——import 副作用会破坏可测试性，
    mock 测试和路由测试不应被真实 key 绑架；只有真正要调 LLM 时才校验）。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "请先设置环境变量 DEEPSEEK_API_KEY，或在项目根目录 .env 文件中写入"
            " DEEPSEEK_API_KEY=sk-xxx（参考 .env.example）")
    return DEEPSEEK_API_KEY

# 决策路由阈值（沿用原 RAG 项目调参结果，第 2 周从配置文件升级为路由信号）
SIM_HIGH = 0.70        # >= 高置信：直接走 RAG 直答（模型只需组织语言）
SIM_LOW = 0.45         # < 低置信：拒答转人工；[SIM_LOW, SIM_HIGH) 走 Agent 循环
KW_COVERAGE_LOW = 0.30  # 关键词覆盖率下限，与 sim 联合判定高置信
