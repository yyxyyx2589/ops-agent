"""同一问题，两种编排引擎并排跑，人工对比。

用途：答辩 / 面试演示"同一业务从手写循环演进到图编排"。
注意：会真实调用 DeepSeek API（每题两次），单次成本很低，但别放进自动化测试。

用法：
  python compare_engines.py                                  # 用内置问题
  python compare_engines.py "web-01 为什么变慢？"             # 指定问题
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import SYSTEM_PROMPT                              # noqa: E402
from config import BASE_URL, MODEL, require_api_key        # noqa: E402
from llm_client import LLMClient                           # noqa: E402
from router import Router                                  # noqa: E402
from tools import TOOL_REGISTRY                            # noqa: E402

DEFAULT_QUESTIONS = [
    "mysql 服务最近有什么报错日志？",
    "web-01 这台机器为什么变慢？",
    "公司新员工入职，邮箱怎么申请？",
]


def build_router(engine: str):
    """显式构造，避免依赖 AGENT_ENGINE 环境变量与模块重导入。"""
    if engine == "langgraph":
        from agent_langgraph import LangGraphAgent as AgentCls
    else:
        from agent import Agent as AgentCls
    llm = LLMClient(api_key=require_api_key(), base_url=BASE_URL, model=MODEL)
    agent = AgentCls(llm=llm, tools=list(TOOL_REGISTRY.values()),
                     system_prompt=SYSTEM_PROMPT, max_turns=8)
    return Router(rag_tool=TOOL_REGISTRY["rag_search"], agent=agent)


def run_one(engine: str, query: str) -> dict:
    router = build_router(engine)
    t0 = time.time()
    result = router.route(query)
    return {
        "engine": engine,
        "route": result.route,
        "sim": result.sim,
        "kw_coverage": result.kw_coverage,
        "elapsed": time.time() - t0,
        "trace": [(s["name"], s["params"]) for s in result.tool_trace],
        "answer": result.answer,
    }


def show(a: dict, b: dict):
    print(f"\n{'=' * 64}")
    print(f"路由：{a['route']}   sim={a['sim']:.2f}  kw={a['kw_coverage']:.2f}")
    print(f"{'维度':<12}{'手写版 react':<26}{'langgraph':<26}")
    print("-" * 64)
    print(f"{'耗时':<12}{a['elapsed']:.2f}s{'':<21}{b['elapsed']:.2f}s")
    print(f"{'工具调用数':<12}{len(a['trace']):<26}{len(b['trace']):<26}")
    same_trace = a["trace"] == b["trace"]
    print(f"{'工具轨迹':<12}{'一致' if same_trace else '不同（见下）'}")
    for i in range(max(len(a["trace"]), len(b["trace"]))):
        pa = a["trace"][i] if i < len(a["trace"]) else None
        pb = b["trace"][i] if i < len(b["trace"]) else None
        print(f"  [{i + 1}] react={pa}")
        print(f"      lgraph={pb}")
    print("\n说明：真实 API 每次采样不同，两次独立运行的取词顺序天然会有差异，"
          "\n      轨迹/答案不同属预期。等价性由 tests/test_agent_langgraph_parity.py"
          "\n      用固定剧本（mock LLM）逐项断言，不靠并排跑。")
    print(f"\n{'答案语义是否对齐':<12}{'是' if a['answer'] == b['answer'] else '需人工看（采样差异）'}")
    print(f"\n[手写版答案]\n{a['answer']}\n")
    print(f"[LangGraph 版答案]\n{b['answer']}\n")


def main():
    queries = sys.argv[1:] or DEFAULT_QUESTIONS
    for q in queries:
        print(f"\n{'#' * 64}\n提问：{q}")
        a = run_one("react", q)
        b = run_one("langgraph", q)
        show(a, b)


if __name__ == "__main__":
    main()
