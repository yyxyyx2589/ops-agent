"""决策路由测试：用假 LLM 验证三条路由分支。

sim 数值分布（mock 覆盖率算法实测）：
  mysql too many connections 怎么办  -> sim=0.792 -> rag_direct
  nginx 502 bad gateway 怎么排查    -> sim=0.667 -> agent_loop（中区间）
  今天午饭吃什么                     -> sim=0.000 -> refuse_human

运行：python tests/test_router_mock.py
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent  # noqa: E402
from router import Router  # noqa: E402
from tools import TOOL_REGISTRY  # noqa: E402

SYSTEM_PROMPT = "你是运维助手。"


def make_response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class StubLLM:
    """记录调用次数，返回固定文本（Agent 循环里只走'不调工具'分支）。"""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, stream=False):
        self.calls += 1
        return make_response(content="[stub answer]")


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    return cond


def build_router(llm, sim_high=0.70, sim_low=0.45, kw_low=0.30):
    tools = list(TOOL_REGISTRY.values())
    agent = Agent(llm=llm, tools=tools,
                 system_prompt=SYSTEM_PROMPT, max_turns=8)
    rag_tool = TOOL_REGISTRY["rag_search"]
    return Router(rag_tool=rag_tool, agent=agent,
                 sim_high=sim_high, sim_low=sim_low,
                 kw_coverage_low=kw_low)


def test_rag_direct():
    """高置信：'mysql too many connections' 命中 kb-002，走 RAG 直答。"""
    llm = StubLLM()
    router = build_router(llm)
    result = router.route("mysql too many connections 怎么办")

    assert check("rag_direct 路由命中", result.route == "rag_direct")
    assert check("rag_direct sim 超过 0.70", result.sim >= 0.70)
    assert check("rag_direct 未进 Agent 循环", result.tool_trace == [])
    assert check("rag_direct 有溯源片段", len(result.kb_hits) >= 1)
    assert check("rag_direct LLM 只调一次（组织语言）", llm.calls == 1)


def test_refuse_human():
    """超纲问题：'午饭吃什么' 低置信，拒答转人工，不调 LLM。"""
    llm = StubLLM()
    router = build_router(llm)
    result = router.route("今天午饭吃什么")

    assert check("refuse_human 路由命中", result.route == "refuse_human")
    assert check("refuse_human sim 低于 0.45", result.sim < 0.45)
    assert check("refuse_human 不调 LLM", llm.calls == 0)
    assert check("refuse_human 答案含转人工提示", "转人工" in result.answer)


def test_agent_loop():
    """中置信：sim 在 [0.45, 0.70) 之间，走 Agent 循环。

    向量检索后 sim 值和 n-gram 不同，用'nginx 报错了'触发中置信区间。
    """
    llm = StubLLM()
    router = build_router(llm)
    result = router.route("nginx 报错了")

    assert check("agent_loop 路由命中", result.route == "agent_loop")
    assert check("agent_loop sim 在中区间 [0.45, 0.70)",
                 0.45 <= result.sim < 0.70)
    assert check("agent_loop 调用了 LLM（进 Agent 循环）", llm.calls >= 1)
    print(f"    (sim={result.sim:.3f}, kw={result.kw_coverage:.3f})")


if __name__ == "__main__":
    for fn in [test_rag_direct, test_refuse_human, test_agent_loop]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n全部通过：决策路由三条分支逻辑正确。")
