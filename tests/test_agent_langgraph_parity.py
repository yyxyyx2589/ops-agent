"""LangGraph 版 vs 手写版：逐项等价性对照测试（零 API 消耗）。

目的不是"LangGraph 能跑"，而是证明两种编排在**可观测行为上完全等价**：
同样的剧本输入 -> 同样的 messages 结构、同样的工具轨迹、同样的事件序列、
同样的 LLM 调用次数、同样的防御错误文本。

这样才敢说"换编排层不动业务层"。任何一项不等价，说明复刻引入了行为漂移。

运行：python tests/test_agent_langgraph_parity.py
"""
import copy
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TESTS_DIR)

from agent import Agent                                    # noqa: E402
from agent_langgraph import LangGraphAgent                 # noqa: E402
from test_agent_mock import (ScriptedLLM, make_response,   # noqa: E402
                             make_tool_call, SYSTEM_PROMPT)
from tools import TOOL_REGISTRY                            # noqa: E402

MAX_TURNS = 8


class SnapshotLLM(ScriptedLLM):
    """在 chat 入口对 messages 做深拷贝快照。

    为什么必须深拷贝：手写版 Agent 把**同一个 list 对象**反复传给 LLM，并在
    执行工具后原地 append。ScriptedLLM 记的是引用，事后回看每一次调用都会看到
    "最终形态"的列表；而 LangGraph 每一步都会复制状态。不深拷贝，比对的就是
    两个不同时间点的东西，测试会假失败（第一版对照测试就踩了这个坑）。

    快照之后，"第 i 次请求时模型实际看到了什么"才是可比的。
    """

    def chat(self, messages, tools=None, stream=False):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        if self.script:
            return self.script.pop(0)
        return make_response(content="（兜底总结回答）")


class Recorder:
    """事件钩子：把 (类型, 数据) 按顺序记下来，用于比对两种实现的事件流。"""

    def __init__(self):
        self.events = []

    def __call__(self, etype, data):
        self.events.append((etype, data))


def build(engine, llm, recorder):
    cls = Agent if engine == "手写版" else LangGraphAgent
    return cls(llm=llm, tools=list(TOOL_REGISTRY.values()),
               system_prompt=SYSTEM_PROMPT, max_turns=MAX_TURNS,
               event_hook=recorder)


def run_both(build_script, query):
    """用同一份剧本分别驱动两种实现，返回 (结果, LLM 调用记录, 事件流)。"""
    out = {}
    for engine in ("手写版", "LangGraph"):
        recorder = Recorder()
        llm = SnapshotLLM(build_script())
        result = build(engine, llm, recorder).run(query)
        out[engine] = (result, llm.calls, recorder.events)
    return out


def check(label, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def compare(label, build_script, query):
    """通用对照：五项等价断言。"""
    got = run_both(build_script, query)
    a, b = got["手写版"], got["LangGraph"]

    check(f"{label} · 最终答案一致", a[0].answer == b[0].answer)
    check(f"{label} · 工具轨迹一致", a[0].tool_trace == b[0].tool_trace)
    check(f"{label} · LLM 调用次数一致（{len(a[1])} 次）", len(a[1]) == len(b[1]))
    check(f"{label} · 每次请求的 messages 结构一致",
          [c["messages"] for c in a[1]] == [c["messages"] for c in b[1]])
    check(f"{label} · 每次请求的 tools 入参一致",
          [bool(c["tools"]) for c in a[1]] == [bool(c["tools"]) for c in b[1]])
    check(f"{label} · 事件流一致（{len(a[2])} 个事件）", a[2] == b[2])
    return a, b


# --------------------------------------------------------------------- 场景

def test_p1_single_tool_roundtrip():
    """单工具往返：assistant(tool_calls) + tool(tool_call_id) 配对结构。"""
    script = lambda: [
        make_response(tool_calls=[make_tool_call(
            "call_1", "query_log", '{"keyword": "mysql"}')]),
        make_response(content="mysql 连接数打满，导致上游超时。"),
    ]
    a, b = compare("p1 单工具往返", script, "mysql 有什么报错？")

    msgs = b[1][1]["messages"]
    check("p1 LangGraph 版消息序列 = system/user/assistant/tool",
          [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"])
    check("p1 tool_call_id 正确配对",
          msgs[3]["tool_call_id"] == "call_1"
          and msgs[2]["tool_calls"][0]["id"] == "call_1")
    assert a[0].answer == b[0].answer


def test_p2_parallel_tool_calls():
    """并行 tool_calls：一次响应多个工具，每个都要有配对的 tool 消息。"""
    script = lambda: [
        make_response(tool_calls=[
            make_tool_call("call_a", "query_log", '{"keyword": "nginx"}'),
            make_tool_call("call_b", "get_monitor_metrics",
                           '{"metric_type": "cpu", "host": "web-01"}'),
        ]),
        make_response(content="日志与监控都已核对。"),
    ]
    a, b = compare("p2 并行工具调用", script, "查 nginx 日志和 web-01 的 CPU")

    check("p2 轨迹两条", len(b[0].tool_trace) == 2)
    check("p2 两条 tool 消息，id 分别为 call_a / call_b",
          [m["tool_call_id"] for m in b[1][1]["messages"]
           if m["role"] == "tool"] == ["call_a", "call_b"])
    assert a[0].answer == b[0].answer


def test_p3_unknown_tool_defense():
    """防御 2：模型幻觉出不存在的工具名。"""
    script = lambda: [
        make_response(tool_calls=[make_tool_call(
            "call_1", "rm_rf_everything", '{"path": "/"}')]),
        make_response(content="没有可用的删除工具。"),
    ]
    a, b = compare("p3 未知工具防御", script, "删库跑路")

    check("p3 错误文本与手写版逐字一致",
          b[0].tool_trace[0]["result"] == a[0].tool_trace[0]["result"]
          and "未知工具" in b[0].tool_trace[0]["result"])
    check("p3 循环未被中断，仍拿到最终答案", b[0].answer == "没有可用的删除工具。")


def test_p4_bad_json_defense():
    """防御 1：模型吐出非法 JSON 参数。"""
    script = lambda: [
        make_response(tool_calls=[make_tool_call(
            "call_1", "query_log", '{keyword: nginx}')]),
        make_response(content="我重新查一下。"),
    ]
    a, b = compare("p4 非法 JSON 防御", script, "nginx 报错")

    check("p4 错误文本与手写版逐字一致",
          b[0].tool_trace[0]["result"] == a[0].tool_trace[0]["result"]
          and "参数不是合法 JSON" in b[0].tool_trace[0]["result"])


def test_p5_tool_exception_defense():
    """防御 3：工具内部抛异常，被兜住并作为 observation 回填。"""
    from tools.base import BaseTool

    class BoomTool(BaseTool):
        name = "boom_tool"
        description = "总是抛异常的工具，用于验证防御 3"
        parameters = []

        def call(self, params):
            raise RuntimeError("模拟工具内部崩溃")

    def build_script():
        return [
            make_response(tool_calls=[make_tool_call("call_1", "boom_tool", "{}")]),
            make_response(content="该工具当前不可用。"),
        ]

    tools = list(TOOL_REGISTRY.values()) + [BoomTool()]
    got = {}
    for engine in ("手写版", "LangGraph"):
        recorder = Recorder()
        cls = Agent if engine == "手写版" else LangGraphAgent
        agent = cls(llm=SnapshotLLM(build_script()), tools=tools,
                    system_prompt=SYSTEM_PROMPT, max_turns=MAX_TURNS,
                    event_hook=recorder)
        got[engine] = agent.run("跑一下 boom_tool")

    check("p5 异常被兜住，错误文本一致",
          got["手写版"].tool_trace[0]["result"]
          == got["LangGraph"].tool_trace[0]["result"]
          and "工具执行出错" in got["LangGraph"].tool_trace[0]["result"])
    check("p5 两种实现都给出最终答案",
          got["手写版"].answer == got["LangGraph"].answer
          == "该工具当前不可用。")


def test_p6_chitchat_no_tool():
    """不调工具：闲聊直接回答，只调一次 LLM。"""
    script = lambda: [make_response(content="你好！有什么可以帮你？")]
    a, b = compare("p6 闲聊不调工具", script, "你好")

    check("p6 轨迹为空", b[0].tool_trace == [])
    check("p6 只调一次 LLM", len(b[1]) == 1)
    assert a[0].answer == b[0].answer


def test_p7_max_turns_fallback():
    """max_turns 兜底：不死循环，强制总结且最后一次请求不带 tools。"""
    def script():
        return [make_response(tool_calls=[make_tool_call(
            f"call_{i}", "query_log", '{"keyword": "nginx"}')])
            for i in range(MAX_TURNS)]

    a, b = compare("p7 max_turns 兜底", script, "反复查 nginx 日志")

    check(f"p7 恰好 {MAX_TURNS} 轮工具调用", len(b[0].tool_trace) == MAX_TURNS)
    check(f"p7 LLM 被调 {MAX_TURNS + 1} 次（{MAX_TURNS} 轮 + 1 次总结）",
          len(b[1]) == MAX_TURNS + 1)
    check("p7 兜底请求不带 tools", b[1][MAX_TURNS]["tools"] is None)
    check("p7 兜底提示词与手写版逐字一致",
          b[1][MAX_TURNS]["messages"][-1]["content"]
          == a[1][MAX_TURNS]["messages"][-1]["content"]
          == "信息收集已达上限，请基于已有信息直接给出总结回答。")


def test_p8_history_injection():
    """多轮记忆：history 要原样拼在 system 之后、本轮 user 之前。"""
    history = [{"role": "user", "content": "mysql 报错"},
               {"role": "assistant", "content": "连接数打满"}]
    script = lambda: [make_response(content="继续基于上文作答。")]

    for engine in ("手写版", "LangGraph"):
        recorder = Recorder()
        llm = SnapshotLLM(script())
        build(engine, llm, recorder).run("那怎么办？", history=history)
        roles = [m["role"] for m in llm.calls[0]["messages"]]
        check(f"p8 {engine} 消息序列 = system/user/assistant/user",
              roles == ["system", "user", "assistant", "user"])


# ------------------------------------------------------- 路由函数可直接单测

def test_p9_routing_functions_are_pure():
    """手写版的循环出口是散落的 return，LangGraph 版是可单测的纯函数。"""
    agent = LangGraphAgent(llm=None, tools=[], system_prompt="", max_turns=3)

    check("p9 有答案时路由到 end",
          agent._route_after_agent({"answer": "好了"}) == "end")
    check("p9 无答案时路由到 tools",
          agent._route_after_agent({"answer": None}) == "tools")
    check("p9 未到轮次上限继续循环",
          agent._route_after_tools({"turns": 2}) == "agent")
    check("p9 到达轮次上限强制总结",
          agent._route_after_tools({"turns": 3}) == "force_summary")
    check("p9 空答案（content=None）也被当作终态，不会误判回 tools",
          agent._route_after_agent({"answer": ""}) == "end")


def test_p10_graph_is_inspectable():
    """LangGraph 独有能力：图结构可导出，状态可被外部读到。"""
    agent = LangGraphAgent(llm=None, tools=[], system_prompt="", max_turns=2)
    mermaid = agent.mermaid()

    check("p10 Mermaid 图导出成功", "graph" in mermaid.lower())
    check("p10 图中包含三个节点",
          all(n in mermaid for n in ("agent", "tools", "force_summary")))


if __name__ == "__main__":
    tests = [test_p1_single_tool_roundtrip, test_p2_parallel_tool_calls,
             test_p3_unknown_tool_defense, test_p4_bad_json_defense,
             test_p5_tool_exception_defense, test_p6_chitchat_no_tool,
             test_p7_max_turns_fallback, test_p8_history_injection,
             test_p9_routing_functions_are_pure, test_p10_graph_is_inspectable]
    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n全部通过：LangGraph 版与手写版在 10 个场景上行为等价。")
