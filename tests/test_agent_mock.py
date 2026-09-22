"""主循环逻辑测试：用假 LLM（ScriptedLLM）驱动真实 Agent，零 API 消耗。

覆盖 Day3 路标的 v1-v5 全部场景：
  1. 单工具往返闭环 + 消息结构正确性（tool_call_id 配对）
  2. 未知工具防御（模型幻觉出不存在的工具名）
  3. 非法 JSON 参数防御（模型生成了坏参数）
  4. 不调工具场景（闲聊直接回答）
  5. max_turns 兜底（不死循环，强制总结且不带 tools）

运行：python tests/test_agent_mock.py
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, AgentResult  # noqa: E402
from tools import TOOL_REGISTRY  # noqa: E402

SYSTEM_PROMPT = "你是运维助手。"


def make_tool_call(call_id, name, arguments):
    def dump(**kwargs):
        return {"id": call_id, "type": "function",
                "function": {"name": name, "arguments": arguments}}
    return SimpleNamespace(id=call_id, model_dump=dump,
                           function=SimpleNamespace(name=name,
                                                    arguments=arguments))


def make_response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ScriptedLLM:
    """按剧本依次返回预设响应；剧本耗尽后返回默认总结。记录每次收到的请求。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, stream=False):
        self.calls.append({"messages": messages, "tools": tools})
        if self.script:
            return self.script.pop(0)
        return make_response(content="（兜底总结回答）")


def build_agent(llm):
    return Agent(llm=llm, tools=list(TOOL_REGISTRY.values()),
                 system_prompt=SYSTEM_PROMPT, max_turns=8)


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    return cond


def test_single_tool_roundtrip():
    llm = ScriptedLLM([
        make_response(tool_calls=[make_tool_call(
            "call_1", "query_log", '{"keyword": "mysql"}')]),
        make_response(content="mysql 连接数打满，导致上游超时。"),
    ])
    result = build_agent(llm).run("mysql 有什么报错？")

    assert check("v1 最终答案返回", result.answer == "mysql 连接数打满，导致上游超时。")
    assert check("v1 trace 有一条记录", len(result.tool_trace) == 1)

    step = result.tool_trace[0]
    assert check("v1 工具名正确", step["name"] == "query_log")
    assert check("v1 结果非空且含 mysql 日志",
                 step["result"] and "mysql" in step["result"])

    # 第 2 次调用的 messages 结构：assistant(带 tool_calls) + tool(带 id 配对)
    msgs = llm.calls[1]["messages"]
    roles = [m["role"] for m in msgs]
    assert check("v1 消息序列 = system/user/assistant/tool",
                 roles == ["system", "user", "assistant", "tool"])
    assert check("v1 assistant 消息含 tool_calls 声明",
                 "tool_calls" in msgs[2])
    assert check("v1 tool 消息 tool_call_id 配对",
                 msgs[3]["tool_call_id"] == "call_1"
                 and msgs[2]["tool_calls"][0]["id"] == "call_1")


def test_unknown_tool_defense():
    llm = ScriptedLLM([
        make_response(tool_calls=[make_tool_call(
            "call_1", "rm_rf_everything", '{"path": "/"}')]),
        make_response(content="没有可用的删除工具，我不能执行该操作。"),
    ])
    result = build_agent(llm).run("删库跑路")
    assert check("v2 幻觉工具不崩溃，错误回传",
                 "未知工具" in result.tool_trace[0]["result"])
    assert check("v2 循环继续，拿到最终答案",
                 result.answer == "没有可用的删除工具，我不能执行该操作。")


def test_bad_json_defense():
    llm = ScriptedLLM([
        make_response(tool_calls=[make_tool_call(
            "call_1", "query_log", '{keyword: nginx}')]),  # 无引号，非法 JSON
        make_response(content="我重新查一下。"),
    ])
    result = build_agent(llm).run("nginx 报错")
    assert check("v3 非法参数不崩溃，错误回传",
                 "参数不是合法 JSON" in result.tool_trace[0]["result"])


def test_no_tool_chitchat():
    llm = ScriptedLLM([make_response(content="你好！有什么可以帮你？")])
    result = build_agent(llm).run("你好")
    assert check("v4 闲聊不调工具", result.tool_trace == [])
    assert check("v4 直接回答", "你好" in result.answer)
    assert check("v4 只调了一次 LLM", len(llm.calls) == 1)


def test_max_turns_fallback():
    # 剧本：8 轮全部要求调工具，模拟"死缠烂打"型问题
    script = [make_response(tool_calls=[make_tool_call(
        f"call_{i}", "query_log", '{"keyword": "nginx"}')])
        for i in range(8)]
    llm = ScriptedLLM(script)
    result = build_agent(llm).run("反复查 nginx 日志")

    assert check("v5 到顶强制总结", result.answer == "（兜底总结回答）")
    assert check("v5 共 8 轮工具调用", len(result.tool_trace) == 8)
    assert check("v5 LLM 被调 9 次（8 轮 + 1 次总结）", len(llm.calls) == 9)
    assert check("v5 总结请求不带 tools", llm.calls[8]["tools"] is None)


if __name__ == "__main__":
    for fn in [test_single_tool_roundtrip, test_unknown_tool_defense,
               test_bad_json_defense, test_no_tool_chitchat,
               test_max_turns_fallback]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n全部通过：主循环五个场景（v1-v5）逻辑正确。")
