"""ReAct 主循环：LangGraph 编排版。

对照物是 agent.py 的**手写 for 循环版**。两者对外接口、事件协议、防御链、
max_turns 语义完全一致，可以互换使用（见 tests/test_agent_langgraph_parity.py
的逐项等价断言）。

一、图结构（`python agent_langgraph.py` 可打印 Mermaid 图）

    START -> agent ──(无 tool_calls)──────────────> END
               │
               └──(有 tool_calls)──> tools ──(turns < max)──> agent
                                        │
                                        └──(turns >= max)──> force_summary -> END

二、和手写版的差异（面试对照讲点）

  1. 控制流从"命令式的 while/for + if 出口"变成"声明式的节点 + 条件边"。
     手写版的循环终止条件散落在 for 上界和 return 三处；LangGraph 版把
     "下一步去哪"收敛成 _route_after_agent / _route_after_tools 两个纯函数，
     变成可单测的决策逻辑——但要付出一层框架抽象的代价。

  2. 状态显式化。手写版的 messages / tool_trace 是函数内局部变量，随调用结束
     消失；LangGraph 版把它们放进 AgentState，每一步都可以被外部 checkpoint
     读到（这也是 LangGraph 支持断点续跑 / human-in-the-loop 的前提）。

  3. 消息用 `operator.add` reducer 追加，**刻意保留 OpenAI 原始 dict**，
     不用 add_messages（那会转成 LangChain 的 BaseMessage 对象）。这样两种实现
     的 messages 结构逐字段可比，等价性可断言，也避免多一层消息转换。

  4. 两层上限保护：max_turns 是**业务语义**的轮次上限（本模块），
     recursion_limit 是**图执行**的步数上限（LangGraph 运行时）。
     两者要一起设，否则调大 max_turns 会被 recursion_limit 先掐死。

三、刻意保留的部分（不要为了"用框架"而丢掉的工程资产）

  - 三层防御链（JSON 解析 -> 工具白名单 -> 执行异常）原样保留。这种"错误也
    是一条合法 observation，交给模型自愈"的设计是 Agent 稳定性的关键，
    和编排框架无关。
  - System Prompt、工具 schema、event_hook 事件协议（tool_call / tool_result /
    answer_chunk / answer_done）与手写版逐字一致，保证前端 SSE 与评估脚本
    （eval.py）零改动。
"""
from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import AgentResult
from tools.base import BaseTool, to_openai_schema

if TYPE_CHECKING:  # 只做类型提示，不产生运行时依赖（与 agent.py 保持一致）
    from llm_client import LLMClient

# max_turns 兜底提示词：与 agent.py 逐字一致
LIMIT_HINT = "信息收集已达上限，请基于已有信息直接给出总结回答。"


class AgentState(TypedDict):
    """图状态。

    - messages / tool_trace 用 operator.add 追加（LangGraph 的 reducer 通道）
    - turns 是覆盖式通道：节点返回什么就是什么
    - answer 初值 None，非 None 即代表已有终态答案（用 `is not None` 判断，
      而不是判断空串——模型可能返回 content=None 的空答案）
    """

    messages: Annotated[list, operator.add]
    tool_trace: Annotated[list, operator.add]
    turns: int
    answer: Optional[str]


def execute_tool_safely(tool_map: dict, name: str, raw_arguments: str) -> str:
    """三层防御链，与 agent.py 的 Agent._execute_tool 语义逐行对齐。

    任何一层失败都不中断循环——错误本身就是有效的 observation，
    模型看到错误会自行调整参数或换工具重试（Agent 自愈）。
    """
    # 防御 1：arguments 是 JSON 字符串，可能不合法
    try:
        params = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": f"参数不是合法 JSON: {raw_arguments}"},
                          ensure_ascii=False)

    # 防御 2：模型可能幻觉出注册表里不存在的工具
    tool = tool_map.get(name)
    if tool is None:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    # 防御 3：工具执行可能抛异常
    try:
        return tool.call(params)
    except Exception as e:
        return json.dumps({"error": f"工具执行出错: {e}"}, ensure_ascii=False)


class LangGraphAgent:
    """和 agent.Agent 同签名的 LangGraph 实现。"""

    def __init__(self, llm: "LLMClient", tools: list[BaseTool],
                 system_prompt: str, max_turns: int = 8,
                 event_hook=None):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.tool_map = {t.name: t for t in tools}
        # 工具 schema 是静态的，编译期算一次就够了；放进 state 每轮复制纯属浪费
        self._schemas = [to_openai_schema(t) for t in tools]
        self.event_hook = event_hook  # Router 会运行期覆盖它，必须是可变属性
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ 构图

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("agent", self._node_agent)
        builder.add_node("tools", self._node_tools)
        builder.add_node("force_summary", self._node_force_summary)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", self._route_after_agent,
            {"tools": "tools", "end": END})
        builder.add_conditional_edges(
            "tools", self._route_after_tools,
            {"agent": "agent", "force_summary": "force_summary"})
        builder.add_edge("force_summary", END)
        return builder.compile()

    # ------------------------------------------------------------------ 节点

    def _node_agent(self, state: AgentState) -> dict:
        """② 发起请求（带全部工具 schema）。

        出口在这里分叉：没有 tool_calls 说明模型给出了最终答案。
        """
        resp = self.llm.chat(state["messages"], tools=self._schemas)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            self._emit("answer_chunk", {"content": answer})
            self._emit("answer_done", {})
            return {"answer": answer}

        # ⑥ 前半：assistant 消息先回填（含 tool_calls 声明）
        return {"messages": [{
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump(exclude_none=True)
                           for tc in msg.tool_calls],
        }]}

    def _node_tools(self, state: AgentState) -> dict:
        """④ 执行工具 + ⑥ 后半回填。

        并行 tool_calls 会在这里被逐个执行（每个都要有配对的 tool 消息）。
        """
        last = state["messages"][-1]
        new_messages, trace = [], []

        for tc in last["tool_calls"]:
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments") or "{}"
            params = self._safe_parse(raw)

            self._emit("tool_call", {"name": name, "params": params})
            result = execute_tool_safely(self.tool_map, name, raw)
            self._emit("tool_result", {"name": name, "result": result})

            trace.append({"name": name, "params": params, "result": result[:200]})
            # ⑥ 后半：tool 消息靠 tool_call_id 与声明配对
            new_messages.append({"role": "tool",
                                 "tool_call_id": tc["id"],
                                 "content": result})

        return {"messages": new_messages, "tool_trace": trace,
                "turns": state["turns"] + 1}

    def _node_force_summary(self, state: AgentState) -> dict:
        """max_turns 兜底：不带 tools 再调一次，强制基于已有信息总结。"""
        messages = list(state["messages"]) + [
            {"role": "user", "content": LIMIT_HINT}]
        resp = self.llm.chat(messages)
        answer = resp.choices[0].message.content or ""
        self._emit("answer_chunk", {"content": answer})
        self._emit("answer_done", {})
        return {"answer": answer}

    # ------------------------------------------------------------------ 路由

    def _route_after_agent(self, state: AgentState) -> str:
        """决策逻辑从循环里抽出来，变成可单测的纯函数。"""
        return "end" if state["answer"] is not None else "tools"

    def _route_after_tools(self, state: AgentState) -> str:
        return "force_summary" if state["turns"] >= self.max_turns else "agent"

    # ------------------------------------------------------------------ 运行

    def run(self, user_message: str, history: list = None) -> AgentResult:
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:  # 会话内多轮记忆：把之前轮次的 user/assistant 消息拼进去
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # recursion_limit 是图步数上限（每过一个节点算一步），
        # 一轮对话最多 2 步（agent + tools），再留出兜底与收尾的余量。
        config = {"recursion_limit": self.max_turns * 2 + 5}
        final = self.graph.invoke(
            {"messages": messages, "tool_trace": [], "turns": 0, "answer": None},
            config=config)

        return AgentResult(answer=final["answer"] or "",
                           tool_trace=final["tool_trace"])

    # ------------------------------------------------------------------ 工具

    def _emit(self, etype: str, data: dict):
        """事件发射：SSE 流式时前端实时收到 tool_call/tool_result/answer。"""
        if self.event_hook:
            self.event_hook(etype, data)

    def mermaid(self) -> str:
        """导出 Mermaid 图（写文档 / 答辩画架构图直接贴）。"""
        return self.graph.get_graph().draw_mermaid()

    @staticmethod
    def _safe_parse(raw: str):
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return raw


if __name__ == "__main__":
    # 打印图结构，不调 LLM，不消耗 API
    from tools import TOOL_REGISTRY

    agent = LangGraphAgent(llm=None, tools=list(TOOL_REGISTRY.values()),
                           system_prompt="", max_turns=8)
    print(agent.mermaid())
