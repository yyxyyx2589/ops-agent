"""ReAct 主循环：Function Calling 版。

对应 Day3 路标的五步实现 + 三层防御链。
时序对照（见 docs/Day3-主循环路标.md）：
  ② 请求 -> llm.chat(messages, tools=schemas)
  ③ 模型声明 tool_calls（此时什么都没执行）
  ④ 执行 -> self.tool_map[name].call(params)   <-- 全项目唯一的真实执行点
  ⑥ 回填 -> assistant 消息 + tool 消息（tool_call_id 配对）

event_hook（第 3 周 SSE 流式用）：签名 event_hook(type:str, data:dict)。
不传时行为退化为同步，现有测试不受影响。事件类型：
  tool_call / tool_result / answer_chunk / answer_done
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tools.base import BaseTool, to_openai_schema

if TYPE_CHECKING:  # 只做类型提示，不产生运行时依赖（依赖注入，方便测试和换实现）
    from llm_client import LLMClient


@dataclass
class AgentResult:
    answer: str                       # 最终答案
    tool_trace: list = field(default_factory=list)
    # 每轮一条：{"name", "params", "result"}，第 4 周评估直接复用


class Agent:
    def __init__(self, llm: LLMClient, tools: list[BaseTool],
                 system_prompt: str, max_turns: int = 8,
                 event_hook=None):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.tool_map = {t.name: t for t in tools}
        self.event_hook = event_hook

    def _emit(self, etype: str, data: dict):
        """事件发射：SSE 流式时前端实时收到 tool_call/tool_result/answer。"""
        if self.event_hook:
            self.event_hook(etype, data)

    def run(self, user_message: str, history: list = None) -> AgentResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        if history:  # 会话内多轮记忆：把之前轮次的 user/assistant 消息拼进去
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        schemas = [to_openai_schema(t) for t in self.tools]
        tool_trace = []

        for _turn in range(self.max_turns):
            # ② 发起请求：带上全部工具 schema
            resp = self.llm.chat(messages, tools=schemas)
            msg = resp.choices[0].message

            # 出口：没有 tool_calls 说明模型给出了最终答案
            if not msg.tool_calls:
                answer = msg.content or ""
                self._emit("answer_chunk", {"content": answer})
                self._emit("answer_done", {})
                return AgentResult(answer=answer, tool_trace=tool_trace)

            # ⑥ 前半：assistant 消息先回填（含 tool_calls 声明）
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump(exclude_none=True)
                               for tc in msg.tool_calls],
            })

            # 遍历 tool_calls（可能是多个，每个都要有配对的 tool 消息）
            for tc in msg.tool_calls:
                params = self._safe_parse(tc.function.arguments)
                self._emit("tool_call", {
                    "name": tc.function.name, "params": params,
                })
                result = self._execute_tool(tc)
                self._emit("tool_result", {
                    "name": tc.function.name, "result": result,
                })
                tool_trace.append({
                    "name": tc.function.name,
                    "params": params,
                    "result": result[:200],
                })
                # ⑥ 后半：tool 消息靠 tool_call_id 与声明配对
                messages.append({"role": "tool",
                                 "tool_call_id": tc.id,
                                 "content": result})

        # max_turns 兜底：不带 tools 再调一次，强制基于已有信息总结
        messages.append({"role": "user",
                         "content": "信息收集已达上限，请基于已有信息直接给出总结回答。"})
        resp = self.llm.chat(messages)
        answer = resp.choices[0].message.content or ""
        self._emit("answer_chunk", {"content": answer})
        self._emit("answer_done", {})
        return AgentResult(answer=answer, tool_trace=tool_trace)

    def _execute_tool(self, tc) -> str:
        """三层防御链：JSON 解析 -> 工具白名单 -> 执行异常。
        任何一层失败都不中断循环——错误本身就是有效的 observation，
        模型看到错误会自行调整参数或换工具重试（Agent 自愈）。"""
        name = tc.function.name
        raw = tc.function.arguments or "{}"

        # 防御 1：arguments 是 JSON 字符串，可能不合法
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps({"error": f"参数不是合法 JSON: {raw}"},
                              ensure_ascii=False)

        # 防御 2：模型可能幻觉出注册表里不存在的工具
        tool = self.tool_map.get(name)
        if tool is None:
            return json.dumps({"error": f"未知工具: {name}"},
                              ensure_ascii=False)

        # 防御 3：工具执行可能抛异常
        try:
            return tool.call(params)
        except Exception as e:
            return json.dumps({"error": f"工具执行出错: {e}"},
                              ensure_ascii=False)

    @staticmethod
    def _safe_parse(raw: str):
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return raw
