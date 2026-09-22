# Day 3 路标：ReAct 主循环（agent.py）——项目心脏，你亲手写

> 分层模式：本文档是你的施工图。cli.py（我已写好）定义了调用契约，你实现 agent.py 让它跑起来。卡住随时问。

## 契约（cli.py 依赖这两个东西，签名不能变）

```python
Agent(llm, tools, system_prompt, max_turns=8)
agent.run(user_message: str) -> AgentResult(answer: str, tool_trace: list)
```

## 文件骨架（imports 我给你，run 的实现你填）

```python
import json
from dataclasses import dataclass, field
from llm_client import LLMClient
from tools.base import BaseTool, to_openai_schema


@dataclass
class AgentResult:
    answer: str          # 最终答案
    tool_trace: list     # [{"name":..., "params":..., "result":...}, ...] 每轮一条


class Agent:
    def __init__(self, llm: LLMClient, tools: list[BaseTool],
                 system_prompt: str, max_turns: int = 8):
        self.llm = llm
        self.tools = tools                      # 注意：用构造传入的 tools，
        self.system_prompt = system_prompt      # 不要直接 import TOOL_REGISTRY
        self.max_turns = max_turns              # （依赖注入，方便第4周换工具子集做实验）

    def run(self, user_message: str) -> AgentResult:
        ...
```

## run() 的实现路线（按顺序填，每步都能测）

```
第 0 步：初始化
    messages = [system, user]
    schemas = [to_openai_schema(t) for t in self.tools]   # 你的 Day 2 成果
    tool_trace = []
    tool_map = {t.name: t for t in self.tools}            # 名字 → 工具实例

第 1 步：主循环（for turn in range(self.max_turns)，不是 while True）
    resp = self.llm.chat(messages, tools=schemas)
    msg = resp.choices[0].message

第 2 步：出口判断
    if not msg.tool_calls:                # 没有工具调用 = 模型给出最终答案
        return AgentResult(answer=msg.content, tool_trace=tool_trace)

第 3 步：回填 assistant 消息（坑点 1 的前半，很多人漏这步）
    messages.append({
        "role": "assistant",
        "content": msg.content,           # 可能为 None，没关系照传
        "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
    })

第 4 步：遍历 tool_calls（坑点 2：是列表，可能一轮多个）
    for tc in msg.tool_calls:
        name = tc.function.name
        raw = tc.function.arguments       # ⚠️ JSON 字符串，不是 dict（坑点 3）

        result = 执行链（见下方防御表）
        tool_trace.append({"name": name, "params": ..., "result": result[:200]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        #                                          ↑ 坑点 1 的后半：id 配对

第 5 步：循环耗尽的兜底（坑点 5）
    for 循环正常结束还没 return → 追加一条 user 消息：
    "信息收集已达上限，请基于已有信息直接给出总结回答。"
    再调一次 llm.chat(messages)（这次不传 tools），返回其 content
```

## 第 4 步的防御链（每层都要写，这是 demo4 的考点）

| 顺序 | 防御 | 处理方式 |
|---|---|---|
| 1 | `json.loads(raw)` 失败 | 不 crash，`result = json.dumps({"error": "参数不是合法JSON: ..."})` 回传，模型会自纠 |
| 2 | `name not in tool_map` | `result = json.dumps({"error": f"未知工具: {name}"})`（模型幻觉出不存在的工具） |
| 3 | `tool.call(params)` 抛异常 | `try/except Exception as e`，把错误信息也作为 result 回传 |

三层都过了才拿真正的工具结果。**任何一层失败都不中断循环**——错误本身就是一个有效的 observation，模型看到错误会调整策略重试，这就是 Agent 的自愈能力（面试讲点）。

## 你需要的 SDK 字段速查

```
resp.choices[0].message          # 本次响应的消息对象
  ├── .content                   # 文本回复（有 tool_calls 时常为 None）
  └── .tool_calls                # 列表或 None
       └── 每个元素 tc
            ├── .id                          # "call_abc123"，配对用
            ├── .type                        # "function"
            └── .function
                 ├── .name                   # 工具名
                 └── .arguments              # 参数 JSON 字符串
```

## 分阶段测试（每步跑一次，别一口气写完）

| 阶段 | 测试问题 | 预期 |
|---|---|---|
| v1：只写 0-3 步 + 单工具执行 | "北京今天多少度？" | 天气工具被调，答案含 26 度 |
| v2：加完整防御链 | "帮我查一下 k8s 的日志" | 模型收到"未找到"后如实回答，没瞎编 |
| v3：多轮循环 | "nginx 昨晚为什么报错？昨晚服务整体有什么异常？" | 多次工具调用，tool_trace 有多条记录 |
| v4：不调工具场景 | "你好" | tool_trace 为空，直接回答 |
| v5：max_turns 兜底 | 问一个死循环追查类问题 | 到 8 轮强制总结，不死循环 |

## 面试自测题（写完必须能答）

1. 为什么第 3 步要先 append assistant 消息再 append tool 消息？顺序反了会怎样？
2. `tc.model_dump()` 干了什么？为什么手写 dict 容易错？
3. 工具执行抛异常时，为什么选择"错误回传"而不是整个 run() 抛出？
4. 你的 max_turns 兜底为什么用"再调一次不带 tools"而不是直接返回半截 trace？
