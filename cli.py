import os

from llm_client import LLMClient
from config import BASE_URL, MODEL, require_api_key
from router import Router
from tools import TOOL_REGISTRY

# 编排引擎开关：react（默认，手写主循环）/ langgraph（图编排）
AGENT_ENGINE = os.environ.get("AGENT_ENGINE", "react").strip().lower()


def _agent_class():
    """按需导入编排引擎，避免没装 langgraph 时连手写版都跑不起来。"""
    if AGENT_ENGINE == "langgraph":
        from agent_langgraph import LangGraphAgent
        return LangGraphAgent
    from agent import Agent
    return Agent

SYSTEM_PROMPT = """你是企业 IT 运维助手，服务对象是公司内部工程师。

工具使用规范：
1. 故障排查、报错分析类问题，必须先用 query_log 查真实日志，禁止凭空猜测原因。
2. 一次工具结果不够就多轮查询，换不同关键词（如服务名、错误类型），直到信息足够。
3. 工具返回错误或未找到时，如实告知，可以调整关键词重试，但最多重试 2 次。
4. 闲聊、常识问题直接回答，不要调用工具。
5. 基于日志得出的结论，回答里要指出依据（引用了哪条日志）。
6. 如已有知识库先验信息，优先基于先验判断，再决定是否调工具补充。
"""

ROUTE_BADGE = {
    "rag_direct":  "[RAG 直答]",
    "agent_loop":  "[Agent 循环]",
    "refuse_human": "[拒答转人工]",
}


def build_router() -> Router:
    llm = LLMClient(api_key=require_api_key(), base_url=BASE_URL, model=MODEL)
    tools = list(TOOL_REGISTRY.values())
    agent = _agent_class()(llm=llm, tools=tools,
                           system_prompt=SYSTEM_PROMPT, max_turns=8)
    rag_tool = TOOL_REGISTRY["rag_search"]
    return Router(rag_tool=rag_tool, agent=agent)



def print_route(result):
    """打印路由结果：徽章 + 置信度 + 工具轨迹 + 答案 + 溯源。"""
    badge = ROUTE_BADGE.get(result.route, f"[{result.route}]")
    print(f"\n  {badge}  sim={result.sim:.2f}  kw_coverage={result.kw_coverage:.2f}")

    if result.tool_trace:
        for i, step in enumerate(result.tool_trace, 1):
            print(f"  [trace {i}] {step['name']}({step['params']})")
            print(f"           -> {step['result'][:120]}")
    else:
        print("  [trace] 本次未调用工具")

    if result.kb_hits:
        print("  [溯源] 命中片段：")
        for kb in result.kb_hits:
            print(f"    {kb['id']} (sim={kb['sim']}) {kb['text'][:60]}...")

    print(f"\n助手: {result.answer}")


def main():
    router = build_router()
    print("=" * 50)
    print("Ops Agent 已启动（输入 exit 退出）")
    print(f"编排引擎：{AGENT_ENGINE}（AGENT_ENGINE=langgraph 可切换）")
    print("路由模式：RAG直答 / Agent循环 / 拒答转人工")
    print("=" * 50)
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            result = router.route(user_input)
        except Exception as e:
            print(f"[错误] {e}")
            continue
        print_route(result)


if __name__ == "__main__":
    main()
