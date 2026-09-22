"""Ops Agent Web 服务：FastAPI + SSE 流式。

事件流协议（前端 EventSource 监听）：
  event: route        data: {route, sim, kw_coverage}     路由决策
  event: tool_call    data: {name, params}                Agent 工具调用
  event: tool_result  data: {name, result}                工具结果
  event: answer_chunk data: {content}                     答案片段（打字机）
  event: answer_done  data: {}                            答案结束
  event: done         data: {route, answer, sim, ...}    完整结果（含 kb_hits）
  event: error        data: {message}                     错误

会话记忆：SQLite 持久化（db/ops.db 的 sessions 表），重启不丢。

启动：uvicorn server:app --reload --port 8000
"""
import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

# 让 server 能找到 db 模块
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from config import BASE_URL, MODEL, require_api_key
from router import Router
from tools import TOOL_REGISTRY
from db.db import save_message, load_history

SYSTEM_PROMPT = """你是企业 IT 运维助手，服务对象是公司内部工程师。

工具使用规范：
1. 故障排查、报错分析类问题，必须先用 query_log 查真实日志，禁止凭空猜测原因。
2. 一次工具结果不够就多轮查询，换不同关键词（如服务名、错误类型），直到信息足够。
3. 工具返回错误或未找到时，如实告知，可以调整关键词重试，但最多重试 2 次。
4. 闲聊、常识问题直接回答，不要调用工具。
5. 基于日志得出的结论，回答里要指出依据（引用了哪条日志）。
6. 如已有知识库先验信息，优先基于先验判断，再决定是否调工具补充。
"""

app = FastAPI(title="Ops Agent")
WEB_DIR = Path(__file__).parent / "web"

# 编排引擎开关：react（默认，手写主循环）/ langgraph（图编排）
# 两种实现同签名、同事件协议，切换不影响 Router、SSE 前端与 eval.py。
AGENT_ENGINE = os.environ.get("AGENT_ENGINE", "react").strip().lower()


def _agent_class():
    """按需导入编排引擎。

    延迟导入的两个理由：langgraph 是可选依赖（没装也能跑手写版），
    且它自身 import 有秒级开销，不该拖慢服务冷启动。
    """
    if AGENT_ENGINE == "langgraph":
        from agent_langgraph import LangGraphAgent
        return LangGraphAgent
    from agent import Agent
    return Agent


def build_router(event_hook=None) -> Router:
    llm = LLMClient(api_key=require_api_key(), base_url=BASE_URL, model=MODEL)
    tools = list(TOOL_REGISTRY.values())
    agent = _agent_class()(llm=llm, tools=tools,
                           system_prompt=SYSTEM_PROMPT, max_turns=8,
                           event_hook=event_hook)
    rag_tool = TOOL_REGISTRY["rag_search"]
    return Router(rag_tool=rag_tool, agent=agent, event_hook=event_hook)



def _result_to_dict(result) -> dict:
    return {
        "route": result.route,
        "answer": result.answer,
        "sim": result.sim,
        "kw_coverage": result.kw_coverage,
        "tool_trace": result.tool_trace,
        "kb_hits": result.kb_hits,
    }


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/report")
def report_page():
    """静态评估报告页（三基线对比）。"""
    return FileResponse(WEB_DIR / "report.html")


@app.get("/api/eval-report")
def eval_report():
    """返回 eval.py 产出的评估报告 JSON；不存在时返回空对象。"""
    report_file = Path(__file__).parent / "reports" / "eval_report.json"
    if not report_file.exists():
        return JSONResponse({})
    with open(report_file, encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.get("/api/chat/stream")
def chat_stream(query: str = Query(...), session_id: str = Query("")):
    """SSE 流式对话。GET + EventSource，query 作为 query param。

    会话记忆：SQLite 持久化，每次对话从 DB 加载历史 + 保存本轮。
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:8]

    # 从 SQLite 加载会话历史
    history = load_history(session_id)

    q: queue.Queue = queue.Queue()

    def worker():
        try:
            router = build_router(
                event_hook=lambda t, d: q.put((t, d)))
            result = router.route(query, history=history or None)
            # 保存本轮对话到 SQLite（持久化）
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", result.answer)
            q.put(("done", _result_to_dict(result)))
        except Exception as e:
            q.put(("error", {"message": str(e)}))

    threading.Thread(target=worker, daemon=True).start()

    def event_gen():
        while True:
            etype, data = q.get()
            payload = json.dumps(data, ensure_ascii=False)
            yield f"event: {etype}\ndata: {payload}\n\n"
            if etype in ("done", "error"):
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，保证实时
        },
    )


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """获取会话历史（从 SQLite 读取）。"""
    messages = load_history(session_id)
    return JSONResponse({
        "session_id": session_id,
        "messages": messages,
    })


@app.post("/api/sessions/new")
def new_session():
    sid = uuid.uuid4().hex[:8]
    return {"session_id": sid}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
