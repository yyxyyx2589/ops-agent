"""决策路由模块：基于 RAG 检索置信度，把问题分流到三条路径。

三层路由（面试叙事核心："同一业务从 RAG 演进到 Agent"）：
  rag_direct  : sim >= SIM_HIGH 且 kw_coverage >= KW_COVERAGE_LOW
                命中知识库高置信片段，模型只需组织语言，不进 Agent 循环
  agent_loop  : SIM_LOW <= sim < SIM_HIGH
                中置信区间，信息不足，启动 Agent 循环调工具补充
  refuse_human: sim < SIM_LOW
                超出知识库范围，拒答转人工，避免幻觉

设计要点：
  - Router 作为编排层，Agent 作为执行层之一（依赖反转）
  - Router 直接调 RAG 工具拿 sim 分数（不经 LLM），决策后把 RAG 结果
    作为 observation 塞进 Agent 的初始上下文，避免重复检索
  - event_hook 透传给 Agent，SSE 流式时实时推送 route/tool_call/tool_result
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import SIM_HIGH, SIM_LOW, KW_COVERAGE_LOW

if TYPE_CHECKING:
    from agent import Agent
    from tools.base import BaseTool

# 意图检测关键词（面试素材：RAG 相似度对动作型查询打分偏低，KB 是知识型内容
# 没有"查/执行"这类动作描述，低置信动作型查询会被误判到 refuse_human，
# 用动作词+主题词双命中兜底救回 agent_loop）
_ACTION_VERBS = {"查", "查询", "查一下", "查找", "执行", "运行",
                 "诊断", "排查", "检查", "查看", "看看", "看一下",
                 "调出", "拉取", "怎么办", "怎么处理", "如何解决",
                 "处理", "解决", "是什么", "为什么", "怎么样", "多少",
                 "情况", "占用", "使用率", "状态", "异常吗", "够用吗",
                 "重置", "被杀", "变慢"}
_TOOL_THEMES = {"日志", "log", "监控", "monitor", "命令", "diagnose",
                "诊断", "状态", "指标", "报错", "错误", "异常", "故障",
                "磁盘", "内存", "cpu", "进程", "网络", "连接",
                "mysql", "redis", "java", "nginx", "docker", "服务", "web"}


@dataclass
class RouteResult:
    route: str            # "rag_direct" / "agent_loop" / "refuse_human"
    answer: str           # 最终答案
    sim: float            # RAG top1 相似度（路由依据，前端徽章展示）
    kw_coverage: float    # 关键词覆盖率（路由依据）
    tool_trace: list = field(default_factory=list)  # Agent 路径的工具调用轨迹
    kb_hits: list = field(default_factory=list)      # 命中的知识片段（溯源面板）


class Router:
    def __init__(self, rag_tool: BaseTool, agent: Agent,
                 sim_high: float = SIM_HIGH,
                 sim_low: float = SIM_LOW,
                 kw_coverage_low: float = KW_COVERAGE_LOW,
                 event_hook=None):
        self.rag_tool = rag_tool
        self.agent = agent
        self.sim_high = sim_high
        self.sim_low = sim_low
        self.kw_coverage_low = kw_coverage_low
        self.event_hook = event_hook

    def _emit(self, etype: str, data: dict):
        if self.event_hook:
            self.event_hook(etype, data)

    def _detect_intent(self, query: str) -> bool:
        """意图检测：query 是否同时包含动作词 + 工具主题词。

        设计动机：RAG 相似度对"查 nginx 日志"这类动作型查询打分偏低
        （KB 是知识型内容，没有动作描述），低置信会被误判到 refuse_human。
        用"动作词 + 主题词"双命中兜底，把本要拒答的动作型查询救回 agent_loop。

        反例防护：
          - "nginx 502 是什么"：只有主题词无动作词 → 不算意图，走 sim 决策
          - "查一下天气"：只有动作词无主题词 → 不算意图，走 sim 决策
          - "mysql too many connections 怎么办"：sim≥0.70 走 rag_direct，
            根本到不了意图检测层（意图只在 sim<sim_low 时才触发）
        """
        q = query.lower()
        has_action = any(v in q for v in _ACTION_VERBS)
        has_theme = any(t in q for t in _TOOL_THEMES)
        return has_action and has_theme

    def route(self, user_message: str, history: list = None) -> RouteResult:
        # ① 先做一次 RAG 检索，拿置信度信号
        rag_raw = self.rag_tool.call({"query": user_message, "top_k": 3})
        rag = json.loads(rag_raw)
        sim = rag.get("best_sim", 0.0)
        kw_cov = rag.get("best_kw_coverage", 0.0)
        kb_hits = rag.get("results", [])

        # 透传 event_hook 给 Agent（SSE 流式用）
        self.agent.event_hook = self.event_hook

        # ② 阈值决策
        if sim >= self.sim_high and kw_cov >= self.kw_coverage_low:
            self._emit("route", {"route": "rag_direct",
                                 "sim": sim, "kw_coverage": kw_cov})
            return self._route_rag_direct(user_message, sim, kw_cov, kb_hits, history)
        if sim >= self.sim_low:
            self._emit("route", {"route": "agent_loop",
                                 "sim": sim, "kw_coverage": kw_cov})
            return self._route_agent_loop(user_message, sim, kw_cov, kb_hits, history)
        # 低置信：先过意图检测，动作型查询救回 agent_loop（避免误拒答）
        if self._detect_intent(user_message):
            self._emit("route", {"route": "agent_loop",
                                 "sim": sim, "kw_coverage": kw_cov,
                                 "reason": "intent_override"})
            return self._route_agent_loop(user_message, sim, kw_cov, kb_hits, history)
        self._emit("route", {"route": "refuse_human",
                             "sim": sim, "kw_coverage": kw_cov})
        return self._route_refuse_human(user_message, sim, kw_cov, kb_hits)

    # ---------- 三条路径 ----------

    def _route_rag_direct(self, query, sim, kw_cov, kb_hits, history=None) -> RouteResult:
        """高置信：把命中片段交给 LLM 组织成自然语言答案，不进 Agent 循环。"""
        context = "\n".join(f"[{kb['id']}] {kb['text']}" for kb in kb_hits[:1])
        prompt = (f"基于以下知识库片段，回答用户问题。"
                  f"答案里要引用片段编号。\n\n知识片段：\n{context}\n\n问题：{query}")

        messages = [
            {"role": "system", "content": "你是运维知识助手，基于知识库片段回答。"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        resp = self.agent.llm.chat(messages)
        answer = resp.choices[0].message.content or ""
        self._emit("answer_chunk", {"content": answer})
        self._emit("answer_done", {})

        return RouteResult(route="rag_direct", answer=answer,
                          sim=sim, kw_coverage=kw_cov, kb_hits=kb_hits[:1])

    def _route_agent_loop(self, query, sim, kw_cov, kb_hits, history=None) -> RouteResult:
        """中置信：把 RAG 命中片段作为先验 observation 注入 Agent 初始上下文。

        门控：只有 sim >= sim_low 才注入先验。sim 低于阈值的命中实际不相关，
        注入只会误导 LLM（如"web-01 为什么变慢"sim=0.0，却注入了 mysql 的 KB，
        导致 LLM 开头复述 mysql 处理方法而非直接调工具诊断）。
        sim < sim_low 时（意图检测救回的查询）纯走 Agent 调工具，不注入先验。
        """
        prior = ""
        if kb_hits and sim >= self.sim_low:
            prior = f"\n\n[知识库先验，相似度 {sim:.2f}] {kb_hits[0]['text']}"
        enriched_query = query + prior

        result = self.agent.run(enriched_query, history=history)
        return RouteResult(route="agent_loop", answer=result.answer,
                          sim=sim, kw_coverage=kw_cov,
                          tool_trace=result.tool_trace,
                          kb_hits=kb_hits[:1] if sim >= self.sim_low else [])

    def _route_refuse_human(self, query, sim, kw_cov, kb_hits) -> RouteResult:
        """低置信：超出知识库范围，拒答并说明转人工，避免模型幻觉。"""
        answer = (f"该问题超出当前知识库覆盖范围（最高相似度 {sim:.2f}，"
                  f"低于阈值 {self.sim_low}），已转人工处理。"
                  f"建议补充知识库或联系值班工程师。")
        self._emit("answer_chunk", {"content": answer})
        self._emit("answer_done", {})
        return RouteResult(route="refuse_human", answer=answer,
                          sim=sim, kw_coverage=kw_cov, kb_hits=[])
