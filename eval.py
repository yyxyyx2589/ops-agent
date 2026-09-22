"""第 4 周评估脚本：三基线对比 + 五项指标。

三基线：
  baseline_rag        纯 RAG（只用 rag_search 直答，不进 Agent 循环）
  baseline_rag_lora   RAG + LoRA（设计预留基线，基座为本地 Qwen2.5-1.5B+LoRA，当前未接入）
  baseline_agent      Agent（本项目：Router + Agent 循环 + 工具）

五项指标：
  1. 工具选择准确率   Agent 路径：模型选的工具和标注的期望工具是否一致
  2. 参数生成准确率   Agent 路径：工具参数 JSON 和标注参数是否一致
  3. 端到端解决率     三基线：答案是否命中标注的期望要点
  4. 平均轮次         Agent 路径：达到答案用了几轮工具调用
  5. 平均延迟         三基线：单次处理耗时（ms）

运行（需设 DEEPSEEK_API_KEY）：
  python eval.py --baseline agent --data data/eval_questions.jsonl
  python eval.py --baseline all   # 跑全部三基线，生成 reports/eval_report.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 让 data/ 和 reports/ 存在
DATA_FILE = Path(__file__).parent / "data" / "eval_questions.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"


def load_eval_data(path: Path) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_baseline_agent(question: str, router) -> dict:
    """Agent 基线：走完整 Router + Agent 循环。"""
    t0 = time.time()
    result = router.route(question)
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "answer": result.answer,
        "route": result.route,
        "tool_trace": result.tool_trace,
        "sim": result.sim,
        "kw_coverage": result.kw_coverage,
        "latency_ms": latency_ms,
        "turns": len(result.tool_trace),
    }


def run_baseline_rag(question: str, rag_tool, llm) -> dict:
    """纯 RAG 基线：只检索 + 直答，不进 Agent 循环（模拟原 RAG 项目）。"""
    t0 = time.time()
    raw = rag_tool.call({"query": question, "top_k": 1})
    rag = json.loads(raw)
    kb = rag["results"][0] if rag["results"] else None

    if kb:
        prompt = f"基于知识片段回答：\n{kb['text']}\n\n问题：{question}"
        resp = llm.chat([
            {"role": "system", "content": "你是运维助手。"},
            {"role": "user", "content": prompt},
        ])
        answer = resp.choices[0].message.content or ""
    else:
        answer = "未检索到相关知识。"

    return {
        "answer": answer,
        "sim": rag.get("best_sim", 0),
        "latency_ms": int((time.time() - t0) * 1000),
        "turns": 0,
        "kb_id": kb["id"] if kb else None,
    }


def run_baseline_rag_lora(question: str, rag_tool, lora_llm) -> dict:
    """RAG+LoRA 基线（设计预留，未接入真实 LoRA）：
    若接入本地 Qwen2.5-1.5B+LoRA，lora_llm 用 transformers 加载，接口与 LLMClient 一致。
    """
    # 占位：复用纯 RAG 逻辑，第 4 周接真实 LoRA 后替换 lora_llm
    return run_baseline_rag(question, rag_tool, lora_llm)


def score_answer(answer: str, expected_points: list) -> float:
    """答案命中期望要点的比例（0~1）。"""
    if not expected_points:
        return 1.0
    hit = sum(1 for p in expected_points if p in answer)
    return round(hit / len(expected_points), 3)


def score_tool_accuracy(trace: list, expected_tools: list) -> dict:
    """工具选择准确率 + 参数准确率。"""
    if not expected_tools:
        return {"tool_accuracy": 1.0, "param_accuracy": 1.0}
    actual_names = [s["name"] for s in trace]
    # 工具选择准确率：期望工具集有多少被实际调用
    expected_set = set(t["name"] for t in expected_tools)
    actual_set = set(actual_names)
    tool_acc = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0

    # 参数准确率：期望工具的参数是否匹配（简化：只检查 keyword/location 等关键字段）
    param_hits = 0
    param_total = 0
    for exp in expected_tools:
        for act in trace:
            if act["name"] == exp["name"]:
                param_total += 1
                exp_p = exp.get("params", {})
                act_p = act.get("params", {})
                if all(act_p.get(k) == v for k, v in exp_p.items()):
                    param_hits += 1
                break
        else:
            param_total += 1  # 期望调用但没调
    param_acc = param_hits / param_total if param_total else 1.0

    return {"tool_accuracy": round(tool_acc, 3),
            "param_accuracy": round(param_acc, 3)}


def evaluate(baseline: str, data: list, components: dict) -> list:
    results = []
    for item in data:
        q = item["question"]
        expected_points = item.get("expected_points", [])
        expected_tools = item.get("expected_tools", [])

        if baseline == "agent":
            out = run_baseline_agent(q, components["router"])
            tool_scores = score_tool_accuracy(out["tool_trace"], expected_tools)
        elif baseline == "rag":
            out = run_baseline_rag(q, components["rag_tool"], components["llm"])
            tool_scores = {"tool_accuracy": 0, "param_accuracy": 0}
        elif baseline == "rag_lora":
            out = run_baseline_rag_lora(q, components["rag_tool"], components["lora_llm"])
            tool_scores = {"tool_accuracy": 0, "param_accuracy": 0}
        else:
            raise ValueError(f"未知 baseline: {baseline}")

        answer_score = score_answer(out["answer"], expected_points)
        results.append({
            "qid": item["qid"],
            "question": q,
            "baseline": baseline,
            "answer": out["answer"],
            "answer_score": answer_score,
            "tool_accuracy": tool_scores["tool_accuracy"],
            "param_accuracy": tool_scores["param_accuracy"],
            "turns": out.get("turns", 0),
            "latency_ms": out["latency_ms"],
            "route": out.get("route", "—"),
            "sim": out.get("sim", 0),
        })
        print(f"  [{item['qid']}] {baseline:8} score={answer_score} "
              f"turns={out.get('turns', 0)} {out['latency_ms']}ms")
    return results


def aggregate(results: list) -> dict:
    """汇总指标。"""
    n = len(results) or 1
    return {
        "count": len(results),
        "answer_score_avg": round(sum(r["answer_score"] for r in results) / n, 3),
        "tool_accuracy_avg": round(sum(r["tool_accuracy"] for r in results) / n, 3),
        "param_accuracy_avg": round(sum(r["param_accuracy"] for r in results) / n, 3),
        "turns_avg": round(sum(r["turns"] for r in results) / n, 2),
        "latency_ms_avg": round(sum(r["latency_ms"] for r in results) / n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="agent",
                        choices=["agent", "rag", "rag_lora", "all"])
    parser.add_argument("--data", default=str(DATA_FILE))
    args = parser.parse_args()

    data = load_eval_data(Path(args.data))
    print(f"加载 {len(data)} 条测试问题")

    # 组装组件（需要 DEEPSEEK_API_KEY）
    from config import BASE_URL, MODEL, require_api_key
    from llm_client import LLMClient
    from agent import Agent
    from router import Router
    from tools import TOOL_REGISTRY
    from cli import SYSTEM_PROMPT  # cli.py 顶层无副作用，直接 import（比正则抽源码可靠）

    llm = LLMClient(api_key=require_api_key(), base_url=BASE_URL, model=MODEL)
    agent = Agent(llm=llm, tools=list(TOOL_REGISTRY.values()),
                 system_prompt=SYSTEM_PROMPT, max_turns=8)
    router = Router(rag_tool=TOOL_REGISTRY["rag_search"], agent=agent)
    components = {
        "llm": llm, "rag_tool": TOOL_REGISTRY["rag_search"],
        "router": router, "lora_llm": llm,  # 预留基线，未接入真实 LoRA（复用纯 RAG 链路）
    }

    baselines = ["agent", "rag"] if args.baseline == "all" else [args.baseline]
    if "rag_lora" in baselines:
        print("⚠️  rag_lora 为设计预留基线（未接入真实 LoRA，复用纯 RAG 链路），数据与 rag 相同。\n"
              "    若接入本地 Qwen2.5-1.5B+LoRA，替换 lora_llm 才是有效数据。")
    REPORTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for bl in baselines:
        print(f"\n=== 基线: {bl} ===")
        results = evaluate(bl, data, components)
        agg = aggregate(results)
        all_results[bl] = {"aggregate": agg, "details": results}
        print(f"  汇总: {agg}")

    out_file = REPORTS_DIR / "eval_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_file}")


if __name__ == "__main__":
    main()
