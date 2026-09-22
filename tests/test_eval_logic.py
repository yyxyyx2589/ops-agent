"""评估逻辑测试：验证评分函数（不调 LLM，零 API 消耗）。

覆盖：
  1. score_answer   —— 答案命中期望要点比例
  2. score_tool_accuracy —— 工具选择准确率 + 参数准确率
  3. aggregate      —— 汇总指标计算

运行：python tests/test_eval_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import score_answer, score_tool_accuracy, aggregate


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    return cond


def test_score_answer():
    points = ["max_connections", "连接池"]
    assert check("全命中得 1.0",
                 score_answer("调高 max_connections 并加连接池", points) == 1.0)
    assert check("命中一个得 0.5",
                 score_answer("调高 max_connections", points) == 0.5)
    assert check("全没命中得 0.0",
                 score_answer("重启服务", points) == 0.0)
    assert check("无期望要点默认 1.0",
                 score_answer("任意答案", []) == 1.0)


def test_score_tool_accuracy():
    expected = [
        {"name": "query_log", "params": {"keyword": "nginx"}},
        {"name": "get_monitor_metrics", "params": {"metric_type": "cpu"}},
    ]
    # 完全匹配
    trace_full = [
        {"name": "query_log", "params": {"keyword": "nginx"}, "result": "..."},
        {"name": "get_monitor_metrics", "params": {"metric_type": "cpu"}, "result": "..."},
    ]
    s = score_tool_accuracy(trace_full, expected)
    assert check("全匹配：工具准确率 1.0", s["tool_accuracy"] == 1.0)
    assert check("全匹配：参数准确率 1.0", s["param_accuracy"] == 1.0)

    # 只调了一个工具
    trace_half = [{"name": "query_log", "params": {"keyword": "nginx"}, "result": "..."}]
    s = score_tool_accuracy(trace_half, expected)
    assert check("调一半：工具准确率 0.5", s["tool_accuracy"] == 0.5)

    # 参数错了
    trace_bad_param = [
        {"name": "query_log", "params": {"keyword": "mysql"}, "result": "..."},
    ]
    s = score_tool_accuracy(trace_bad_param, expected)
    assert check("参数错：参数准确率 0.0", s["param_accuracy"] == 0.0)

    # 无期望工具
    s = score_tool_accuracy([], [])
    assert check("无期望工具默认全对", s["tool_accuracy"] == 1.0)


def test_aggregate():
    results = [
        {"answer_score": 1.0, "tool_accuracy": 1.0, "param_accuracy": 1.0,
         "turns": 2, "latency_ms": 100},
        {"answer_score": 0.5, "tool_accuracy": 0.5, "param_accuracy": 0.0,
         "turns": 4, "latency_ms": 300},
    ]
    agg = aggregate(results)
    assert check("答案分均值 0.75", agg["answer_score_avg"] == 0.75)
    assert check("工具准确率均值 0.75", agg["tool_accuracy_avg"] == 0.75)
    assert check("参数准确率均值 0.5", agg["param_accuracy_avg"] == 0.5)
    assert check("平均轮次 3.0", agg["turns_avg"] == 3.0)
    assert check("平均延迟 200ms", agg["latency_ms_avg"] == 200)
    assert check("计数 2", agg["count"] == 2)


if __name__ == "__main__":
    for fn in [test_score_answer, test_score_tool_accuracy, test_aggregate]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n全部通过：评估评分逻辑正确。")
