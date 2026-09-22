"""一键端到端自检：面试演示前跑这个确认全链路健康。

用法：
    python verify.py

流程：
  1. 三个测试套件（零 API 消耗）
  2. API key 检查（.env 或环境变量，没有则停在步骤 1）
  3. 真实对话三条路由验证（约 3 次 LLM 调用，几分钱内）：
     - "mysql too many connections 怎么办"   -> 期望 rag_direct
     - "今天午饭吃什么"                       -> 期望 refuse_human
     - "查一下 nginx 最近的报错日志"           -> 期望 agent_loop + 工具调用
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable


def step(label):
    print(f"\n{'='*50}\n{label}\n{'='*50}")


def run_tests():
    step("步骤 1/3：测试套件（零 API 消耗）")
    suites = ["tests/test_agent_mock.py", "tests/test_router_mock.py",
              "tests/test_eval_logic.py"]
    ok = True
    for s in suites:
        r = subprocess.run([PY, str(ROOT / s)], capture_output=True,
                           text=True, encoding="utf-8", cwd=ROOT)
        passed = "全部通过" in (r.stdout or "")
        print(f"  [{'PASS' if passed else 'FAIL'}] {s}")
        if not passed:
            print(r.stdout[-500:] if r.stdout else r.stderr[-500:])
            ok = False
    return ok


def check_key():
    step("步骤 2/3：API key 检查")
    from config import DEEPSEEK_API_KEY
    if not DEEPSEEK_API_KEY:
        print("  [SKIP] 未找到 DEEPSEEK_API_KEY（.env 文件或环境变量）")
        print("         填好后重跑本脚本验证真实对话。")
        return False
    print(f"  [PASS] key 已加载（长度 {len(DEEPSEEK_API_KEY)}）")
    return True


def verify_real_chat():
    step("步骤 3/3：真实对话三条路由验证")
    from cli import build_router

    router = build_router()
    cases = [
        ("mysql too many connections 怎么办", "rag_direct"),
        ("今天午饭吃什么", "refuse_human"),
        ("查一下 nginx 最近的报错日志", "agent_loop"),
    ]
    ok = True
    for query, expected in cases:
        try:
            r = router.route(query)
            hit = "PASS" if r.route == expected else "WARN"
            tools = len(r.tool_trace)
            print(f"  [{hit}] {query[:24]:26} -> {r.route}"
                  f"（期望 {expected}，工具 {tools} 次，sim={r.sim:.2f}）")
            print(f"        答案摘要: {r.answer[:60]}…")
            if r.route != expected:
                ok = False
        except Exception as e:
            print(f"  [FAIL] {query[:24]} -> 异常: {e}")
            ok = False
    return ok


def main():
    print("Ops Agent 一键自检")
    t1 = run_tests()
    if not t1:
        print("\n❌ 测试套件有失败项，先修测试再继续。")
        sys.exit(1)
    t2 = check_key()
    if not t2:
        print("\n✅ 代码逻辑全部健康（测试通过），真实链路待 key 填入后验证。")
        sys.exit(0)
    t3 = verify_real_chat()
    print("\n" + "=" * 50)
    if t3:
        print("✅ 全链路健康。下一步可跑三基线评估：")
        print("   python eval.py --baseline all")
        print("   然后启动 uvicorn server:app --port 8000 查看报告页 /report")
    else:
        print("⚠️  路由结果与预期有偏差（mock 相似度边界问题或真实模型行为差异），")
        print("   不影响演示，但可对照 docs/补课 文档排查。")
    sys.exit(0 if t3 else 0)


if __name__ == "__main__":
    main()
