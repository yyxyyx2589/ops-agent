# Ops Agent — 企业 IT 运维智能体

基于 Function Calling 的运维 Agent，把"RAG 检索置信度"升级为"决策路由信号"，
实现 **同一业务从 RAG → Agent 的演进**（LoRA 为独立微调验证线）。

## 架构

```
用户问题
    │
    ▼
┌──────────┐  RAG 检索拿 sim 分数
│  Router  │─────────────────────────────┐
└──────────┘                             │
    │ sim≥0.70 & kw≥0.30                 │ 0.45≤sim<0.70          │ sim<0.45
    ▼ RAG 直答                           ▼ Agent 循环              ▼ 拒答转人工
┌──────────┐                        ┌──────────┐               ┌──────────┐
│ LLM 组织 │                        │ ReAct 循环│               │ 兜底文案 │
│ 语言回答 │                        │ 工具调用  │               │ 转人工   │
└──────────┘                        └──────────┘               └──────────┘
                                         │
                                    ┌────┴────┬────────┬────────┐
                                    ▼         ▼        ▼        ▼
                              query_log  monitor  diagnose  rag_search
```

## 核心模块

| 文件 | 职责 |
|---|---|
| `agent.py` | ReAct 主循环（Function Calling 版）+ 三层防御链 + event_hook（**默认引擎**） |
| `agent_langgraph.py` | 同一主循环的 LangGraph 图编排版（同签名 / 同事件协议，可互换） |
| `router.py` | 决策路由：RAG直答 / Agent循环 / 拒答转人工 |
| `tools/` | 5 个工具：query_log / get_monitor_metrics / run_diagnostic_command / rag_search / get_current_weather |
| `server.py` | FastAPI + SSE 流式后端 |
| `web/index.html` | 前端单页：流式对话窗 + Agent 时间线 + 路由徽章 + 溯源面板 |
| `cli.py` | 命令行版（带路由徽章 + tool_trace 打印） |
| `eval.py` | 评估（纯 RAG / Agent；rag_lora 为预留基线） |
| `compare_engines.py` | 演示：同一问题用两种编排引擎并排跑 |
| `docs/LangGraph-复刻对照.md` | 两种编排的取舍对照 + 等价性证明 + 面试 Q&A |

## 快速开始

> **首次克隆后必须先初始化数据**：`db/ops.db`（真实日志与监控指标）和 `data/chroma/`（向量库）
> 体积较大、可由脚本确定性重建，因此未纳入版本库。跳过这步会因为查不到数据而失败。

```bash
# 1. 环境
conda create -n ops-agent python=3.11 -y && conda activate ops-agent
pip install -r requirements.txt

# 2. API key（DeepSeek）——二选一
#    方式 A：写进项目根目录 .env（推荐，换终端不用重设）
cp .env.example .env && vim .env        # 填入 DEEPSEEK_API_KEY=sk-xxx
#    方式 B：临时环境变量
export DEEPSEEK_API_KEY=sk-你的key

# 3. 初始化数据（首次克隆必做，零 API 消耗）
python db/init_db.py       # 生成 SQLite：nginx/mysql/OOM/磁盘告警日志 + web-01 监控指标
python db/init_chroma.py   # 构建 ChromaDB 向量库：15 条运维知识 + 余弦相似度自检

# 4. 自检（跑测试 + 三条路由真实验证，需要 API key）
python verify.py

# 命令行版
python cli.py

# 切换到 LangGraph 编排引擎（默认 react = 手写主循环）
AGENT_ENGINE=langgraph python cli.py
AGENT_ENGINE=langgraph uvicorn server:app --reload --port 8000

# 两种引擎并排对比（真实 API，演示用）
python compare_engines.py "web-01 这台机器为什么变慢？"

# Web 版（SSE 流式 + 前端）
uvicorn server:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

初始化脚本输出示例（第 3 步）：

```
数据库初始化完成: .../db/ops.db
ChromaDB 初始化完成: .../data/chroma
  q=mysql 连接数打满                 top=kb-002 sim=0.834
```

## 工具说明

| 工具 | 触发场景 | 数据来源（真实） |
|---|---|---|
| `query_log` | 故障排查、报错分析 | nginx 502 / mysql 连接耗尽 / OOM / 磁盘告警，串成故障链（SQLite 真实日志 552 条） |
| `get_monitor_metrics` | 资源使用、负载查询 | web-01 CPU 43.4% + 内存 5.1/8GB + 磁盘 80.4%（SQLite 真实指标 9075 条） |
| `run_diagnostic_command` | 看实时系统状态 | 白名单沙箱：ps/netstat/df/free/systemctl status（Docker 容器真实执行） |
| `rag_search` | 知识检索 + 路由信号源 | 15 条 KB + ChromaDB 向量检索 + sentence-transformers 向量化 + 余弦相似度 |

## 路由阈值

| 阈值 | 值 | 含义 |
|---|---|---|
| SIM_HIGH | 0.70 | ≥ 且 kw≥0.30 → RAG 直答 |
| SIM_LOW | 0.45 | < → 拒答转人工；[SIM_LOW, SIM_HIGH) → Agent 循环 |
| KW_COVERAGE_LOW | 0.30 | 关键词覆盖率下限 |

## 测试

> 需先完成「快速开始」第 3 步的数据初始化，否则 `query_log` 查不到日志、mock 测试会断言失败。

```bash
# 主循环逻辑测试（mock LLM，零 API 消耗）
python tests/test_agent_mock.py

# 手写版 vs LangGraph 版等价性对照（10 场景，零 API 消耗）
python tests/test_agent_langgraph_parity.py

# 决策路由测试
python tests/test_router_mock.py

# 评估（需 API key）：--baseline all 跑 agent+rag 两基线
python eval.py --baseline all
# rag_lora 为预留基线，需显式指定：python eval.py --baseline rag_lora
```

## 演示脚本（面试 3 分钟）

1. **知识直答**：问 "mysql too many connections 怎么办" → 徽章 [RAG 直答]，溯源面板亮出片段
2. **Agent 循环**：问 "昨晚 web-01 为什么变慢" → 徽章 [Agent 循环]，时间线展开查日志→查监控
3. **拒答兜底**：问 "今天午饭吃什么" → 徽章 [拒答转人工]
4. **评估收尾**：切 eval_report.json → 三基线对比图

## 技术选型

- LLM：DeepSeek API（OpenAI 兼容，CPU-only 友好）
- 编排：**双引擎**——`agent.py` 手写 ReAct 循环（默认，零框架依赖）+ `agent_langgraph.py` LangGraph 图编排，
  两者同签名、同事件协议，用 10 场景等价性对照测试保证行为不漂移。详见 `docs/LangGraph-复刻对照.md`
- 前端：原生 HTML+JS（非 Gradio），支持自定义时间线/徽章/溯源
- 流式：SSE 事件流（route/tool_call/tool_result/answer_chunk/done）
