"""ChromaDB 向量库初始化脚本。

把运维知识库从 Python 常量迁移到 ChromaDB + sentence-transformers embedding。

embedding 模型：paraphrase-multilingual-MiniLM-L12-v2
  - 多语言含中文，模型 ~100MB
  - CPU 推理可接受（单次 embedding ~50ms）
  - 384 维向量
  - 需设 HF_HUB_OFFLINE=1 跳过 huggingface SSL 检查（模型已缓存）

ChromaDB 持久化：./data/chroma/（本地文件，重启不丢）
距离度量：cosine（余弦距离），sim = 1 - distance

运行：python db/init_chroma.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path

# 15 条运维知识库（覆盖 nginx/mysql/oom/disk/redis/java/docker/load/网络/JVM/systemctl/cpu）
KB_DOCUMENTS = [
    {"id": "kb-001", "text": "nginx 502 Bad Gateway 错误通常由上游服务无响应或超时导致。排查步骤：查 nginx error.log 确认上游地址；检查上游服务是否存活；查上游服务资源是否耗尽（OOM/连接数打满）。",
     "keywords": ["nginx", "502", "bad gateway", "上游", "超时", "无响应"]},
    {"id": "kb-002", "text": "mysql Too many connections 错误表示连接数达到 max_connections 上限。处理方法：临时调高 max_connections；排查连接泄漏（连接未释放的慢查询）；应用侧引入连接池。",
     "keywords": ["mysql", "too many connections", "连接数", "max_connections"]},
    {"id": "kb-003", "text": "Linux OOM Killer 在内存耗尽时按 oom_score 杀进程释放内存。dmesg 可见 Out of memory Killed process。处理：扩内存或限制进程内存上限；调整 oom_score_adj 保护关键进程。",
     "keywords": ["oom", "out of memory", "killed", "内存", "oom_score"]},
    {"id": "kb-004", "text": "磁盘空间不足会导致日志写不进、服务异常。排查：df -h 查各挂载点使用率；du -sh 定位大目录；清理 docker images / 旧日志。",
     "keywords": ["磁盘", "disk", "no space", "空间", "df"]},
    {"id": "kb-005", "text": "nginx upstream timeout 表示反向代理转发请求到上游服务时超时。常见原因：上游服务 CPU/内存耗尽、上游进程崩溃、网络不通。解决：调大 proxy_read_timeout、配置 proxy_next_upstream 故障转移。",
     "keywords": ["nginx", "upstream", "timeout", "proxy", "上游"]},
    {"id": "kb-006", "text": "redis 连接重置 connection reset by peer 常见原因：redis 服务重启或 OOM 被杀；客户端连接超时；网络抖动。排查：查 redis 日志确认是否 OOM、查网络 retransmits。",
     "keywords": ["redis", "connection reset", "连接重置", "oom"]},
    {"id": "kb-007", "text": "Java 应用 CPU 占用过高排查：top -Hp pid 找高 CPU 线程；jstack pid 导出线程栈；查是否有死循环/频繁 GC。常见原因：死循环、大对象 GC、线程池满。",
     "keywords": ["java", "cpu", "jstack", "gc", "线程"]},
    {"id": "kb-008", "text": "Docker 磁盘占用过高排查：docker system df 查看占用；docker image prune 清理悬空镜像；docker volume prune 清理无用卷。/var/lib/docker 是默认存储路径。",
     "keywords": ["docker", "磁盘", "image", "volume", "prune"]},
    {"id": "kb-009", "text": "Linux 系统负载 load average 高排查：uptime 查 1/5/15 分钟负载；top 看哪个进程占 CPU；查是否有 D 状态进程。负载超过 CPU 核数说明过载。",
     "keywords": ["load", "负载", "cpu", "top", "uptime"]},
    {"id": "kb-010", "text": "mysql 慢查询排查：开启 slow_query_log；用 EXPLAIN 分析执行计划；加索引优化；查是否有全表扫描。慢查询会导致连接数堆积。",
     "keywords": ["mysql", "慢查询", "slow", "explain", "索引"]},
    {"id": "kb-011", "text": "网络连接 TIME_WAIT 过多排查：netstat 查 TIME_WAIT 数量；调整 tcp_tw_reuse；应用侧用长连接/连接池。TIME_WAIT 过多会耗尽端口导致新连接失败。",
     "keywords": ["time_wait", "netstat", "tcp", "连接", "端口"]},
    {"id": "kb-012", "text": "Java JVM 内存调优：-Xms -Xmx 设置堆大小；堆不宜超过物理内存 50%；用 jstat 监控 GC；OOM 时 dump 堆用 MAT 分析。8G 机器跑 -Xmx4g 偏激进易触发 OOM。",
     "keywords": ["java", "jvm", "xmx", "内存", "gc", "oom", "堆"]},
    {"id": "kb-013", "text": "nginx 配置优化：worker_processes 设为 CPU 核数；worker_connections 调大并发连接数；开启 gzip 压缩；配置 proxy_cache。proxy_read_timeout 默认 60s 可按需调大。",
     "keywords": ["nginx", "worker", "proxy", "gzip", "cache", "配置"]},
    {"id": "kb-014", "text": "系统服务状态排查：systemctl status service 查服务是否 active。inactive 表示服务停了，failed 表示启动失败。journalctl -u service 查服务日志。",
     "keywords": ["systemctl", "status", "service", "服务", "journalctl"]},
    {"id": "kb-015", "text": "CPU 使用率高的通用排查思路：top 看进程级 CPU；区分 user CPU（应用计算）和 system CPU（内核/IO）；高 system CPU 可能是大量 IO/上下文切换；load_avg 超过核数说明过载。",
     "keywords": ["cpu", "使用率", "top", "load", "system"]},
]


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    chroma_path = str(Path(__file__).parent.parent / "data" / "chroma")
    print(f"ChromaDB 路径: {chroma_path}")

    print("加载 embedding 模型（paraphrase-multilingual-MiniLM-L12-v2）...")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print(f"模型加载完成，向量维度: {embed_model.get_embedding_dimension()}")

    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection("ops_kb")
        print("已删除旧 collection")
    except Exception:
        pass

    # 创建 collection，用 cosine 距离（余弦相似度 = 1 - distance）
    collection = client.create_collection(
        name="ops_kb",
        metadata={"description": "运维知识库"},
        configuration={"hnsw:space": "cosine"})

    print(f"开始 embedding {len(KB_DOCUMENTS)} 条知识...")
    texts = [doc["text"] for doc in KB_DOCUMENTS]
    embeddings = embed_model.encode(texts).tolist()

    collection.add(
        ids=[doc["id"] for doc in KB_DOCUMENTS],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"keywords": ",".join(doc["keywords"])} for doc in KB_DOCUMENTS],
    )
    print(f"插入完成，collection 现有 {collection.count()} 条")

    # 验证查询（用 query_embeddings，不用 query_texts 避免 ChromaDB 内部重复 embedding）
    print("\n=== 验证查询（cosine 相似度）===")
    test_queries = [
        "nginx 502 怎么排查",
        "mysql 连接数打满",
        "内存不够被杀",
        "磁盘满了",
        "昨晚服务为什么变慢",
        "java 占内存太高",
    ]
    for q in test_queries:
        q_emb = embed_model.encode([q]).tolist()
        results = collection.query(
            query_embeddings=q_emb,
            n_results=1,
            include=["documents", "distances"])
        top_id = results["ids"][0][0]
        dist = results["distances"][0][0]
        sim = 1 - dist  # cosine distance → similarity
        print(f"  q={q:30} top={top_id} sim={sim:.3f}")

    print(f"\nChromaDB 初始化完成: {chroma_path}")


if __name__ == "__main__":
    main()
