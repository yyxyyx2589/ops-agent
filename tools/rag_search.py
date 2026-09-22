"""RAG 检索工具（ChromaDB + sentence-transformers 向量检索版）。

改造：从 Python 常量 + n-gram 相似度改成 ChromaDB 向量库 + embedding 余弦相似度。

双重身份（面试可讲）：
  1. 被 Router 直接调用 —— 拿 sim 分数做路由决策（高置信直答 / 中置信走 Agent / 低置信拒答）
  2. 被 Agent 调用 —— 作为 Agent 工具池的一员，给 LLM 看检索到的知识片段

embedding 模型：paraphrase-multilingual-MiniLM-L12-v2（384 维，多语言含中文）
向量库：ChromaDB（持久化到 data/chroma/，cosine 距离）
相似度：手动算余弦相似度（ChromaDB distance 是 squared L2，不直接用）
关键词覆盖率：从 metadata.keywords 计算

初始化：python db/init_chroma.py（建库 + 导入 15 条 KB）
"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import register_tool
from tools.base import BaseTool

# 模块级单例（避免每次调用都重新加载模型）
_embed_model = None
_chroma_collection = None

CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma")


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = client.get_collection("ops_kb")
    return _chroma_collection


def _cosine_similarity(vec_a, vec_b) -> float:
    """手动计算余弦相似度（ChromaDB distance 是 squared L2，不直接用）。

    面试讲点：ChromaDB 的 hnsw:space=cosine 返回的是归一化后的 squared L2，
    而不是标准 cosine distance。为了拿到准确的余弦相似度，自己用 numpy 算。
    """
    import numpy as np
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _kw_coverage(query: str, keywords_str: str) -> float:
    """query 命中 KB 关键词的比例。keywords_str 是逗号分隔的字符串。"""
    q_lower = query.lower()
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    if not keywords:
        return 0.0
    hit = sum(1 for kw in keywords if kw.lower() in q_lower)
    return hit / len(keywords)


@register_tool
class RagSearchTool(BaseTool):
    name = "rag_search"
    description = ("检索运维知识库，返回与问题最相关的知识片段。"
                   "当用户询问已知故障模式、运维知识、操作步骤时调用。"
                   "返回结果包含相似度分数 sim 和关键词覆盖率 kw_coverage，"
                   "供路由层判断置信度。"
                   "基于向量检索（sentence-transformers embedding + ChromaDB），"
                   "语义相近的问题能命中即使没有字面重叠。")
    parameters = [
        {"name": "query", "type": "string",
         "description": "要检索的问题或关键词", "required": True},
        {"name": "top_k", "type": "integer",
         "description": "返回前 K 条结果，默认 1", "required": False},
    ]

    def call(self, params: dict) -> str:
        query = params.get("query", "")
        top_k = params.get("top_k", 1)

        if not query:
            return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

        try:
            model = _get_embed_model()
            collection = _get_collection()
        except Exception as e:
            return json.dumps(
                {"error": f"向量库未初始化: {e}",
                 "hint": "运行 python db/init_chroma.py 初始化"},
                ensure_ascii=False)

        # 1. query embedding
        q_emb = model.encode([query]).tolist()

        # 2. ChromaDB 检索（取 top_k*3 条候选，再用余弦相似度精排）
        candidates = collection.query(
            query_embeddings=q_emb,
            n_results=min(max(top_k * 3, 5), 15),
            include=["documents", "metadatas"])

        # 3. 用手动余弦相似度精排（ChromaDB distance 不直接用）
        import numpy as np
        q_vec = np.array(q_emb[0])
        scored = []
        for i, doc_id in enumerate(candidates["ids"][0]):
            doc_text = candidates["documents"][0][i]
            keywords_str = candidates["metadatas"][0][i].get("keywords", "")

            # 取这条 KB 的 embedding
            kb_data = collection.get(ids=[doc_id], include=["embeddings"])
            kb_vec = np.array(kb_data["embeddings"][0])

            sim = _cosine_similarity(q_vec, kb_vec)
            kw_cov = _kw_coverage(query, keywords_str)
            score = 0.7 * sim + 0.3 * kw_cov  # 综合评分（与原 RAG 项目加权一致）

            scored.append({
                "id": doc_id,
                "text": doc_text,
                "sim": round(sim, 3),
                "kw_coverage": round(kw_cov, 3),
                "score": round(score, 3),
                "keywords": keywords_str,
            })

        # 按综合分排序
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:max(top_k, 1)]

        if not top:
            return json.dumps(
                {"query": query, "best_sim": 0.0, "best_kw_coverage": 0.0,
                 "results": []},
                ensure_ascii=False)

        best = top[0]
        return json.dumps(
            {"query": query,
             "best_sim": best["sim"],
             "best_kw_coverage": best["kw_coverage"],
             "results": top},
            ensure_ascii=False)
