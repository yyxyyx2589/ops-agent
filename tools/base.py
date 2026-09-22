from abc import ABC, abstractmethod

class BaseTool(ABC):
    """所有工具的基类。description 和 parameters 会被注入给 LLM。"""

    name: str            # 工具名，如 "query_log"
    description: str     # 给 LLM 看的功能描述——写得好坏直接决定调用准确率
    parameters: list     # JSON Schema 参数声明，如
                         # [{"name": "keyword", "type": "string",
                         #   "description": "日志关键词", "required": True}]

    @abstractmethod
    def call(self, params: dict) -> str:
        """执行工具。入参是 LLM 生成的参数 dict，返回值必须是 str（序列化后回填给 LLM）。"""

def to_openai_schema(tool: BaseTool) -> dict:
    """把 BaseTool 转成 OpenAI tools 参数格式：
    {"type": "function", "function": {"name", "description", "parameters"}}
    注意 OpenAI 的 parameters 是 JSON Schema 对象（properties/required），要做一次格式转换。"""
    properties={}
    for p in tool.parameters:
        properties[p["name"]]={"type": p["type"], "description": p["description"]}
    required = [p["name"] for p in tool.parameters if p.get("required", False)]

    return {"type": "function", "function": {"name": tool.name,
                                             "description": tool.description,
                                             "parameters": {"type": "object",
                                                            "properties": properties,
                                                            "required": required}}}