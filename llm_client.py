from openai import OpenAI


class LLMClient:
    """OpenAI 兼容 LLM 客户端。DeepSeek/Qwen 本地服务都是同一套接口。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client=OpenAI(base_url=base_url, api_key=api_key)
        self.model=model

    def chat(self, messages: list, tools: list | None = None,
             stream: bool = False) -> object:
        """调用 chat.completions.create。
        - messages: OpenAI 消息格式
        - tools:    OpenAI 工具 schema 列表（本周暂时不用，第 2 周接上）
        - stream:   本周先 False，第 3 周再做流式
        返回原始 response 对象，解析交给 Agent 层。
        """
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=stream
        )