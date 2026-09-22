FROM python:3.11-slim

WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动 Web 服务（SSE 流式 + 前端）
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
