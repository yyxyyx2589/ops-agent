FROM alpine:3.19

# 装运维诊断常用工具（用更基础的包，避免网络问题）
RUN apk add --no-cache procps util-linux busybox-extras && \
    mkdir -p /var/log

# 写几条真实格式的日志（让 grep/cat 有东西查）
RUN echo "2026-09-04 23:12:01 [ERROR] nginx: upstream timed out from 10.0.2.15:8080" >> /var/log/nginx_error.log && \
    echo "2026-09-04 23:20:11 [ERROR] kernel: Out of memory: Killed process 3321 (java)" >> /var/log/messages && \
    echo "2026-09-04 23:31:05 [WARN] disk: /var usage 92%" >> /var/log/messages

CMD ["sleep", "infinity"]
