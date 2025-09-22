# Dev-only image: cài dependencies từ requirements.txt, KHÔNG copy code/models vào image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Cài công cụ tối thiểu cho healthcheck/diagnostics
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Chỉ copy file dependencies để tận dụng cache khi code thay đổi
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Tạo sẵn các thư mục hay dùng
RUN mkdir -p /app/data /app/models

# Streamlit mặc định chạy port 8501
EXPOSE 8501

# Không CMD ở đây; docker-compose sẽ quy định command