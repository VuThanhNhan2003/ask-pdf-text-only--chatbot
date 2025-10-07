# ===================================
# Development-only Dockerfile
# ===================================
FROM python:3.11-slim

# Disable pycache + better logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache

WORKDIR /app

# Cài công cụ tối thiểu
RUN apt-get update && apt-get install -y \
    curl wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Tạo sẵn các thư mục
RUN mkdir -p /app/src /app/data /app/models /app/logs \
    /app/components /app/auth /app/services /app/database /hf_cache

# IMPORTANT: Tạo __init__.py cho các packages
RUN touch /app/__init__.py \
    && touch /app/components/__init__.py \
    && touch /app/auth/__init__.py \
    && touch /app/services/__init__.py \
    && touch /app/database/__init__.py

# Expose Streamlit port
EXPOSE 8501