# ===================================
# Development-only Dockerfile with GGUF support
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

# Install build tools for llama-cpp-python compilation
RUN apt-get update && apt-get install -y \
    curl wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# Note: llama-cpp-python will compile from source (takes 5-10 minutes first time)
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/src /app/data /app/models /app/logs \
    /app/components /app/auth /app/services /app/database \
    /hf_cache /app/models/gguf

# IMPORTANT: Create __init__.py for packages
RUN touch /app/__init__.py \
    && touch /app/components/__init__.py \
    && touch /app/auth/__init__.py \
    && touch /app/services/__init__.py \
    && touch /app/database/__init__.py

# Expose Streamlit port
EXPOSE 8501