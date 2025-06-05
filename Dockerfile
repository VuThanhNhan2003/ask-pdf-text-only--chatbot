# Sử dụng Python 3.12.7 official image
FROM python:3.12.7-slim

# Thiết lập working directory
WORKDIR /app

# Cài đặt system dependencies (nếu cần)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --timeout 100

# Copy toàn bộ source code
COPY . .

# Expose port cho Streamlit (mặc định là 8501)
EXPOSE 8501

# Chạy Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]