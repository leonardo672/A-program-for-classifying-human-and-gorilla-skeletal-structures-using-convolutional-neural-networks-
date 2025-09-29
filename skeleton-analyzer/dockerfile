# -----------------------------
# Dockerfile for skeleton-analyzer (headless)
# -----------------------------

FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# -----------------------------
# Install system dependencies (minimal)
# -----------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    unzip \
    unixodbc \
    unixodbc-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements and install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# -----------------------------
# Copy application code
# -----------------------------
COPY . .

# -----------------------------
# Create data directory (if not using mounted volume)
# -----------------------------
RUN mkdir -p /app/Data

# -----------------------------
# Run headless by default
# -----------------------------
CMD ["python", "main.py", "--mode", "headless"]
