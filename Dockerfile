# Use a base Python image
FROM python:3.12-slim

# Set app directory
WORKDIR /app
COPY . /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies, including OpenGL libraries for libGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --upgrade pip \
    && pip install --no-cache-dir uv

# Run uv
RUN uv sync

# Default command
CMD ["uv", "run", "preprocess.py"]
