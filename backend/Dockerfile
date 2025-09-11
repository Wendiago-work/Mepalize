FROM python:3.13-slim

# Install system dependencies for OpenCV headless and ONNXRuntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgthread-2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -e . && \
    pip uninstall -y opencv-python -y && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.11.0.86

# ---- Models and Dependencies ----
# RapidOCR models will be downloaded on first use
# Chroma DB handles embeddings natively - no local models needed

# Copy application code
COPY . .

# Set environment variables
ENV PORT=10000
ENV UVICORN_WORKERS=2
ENV PYTHONPATH=/app

# Expose port (Render uses PORT env var, default 10000)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start the application
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 2"]
