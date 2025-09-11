FROM python:3.13-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgthread-2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (production install for better caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache . && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.11.0.86

# App code after deps for better caching
COPY . .

# Fast-boot environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UVICORN_WORKERS=1

# EXPOSE is optional for Render; if you keep it, use a literal (no env expansion)
EXPOSE 10000

# Healthcheck (only if your app serves /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-10000}/health || exit 1

# Start command: bind to 0.0.0.0 and $PORT with proxy headers for Render
CMD ["sh", "-lc", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers ${UVICORN_WORKERS:-1} --proxy-headers --forwarded-allow-ips='*'"]
