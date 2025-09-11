# ✅ Use a supported Python
FROM python:3.12-slim

# System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache .

# App code
COPY . .

# Runtime env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UVICORN_WORKERS=1

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-10000}/health || exit 1

CMD ["sh","-lc","exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers ${UVICORN_WORKERS:-1} --proxy-headers --forwarded-allow-ips='*'"]
