# ✅ Use a supported Python
FROM python:3.12-slim

# System deps (OpenMP, GLib; curl for healthcheck; headless OpenCV avoids libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgthread-2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies (pin ORT + RapidOCR to known-good)
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache "onnxruntime==1.22.1" "rapidocr-onnxruntime==1.4.4" . && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.11.0.86

# App code
COPY . .

# Runtime env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UVICORN_WORKERS=1 \
    # Make HF/ORT caches writable on Render (no disk) or point to your mounted disk
    HF_HOME=/tmp/hf \
    HUGGINGFACE_HUB_CACHE=/tmp/hf \
    XDG_CACHE_HOME=/tmp/.cache \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-10000}/health || exit 1

CMD ["sh","-lc","exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers ${UVICORN_WORKERS:-1} --proxy-headers --forwarded-allow-ips='*'"]
