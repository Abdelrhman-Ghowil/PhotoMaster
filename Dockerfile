# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
ARG CPU_BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm
ARG GPU_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

FROM ${CPU_BASE_IMAGE} AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    OBJECT_ERASER_MODEL_PATH=/app/assets/big-lama.pt \
    REALESRGAN_MODEL_PATH=/app/assets/models/RealESRGAN_x4plus.pth \
    TORCH_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && mkdir -p /app/assets/models /app/.cache/huggingface /app/.cache/torch /app/.streamlit \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "Final.py"]

FROM ${GPU_BASE_IMAGE} AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    OBJECT_ERASER_MODEL_PATH=/app/assets/big-lama.pt \
    REALESRGAN_MODEL_PATH=/app/assets/models/RealESRGAN_x4plus.pth \
    TORCH_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 --shell /bin/bash appuser \
    && mkdir -p /app/assets/models /app/.cache/huggingface /app/.cache/torch /app/.streamlit \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "Final.py"]
