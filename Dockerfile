# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install huggingface_hub[hf_xet]

COPY . /app

RUN mkdir -p /app/hf_models /app/outputs /app/logs /app/data/raw /app/data/interim /app/data/processed

# модель скачивается на этапе сборки образа
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python /app/scripts/download_model.py

ENV MODEL_NAME_OR_PATH=/app/hf_models/GigaChat3-10B-A1.8B-bf16
ENV TOKENIZER_NAME_OR_PATH=/app/hf_models/GigaChat3-10B-A1.8B-bf16
ENV MODEL_DEVICE=cpu
ENV MODEL_TORCH_DTYPE=float32
ENV HF_HUB_DISABLE_TELEMETRY=1

CMD ["python", "predict.py", "--input", "data/raw/knowledge_bench_public.csv", "--output", "outputs/submission.csv"]