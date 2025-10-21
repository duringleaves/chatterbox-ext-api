# syntax=docker/dockerfile:1

FROM node:alpine AS ui-builder
WORKDIR /uiapp
COPY ui/ ./ 
RUN npm install && npm run build

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libasound2 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libsndfile1 \
        libxext6 \
        libxrender1 \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python \
    && ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

ARG TORCH_INDEX_URL=""

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --extra-index-url "$TORCH_INDEX_URL" -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

COPY . .
COPY --from=ui-builder /uiapp/dist ./ui/dist

ENV CHATTERBOX_TTS_DEFAULTS_PATH=/app/config/tts_defaults.json

EXPOSE 8888
CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8888"]