# AnimeGAN2 Preprocess API Dockerfile

FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 OpenCV 의존성 설치
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# uv를 사용하여 의존성 설치
COPY animegan2 ./
RUN uv init --bare 
RUN uv venv && . .venv/bin/activate
RUN uv add -U --no-cache-dir \
    "fastapi[standard]" \
    "uvicorn" \
    "opencv-python-headless" \
    "opencv-python" \
    "numpy"