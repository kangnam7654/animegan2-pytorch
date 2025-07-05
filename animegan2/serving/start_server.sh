#!/bin/bash

# AnimeGAN2 Preprocess Server 실행 스크립트

echo "Starting AnimeGAN2 Preprocess Server..."

# uv를 사용하여 의존성 설치
echo "Installing dependencies with uv..."
uv sync

# 서버 시작
echo "Starting server on http://0.0.0.0:8000"
uv run python run_server.py
