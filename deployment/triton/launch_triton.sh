#!/bin/bash

# Set the Triton Inference Server Docker image
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:25.06-py3"

# Model repository path
MODEL_REPO_PATH="$(pwd)/model_repository/"

# Set the ports for HTTP, gRPC, and metrics
HTTP_PORT=8000
GRPC_PORT=8001
METRICS_PORT=8002

# Container name
CONTAINER_NAME="triton_server"

# Check if the model repository exists
echo "Launching Triton Inference Server..."
echo "Using model repository: $MODEL_REPO_PATH"

# Docker command to run Triton Inference Server
docker run --rm -d \
  --gpus all \
  --name $CONTAINER_NAME \
  -p $HTTP_PORT:8000 \
  -p $GRPC_PORT:8001 \
  -p $METRICS_PORT:8002 \
  -v $MODEL_REPO_PATH:/models \
  $TRITON_IMAGE tritonserver \
  --model-repository=/models \
  --strict-model-config=false \
  --log-verbose=1
