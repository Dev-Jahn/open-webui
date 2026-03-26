#!/bin/bash
# Run Open WebUI with video input support via vllm-mlx
#
# Prerequisites:
#   1. vllm-mlx server running:
#      vllm-mlx serve <model-path> --served-model-name "Qwen3-VL-30B-A3B-Instruct" --port 8000
#   2. Docker image built:
#      docker compose build open-webui

docker rm -f open-webui 2>/dev/null

docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -v /tmp/openwebui-data:/app/backend/data \
  --add-host=host.docker.internal:host-gateway \
  -e WEBUI_SECRET_KEY=test-secret \
  -e ENABLE_OLLAMA_API=false \
  -e ENABLE_OPENAI_API=true \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -e VIDEO_SHARED_PATH=/tmp/openwebui-data/uploads \
  ghcr.io/open-webui/open-webui:main

echo "Open WebUI: http://localhost:3000"
