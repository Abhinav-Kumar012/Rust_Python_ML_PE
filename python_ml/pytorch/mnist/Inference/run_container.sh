#!/bin/bash

IMAGE="fastapi-ml-app"
LIB_MOUNT="/mnt/ml-libs"
PY_PATH="/external-libs/ml_env/lib/python3.12/site-packages"

docker run -d \
  -v ${LIB_MOUNT}:/external-libs \
  -e PYTHONPATH=${PY_PATH} \
  -p 8000:8000 \
  ${IMAGE}

echo "🚀 Container running on port 8000"
