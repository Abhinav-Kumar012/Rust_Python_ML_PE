#!/bin/bash

# ==========================================
# 1. Configuration
# ==========================================
IMAGE_NAME="regression_image"
CONTAINER_NAME="regression_container"
HOST_PORT=8000
CONTAINER_PORT=8000

# Paths
NFS_MOUNT_POINT="/mnt/regression-libs"
CONTAINER_LIB_MOUNT="/external-libs"

# FIXED: must match Dockerfile exactly
PYTHON_LIB_PATH="$CONTAINER_LIB_MOUNT/regression_env/lib/python3.12/site-packages"

# ==========================================
# 2. Pre-flight Checks
# ==========================================

if [ ! -d "$NFS_MOUNT_POINT" ] || [ -z "$(ls -A $NFS_MOUNT_POINT)" ]; then
   echo "⚠️  Warning: $NFS_MOUNT_POINT appears empty or unmounted."
   echo "Did you run ./mount_libs.sh?"
   read -p "Press Enter to continue anyway (or Ctrl+C to abort)..."
fi

# ==========================================
# 3. Cleanup old container
# ==========================================

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing container..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

# ==========================================
# 4. Run container (GPU enabled)
# ==========================================

echo "🚀 Starting inference container..."

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -v "$NFS_MOUNT_POINT:$CONTAINER_LIB_MOUNT:ro" \
  -e PYTHONPATH="$PYTHON_LIB_PATH" \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  "$IMAGE_NAME"

# ==========================================
# 5. Status
# ==========================================

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully!"
    echo "📍 API available at: http://localhost:$HOST_PORT"
    docker logs "$CONTAINER_NAME"
else
    echo "❌ Failed to start container."
fi