#!/bin/bash

# ==========================================
# 1. Configuration
# ==========================================
IMAGE_NAME="text_classification_image"
CONTAINER_NAME="text_classification_container"
HOST_PORT=8000
CONTAINER_PORT=8000

# Paths
NFS_MOUNT_POINT="/mnt/text-libs"
CONTAINER_LIB_MOUNT="/external-libs"
# The exact path to site-packages within the mount
PYTHON_LIB_PATH="$CONTAINER_LIB_MOUNT/text_env/lib/python3.12/site-packages"

# ==========================================
# 2. Pre-flight Checks
# ==========================================

# Check if NFS libs are mounted
if [ ! -d "$NFS_MOUNT_POINT" ] || [ -z "$(ls -A $NFS_MOUNT_POINT)" ]; then
   echo "⚠️  Warning: $NFS_MOUNT_POINT appears empty or unmounted."
   echo "Did you run ./mount_libs.sh?"
   read -p "Press Enter to continue anyway (or Ctrl+C to abort)..."
fi

# ==========================================
# 3. Docker Execution
# ==========================================

# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping removing existing container '$CONTAINER_NAME'..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

echo "🚀 Starting inference container..."

# Run Command Explanation:
# --gpus all        : Enable GPU access
# -v ...            : Mount NFS libs and Model volume
# -e PYTHONPATH...  : Point Python to the external libs
# -p ...            : Expose API port

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -v "$NFS_MOUNT_POINT:$CONTAINER_LIB_MOUNT" \
  -v text_model_vol:/models \
  -e PYTHONPATH="$PYTHON_LIB_PATH" \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  "$IMAGE_NAME"

# ==========================================
# 4. Status
# ==========================================
if [ $? -eq 0 ]; then
    echo "✅ Container started successfully!"
    echo "📍 API available at: http://localhost:$HOST_PORT"
    docker logs "$CONTAINER_NAME"
else
    echo "❌ Failed to start container."
fi
