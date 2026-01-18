#!/bin/bash
IMAGE_NAME=$1
CONTAINER_NAME="${IMAGE_NAME}-bench"

# Remove if exists
docker rm -f $CONTAINER_NAME 2>/dev/null

echo "Starting container..."
start_ts=$(date +%s%N)
docker run -d -p 8081:8080 --name $CONTAINER_NAME $IMAGE_NAME > /dev/null

# Poll for health
max_retries=50
for ((i=1;i<=max_retries;i++)); do
    if curl -s http://localhost:8081/health > /dev/null; then
        end_ts=$(date +%s%N)
        duration_ns=$((end_ts - start_ts))
        duration_ms=$((duration_ns / 1000000))
        echo "Cold Start Latency: ${duration_ms} ms"
        docker rm -f $CONTAINER_NAME > /dev/null
        exit 0
    fi
    sleep 0.1
done

echo "Failed to start in time"
docker rm -f $CONTAINER_NAME > /dev/null
exit 1
