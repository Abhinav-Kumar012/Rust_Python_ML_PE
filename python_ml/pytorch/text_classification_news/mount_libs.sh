#!/bin/bash

# ==========================================
# 1. Configuration - NFS Details
# ==========================================
# Replace these with your actual Library VM details
NFS_SERVER_IP="172.16.203.14" 
NFS_EXPORT_PATH="/home/iiitb/Documents/textClassificationVolume"
LOCAL_MOUNT_POINT="/mnt/text-libs"

# ==========================================
# 2. Mount Logic
# ==========================================

# 2.1 Check if mount point exists, create if not
if [ ! -d "$LOCAL_MOUNT_POINT" ]; then
    echo "Creating local mount point at $LOCAL_MOUNT_POINT..."
    sudo mkdir -p "$LOCAL_MOUNT_POINT"
else
    echo "Mount point $LOCAL_MOUNT_POINT already exists."
fi

# 2.2 Mount the NFS share
echo "Mounting NFS share from $NFS_SERVER_IP:$NFS_EXPORT_PATH..."
sudo mount -t nfs "$NFS_SERVER_IP:$NFS_EXPORT_PATH" "$LOCAL_MOUNT_POINT"

# 2.3 Verify success
if [ $? -eq 0 ]; then
    echo "✅ Successfully mounted NFS libs to $LOCAL_MOUNT_POINT"
    ls -l "$LOCAL_MOUNT_POINT"
else
    echo "❌ Failed to mount NFS share."
    exit 1
fi
