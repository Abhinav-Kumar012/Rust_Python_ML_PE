#!/bin/bash

LIB_VM_IP="172.16.203.14"
LIB_PATH="/home/iiitb/Documents/volume"
LOCAL_MOUNT="/mnt/ml-libs"

echo "Installing NFS client..."
sudo apt-get update -y
sudo apt-get install -y nfs-common

echo "Creating mount directory..."
sudo mkdir -p $LOCAL_MOUNT

echo "Mounting ML libraries..."
sudo mount ${LIB_VM_IP}:${LIB_PATH} $LOCAL_MOUNT

echo "Libraries mounted at $LOCAL_MOUNT"
