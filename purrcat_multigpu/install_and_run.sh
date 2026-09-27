#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
DIR="/root/purrcat-multigpu"
mkdir -p "$DIR"
cd "$DIR"

apt-get update -y
apt-get install -y curl ca-certificates build-essential

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Use a Vast/NVIDIA image with CUDA 12.8+ installed."
  exit 1
fi

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo

read -rsp "Enter PurrCat private key: " PURRCAT_PRIVATE_KEY
echo
export PURRCAT_PRIVATE_KEY

cat > package.json <<'PKG'
{"private":true,"type":"module","dependencies":{"ethers":"^6.15.0"}}
PKG
npm install --silent --no-audit --no-fund

curl -fsSL https://purrcat.xyz/deployment.json -o deployment.json
curl -fsSL https://purrcat.xyz/miner/keccak_core.js -o keccak_core.js

if [ ! -f ./purrcat_cuda_worker.cu ] || [ ! -f ./controller.mjs ]; then
  echo "ERROR: miner source files are missing."
  exit 1
fi

echo "Compiling multi-GPU CUDA worker..."
nvcc -O3 --std=c++17 -arch=sm_120 purrcat_cuda_worker.cu -o purrcat_cuda_worker
chmod +x purrcat_cuda_worker

echo "Starting one CUDA worker per NVIDIA GPU."
echo "All workers share the same PurrCat job and use disjoint nonce partitions."
node controller.mjs
