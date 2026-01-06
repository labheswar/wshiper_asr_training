#!/bin/bash
# Distributed Training Launcher for 4x H100 NVL GPUs
# Uses torchrun for proper multi-GPU training with DDP/NCCL

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() { echo -e "\n${BLUE}========== $1 ==========${NC}\n"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse command line arguments
NUM_GPUS=4  # Default for H100 NVL system
MAX_PARQUET_FILES=""
CONFIG_PATH="config/training_config.yaml"

for arg in "$@"; do
    if [[ "$arg" =~ ^--gpus=([0-9]+)$ ]]; then
        NUM_GPUS="${BASH_REMATCH[1]}"
    elif [[ "$arg" =~ ^--config=(.+)$ ]]; then
        CONFIG_PATH="${BASH_REMATCH[1]}"
    elif [[ "$arg" =~ ^--max-parquet-files=([0-9]+)$ ]]; then
        MAX_PARQUET_FILES="--max-parquet-files ${BASH_REMATCH[1]}"
    fi
done

print_header "DISTRIBUTED TRAINING LAUNCHER"
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Config: $CONFIG_PATH"
echo "  Max Parquet Files: ${MAX_PARQUET_FILES:-All files}"
echo ""

# Validate CUDA
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. CUDA not available?"
    exit 1
fi

print_info "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check CUDA_VISIBLE_DEVICES
print_info "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set (will use all GPUs)}"

# Validate number of GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    print_error "Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

print_success "Using $NUM_GPUS GPUs for distributed training"

# Set optimal number of workers per GPU (adjust based on CPU cores)
CPU_CORES=$(nproc)
WORKERS_PER_GPU=$((CPU_CORES / NUM_GPUS))
WORKERS_PER_GPU=$((WORKERS_PER_GPU > 4 ? 4 : WORKERS_PER_GPU))  # Cap at 4

print_info "CPU cores: $CPU_CORES"
print_info "Workers per GPU: $WORKERS_PER_GPU"

# Set threading environment variables for optimal performance
export OMP_NUM_THREADS=$WORKERS_PER_GPU
export MKL_NUM_THREADS=$WORKERS_PER_GPU

# NCCL settings for H100 NVL (NVLink topology)
export NCCL_DEBUG=INFO  # Use INFO for production, TRACE for debugging
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback and docker interfaces
export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA
export NCCL_P2P_LEVEL=NVL  # Use NVLink for peer-to-peer communication

# H100-specific optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For H100 optimizations
export NCCL_NVLS_ENABLE=1  # Enable NVLink Sharp for H100

print_header "LAUNCHING DISTRIBUTED TRAINING"
echo "Backend: NCCL"
echo "Processes: $NUM_GPUS (1 per GPU)"
echo "Master: localhost:29500"
echo ""

# Launch with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    --standalone \
    scripts/train_asr.py \
    --config "$CONFIG_PATH" \
    $MAX_PARQUET_FILES

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Distributed training completed successfully!"
else
    print_error "Distributed training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

print_header "TRAINING COMPLETE"
echo "Check logs in: logs/training/"
echo "Checkpoints in: models/checkpoints/"
echo "Final model in: models/final/"
