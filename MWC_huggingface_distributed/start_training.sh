#!/bin/bash
# MWC ASR Training System - Training Script
# Installs dependencies via pip and runs training in background

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
CLEAN_MODE=false
MAX_PARQUET_FILES=""
NUM_GPUS=""
DISTRIBUTED_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--clean" ]; then
        CLEAN_MODE=true
    elif [ "$arg" = "--all" ]; then
        MAX_PARQUET_FILES=""  # Use all files
    elif [ "$arg" = "--distributed" ]; then
        DISTRIBUTED_MODE=true
    elif [[ "$arg" =~ ^--gpus=([0-9]+)$ ]]; then
        NUM_GPUS="${BASH_REMATCH[1]}"
        DISTRIBUTED_MODE=true
    elif [[ "$arg" =~ ^--([0-9]+)$ ]]; then
        # Extract number from --2, --3, --10, etc.
        MAX_PARQUET_FILES="--max-parquet-files ${BASH_REMATCH[1]}"
    fi
done

# Cleanup function for --clean mode
cleanup() {
    if [ "$CLEAN_MODE" = true ]; then
        print_header "CLEANUP MODE"
        print_info "Removing generated files and directories..."
        
        # Clean up models
        if [ -d "models" ]; then
            rm -rf models
            print_success "Removed models directory"
        fi
        
        # Clean up logs
        if [ -d "logs" ]; then
            rm -rf logs
            print_success "Removed logs directory"
        fi
        
        # Clean up cache
        if [ -d "__pycache__" ]; then
            find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
            print_success "Removed cache directories"
        fi
        
        print_success "Cleanup completed!"
        exit 0
    fi
}

# Handle cleanup mode
cleanup

clear
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MWC ASR TRAINING - Production Ready System  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}\n"

# Check for Python
print_header "1. ENVIRONMENT CHECK"
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_success "Python detected: $PYTHON_VERSION"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    print_success "GPU: $gpu_info"
    print_info "Available GPUs: $AVAILABLE_GPUS"
    HAS_GPU=true
    
    # Auto-detect GPU count for distributed mode if not specified
    if [ "$DISTRIBUTED_MODE" = true ] && [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=$AVAILABLE_GPUS
        print_info "Auto-detected $NUM_GPUS GPUs for distributed training"
    fi
    
    # Validate GPU count
    if [ -n "$NUM_GPUS" ] && [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
        print_error "Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
        exit 1
    fi
else
    print_info "No GPU - will use CPU (slow)"
    HAS_GPU=false
    if [ "$DISTRIBUTED_MODE" = true ]; then
        print_error "Distributed mode requires GPUs"
        exit 1
    fi
fi

# Check pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    print_error "pip not found. Please install pip."
    exit 1
fi
PIP_CMD=$(command -v pip3 2>/dev/null || command -v pip)
print_success "pip detected"

# Install dependencies
print_header "2. INSTALL DEPENDENCIES"
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found"
    exit 1
fi

print_info "Installing dependencies from requirements.txt..."
$PIP_CMD install -r requirements.txt
print_success "All dependencies installed"

# Verify setup
print_header "3. VERIFICATION"
print_info "Verifying installation..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}')"
$PYTHON_CMD -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
$PYTHON_CMD -c "import transformers; print(f'Transformers: {transformers.__version__}')"
$PYTHON_CMD -c "import numpy; print(f'NumPy: {numpy.__version__}')"
print_success "All verifications passed"

# Check data availability
print_header "4. DATA PREPARATION"
print_info "Reading parquet directory from config..."

# Extract parquet directory from config YAML
PARQUET_DIR=$($PYTHON_CMD -c "
import yaml
with open('config/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['data']['parquet']['file_path'])
" 2>/dev/null)

if [ -z "$PARQUET_DIR" ]; then
    print_error "Failed to read parquet directory from config"
    exit 1
fi

print_info "Parquet directory: $PARQUET_DIR"

# Check if directory exists and has parquet files
if [ ! -d "$PARQUET_DIR" ]; then
    print_error "Parquet directory does not exist: $PARQUET_DIR"
    print_info "Please ensure the directory path in config/training_config.yaml is correct"
    exit 1
fi

PARQUET_COUNT=$(find "$PARQUET_DIR" -name '*.parquet' 2>/dev/null | wc -l)

if [ "$PARQUET_COUNT" -eq 0 ]; then
    print_error "No parquet files found in: $PARQUET_DIR"
    print_info "Please add parquet files to the directory specified in config"
    exit 1
fi

print_success "Found ${PARQUET_COUNT} parquet file(s)"

# Run pre-training tests
print_header "5. PRE-TRAINING TESTS"
print_info "Running system tests..."
$PYTHON_CMD test_training.py
print_success "All tests passed"

# Show configuration
print_header "6. CONFIGURATION"
$PYTHON_CMD << 'EOF'
import yaml
with open('config/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"  Model: {config['model']['base_model']}")
print(f"  Epochs: {config['training']['num_epochs']}")
print(f"  Batch size: {config['training']['per_device_train_batch_size']}")
print(f"  Data source: {config['data']['source_type']}")
EOF

# Start training
print_header "7. TRAINING"

# Determine training mode and configure environment
if [ "$DISTRIBUTED_MODE" = true ]; then
    print_header "DISTRIBUTED TRAINING CONFIGURATION"
    print_success "Distributed mode enabled with $NUM_GPUS GPUs"
    
    # Set optimal NCCL environment variables for H100 NVL
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=^lo,docker
    export NCCL_NET_GDR_LEVEL=5
    export NCCL_P2P_LEVEL=NVL
    export NCCL_NVLS_ENABLE=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    
    # Set threading for optimal performance
    CPU_CORES=$(nproc)
    WORKERS_PER_GPU=$((CPU_CORES / NUM_GPUS))
    WORKERS_PER_GPU=$((WORKERS_PER_GPU > 4 ? 4 : WORKERS_PER_GPU))
    export OMP_NUM_THREADS=$WORKERS_PER_GPU
    export MKL_NUM_THREADS=$WORKERS_PER_GPU
    
    print_info "NCCL Backend: Configured for H100 NVL topology"
    print_info "NVLink: GPU 0↔1, GPU 2↔3 (paired)"
    print_info "Workers per GPU: $WORKERS_PER_GPU"
    print_info "CPU Cores: $CPU_CORES"
    
    # Show GPU topology if available
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r idx name mem; do
            echo "  GPU $idx: $name ($mem)"
        done
    fi
    
    print_info "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-All GPUs}"
    echo ""
fi

print_info "Starting ASR training in background with nohup..."

# Show parquet file limit if specified
if [ -n "$MAX_PARQUET_FILES" ]; then
    print_info "Parquet file limit: $MAX_PARQUET_FILES"
else
    print_info "Using all parquet files"
fi
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training/run_${TIMESTAMP}.log"
PID_FILE="logs/training/run_${TIMESTAMP}.pid"
mkdir -p logs/training

print_info "Training will run in background. Logs: $LOG_FILE"

# Launch training based on mode
if [ "$DISTRIBUTED_MODE" = true ]; then
    print_info "Launching distributed training with torchrun ($NUM_GPUS processes)..."
    nohup torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        --standalone \
        scripts/train_asr.py \
        --config config/training_config.yaml \
        $MAX_PARQUET_FILES > "$LOG_FILE" 2>&1 &
else
    print_info "Launching single-GPU training..."
    nohup $PYTHON_CMD scripts/train_asr.py --config config/training_config.yaml $MAX_PARQUET_FILES > "$LOG_FILE" 2>&1 &
fi

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

print_success "Training started in background (PID: $TRAIN_PID)"
print_info "Monitor progress: tail -f $LOG_FILE"
print_info "Check if running: ps -p $TRAIN_PID"
print_info "Stop training: kill $TRAIN_PID"
EXIT_CODE=0

# Results
print_header "8. TRAINING INFO"
print_success "Training has been started in background!"
echo ""
if [ "$DISTRIBUTED_MODE" = true ]; then
    echo "  Mode: DISTRIBUTED ($NUM_GPUS GPUs)"
    echo "  Backend: NCCL with NVLink optimization"
else
    echo "  Mode: Single GPU/CPU"
fi
echo "  Process ID: $TRAIN_PID"
echo "  PID file: $PID_FILE"
echo "  Log file: $LOG_FILE"
if [ -n "$MAX_PARQUET_FILES" ]; then
    echo "  Parquet limit: $MAX_PARQUET_FILES"
fi
echo ""
echo "Useful commands:"
echo "  Monitor logs: tail -f $LOG_FILE"
if [ "$DISTRIBUTED_MODE" = true ]; then
    echo "  Monitor GPUs: watch -n 1 nvidia-smi"
fi
echo "  Check status: ps -p $TRAIN_PID"
echo "  Stop training: kill $TRAIN_PID"
echo ""
echo "Expected outputs (after completion):"
echo "  Final model: models/final/"
echo "  Checkpoints: models/checkpoints/"
echo ""
echo "Usage examples:"
echo "  Single GPU: $0"
echo "  All files: $0 --all"
echo "  Limited files: $0 --10"
echo "  Distributed 4 GPUs: $0 --distributed --gpus=4"
echo "  Distributed auto-detect GPUs: $0 --distributed"
echo "  Distributed with limited data: $0 --distributed --gpus=2 --10"
echo "  Clean old runs: $0 --clean"
echo ""
if [ "$DISTRIBUTED_MODE" = true ]; then
    print_info "Distributed training is running on $NUM_GPUS GPUs."
    print_info "Expected speedup: ~${NUM_GPUS}x (linear scaling)"
fi
print_info "Training is running in background. You can close this terminal."

exit 0