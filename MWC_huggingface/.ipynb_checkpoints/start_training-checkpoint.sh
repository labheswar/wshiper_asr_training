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
for arg in "$@"; do
    if [ "$arg" = "--clean" ]; then
        CLEAN_MODE=true
    elif [ "$arg" = "--all" ]; then
        MAX_PARQUET_FILES=""  # Use all files
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
    print_success "GPU: $gpu_info"
    HAS_GPU=true
else
    print_info "No GPU - will use CPU (slow)"
    HAS_GPU=false
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
nohup $PYTHON_CMD scripts/train_asr.py --config config/training_config.yaml $MAX_PARQUET_FILES > "$LOG_FILE" 2>&1 &
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
echo "  Process ID: $TRAIN_PID"
echo "  PID file: $PID_FILE"
echo "  Log file: $LOG_FILE"
if [ -n "$MAX_PARQUET_FILES" ]; then
    echo "  Parquet limit: $MAX_PARQUET_FILES"
fi
echo ""
echo "Useful commands:"
echo "  Monitor logs: tail -f $LOG_FILE"
echo "  Check status: ps -p $TRAIN_PID"
echo "  Stop training: kill $TRAIN_PID"
echo ""
echo "Expected outputs (after completion):"
echo "  Final model: models/final/"
echo "  Checkpoints: models/checkpoints/"
echo ""
echo "Usage examples:"
echo "  Use all files: $0 --all"
echo "  Use 2 files: $0 --2"
echo "  Use 10 files: $0 --10"
echo ""
print_info "Training is running in background. You can close this terminal."
print_info "To clean up old runs, use: $0 --clean"

exit 0