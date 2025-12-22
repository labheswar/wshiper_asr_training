#!/bin/bash
# MWC ASR Training System - Complete Setup & Training
# Single script for installation, setup, and training

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

clear
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MWC ASR TRAINING - Production Ready System  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}\n"

# Check for conda
print_header "1. ENVIRONMENT CHECK"
if command -v conda &> /dev/null; then
    print_success "Conda detected - using conda for installation"
    USE_CONDA=true
else
    USE_CONDA=false
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    print_success "GPU: $gpu_info"
    HAS_GPU=true
else
    print_info "No GPU - will use CPU (slow)"
    HAS_GPU=false
fi

# Setup environment
print_header "2. CONDA ENVIRONMENT"
ENV_NAME="mwc_exp"

if [ "$USE_CONDA" = true ]; then
    # Clean up existing environment if requested
    if conda env list | grep -q "^$ENV_NAME "; then
        print_info "Found existing environment: $ENV_NAME"
        print_info "Removing old environment for fresh installation..."
        conda deactivate 2>/dev/null || true
        conda env remove -n $ENV_NAME -y
        print_success "Old environment removed"
    fi
    
    # Create fresh environment
    print_info "Creating conda environment $ENV_NAME with Python 3.11..."
    conda create -n $ENV_NAME python=3.11 -y
    print_success "Environment created"
    
    # Activate conda env
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    print_success "Activated: $ENV_NAME"
    
    # Install dependencies (always fresh)
    print_header "3. DEPENDENCIES"
    print_info "Installing PyTorch via pip (avoids MKL conflicts)..."
    if [ "$HAS_GPU" = true ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_info "Installing transformers & training libraries..."
    pip install transformers datasets evaluate peft accelerate
    
    if [ "$HAS_GPU" = true ]; then
        print_info "Installing bitsandbytes for QLoRA..."
        pip install bitsandbytes
    fi
    
    print_info "Installing audio & utility libraries..."
    pip install soundfile librosa sacrebleu jiwer pyyaml pandas numpy tqdm
    
    print_info "Installing ONNX for model export..."
    pip install onnx onnxruntime
    
    print_success "All dependencies installed"
else
    print_error "Conda not found. Please install miniconda/anaconda."
    print_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Verify setup
print_header "4. VERIFICATION"
print_info "Testing Python environment..."
python3 -c "import sys; print(f'Python: {sys.version.split()[0]}')" || { print_error "Python test failed"; exit 1; }

print_info "Testing PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || { print_error "PyTorch import failed"; exit 1; }

print_info "Testing CUDA..."
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" || { print_error "CUDA test failed"; exit 1; }

print_info "Testing transformers..."
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || { print_error "Transformers import failed"; exit 1; }

print_info "Testing audio libraries..."
python3 -c "import soundfile, librosa; print('Audio libraries OK')" || { print_error "Audio libraries failed"; exit 1; }

print_success "All verifications passed"

# Show configuration
print_header "5. CONFIGURATION"
python3 << 'EOF'
import yaml
with open('config/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"  Model: {config['model']['base_model']}")
print(f"  Epochs: {config['training']['num_epochs']}")
print(f"  Batch: {config['training']['per_device_train_batch_size']}")
print(f"  Data: {'HuggingFace' if config['data']['use_huggingface'] else 'Local'}")
EOF

echo ""
if [ -t 0 ]; then
    read -p "Start training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Cancelled"
        exit 0
    fi
fi

# Start training
print_header "6. TRAINING"
print_info "Starting ASR training..."
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training/run_${TIMESTAMP}.log"
mkdir -p logs/training

python3 scripts/train_asr.py --config config/training_config.yaml 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# Results
print_header "7. COMPLETE"
if [ $EXIT_CODE -eq 0 ]; then
    print_success "Training completed! 🎉"
    echo ""
    echo "  Models: models/final/"
    echo "  Checkpoints: models/checkpoints/"
    echo "  Log: $LOG_FILE"
    echo ""
    echo "Model files:"
    ls -lh models/final/ 2>/dev/null || echo "  (No models found - training may have failed)"
    echo ""
    echo "Checkpoint files:"
    ls -lh models/checkpoints/ 2>/dev/null || echo "  (No checkpoints found)"
else
    print_error "Training failed (exit code: $EXIT_CODE)"
    echo "  Check: $LOG_FILE"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 "$LOG_FILE"
    exit $EXIT_CODE
fi

deactivate 2>/dev/null || true
print_success "Done!"
