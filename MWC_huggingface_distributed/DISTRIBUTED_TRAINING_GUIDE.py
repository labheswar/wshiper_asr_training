# ============================================================================
# DISTRIBUTED TRAINING CONFIGURATION GUIDE
# For MWC ASR Training System with 4x H100 NVL GPUs
# ============================================================================

"""
SUMMARY OF CHANGES FOR DISTRIBUTED TRAINING
-------------------------------------------

1. IMPORTS UPDATED (Line ~37):
   - Added: import torch.distributed as dist

2. NEW FUNCTIONS ADDED (After line ~195):
   - setup_distributed(): Initializes distributed training
   - cleanup_distributed(): Cleanup on shutdown
   - setup_logging(): Updated to handle multi-process logging

3. MWCASRTrainer CLASS CHANGES:
   
   a) __init__ method (Line ~690):
      - Added: self.dist_info = setup_distributed() at the start
      - Updated logging to use is_main_process flag
      - Added distributed configuration logging
      - Updated device setup for rank-to-GPU binding
   
   b) load_model() method (Line ~888):
      - Updated device_map for distributed mode
      - Changed from device_map="auto" to device_map={'': local_rank}
   
   c) train() method (Line ~1059):
      - Added distributed-specific TrainingArguments
      - Added: local_rank, ddp_backend, ddp_find_unused_parameters
      - Added synchronization barriers
      - Updated batch size calculations for multi-GPU
   
   d) run() method (Line ~1258):
      - Added cleanup_distributed() call on exit

4. NEW CONFIGURATION OPTIONS (training_config.yaml):
   - distributed_strategy: 'ddp' or 'fsdp'
   - ddp_backend: 'nccl'
   - ddp_find_unused_parameters: false

5. NEW LAUNCH SCRIPT:
   - launch_distributed.sh: torchrun launcher for 4 GPUs
"""

# ============================================================================
# CONFIGURATION FILE UPDATES NEEDED
# ============================================================================

CONFIG_ADDITIONS = """
# Add to config/training_config.yaml under 'training' section:

training:
  # ... existing config ...
  
  # Distributed training settings
  distributed_strategy: "ddp"  # Options: ddp, fsdp
  ddp_backend: "nccl"  # NCCL for NVIDIA GPUs
  ddp_find_unused_parameters: false  # Set true only if needed
  
  # FSDP configuration (optional, for very large models)
  fsdp_config:  # Leave empty for DDP, or configure for FSDP
    # fsdp_transformer_layer_cls_to_wrap: "WhisperEncoderLayer"
    # fsdp_backward_prefetch: "backward_pre"
    # fsdp_state_dict_type: "full_state_dict"
"""

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

USAGE = """
RUNNING DISTRIBUTED TRAINING
-----------------------------

1. Single GPU (no changes needed):
   python scripts/train_asr.py --config config/training_config.yaml

2. Multi-GPU with torchrun (RECOMMENDED for 4x H100):
   torchrun --nproc_per_node=4 --standalone scripts/train_asr.py --config config/training_config.yaml

3. Using the new launcher script:
   chmod +x launch_distributed.sh
   ./launch_distributed.sh

4. With limited data (testing):
   torchrun --nproc_per_node=4 --standalone scripts/train_asr.py --config config/training_config.yaml --max-parquet-files 10

ENVIRONMENT VARIABLES AUTOMATICALLY SET BY TORCHRUN:
- LOCAL_RANK: GPU ID for this process (0-3)
- RANK: Global rank (0-3 for single node)
- WORLD_SIZE: Total processes (4 for 4 GPUs)
- MASTER_ADDR: localhost
- MASTER_PORT: 29500

VERIFICATION:
1. Check GPU utilization across all 4 GPUs:
   watch -n 1 nvidia-smi

2. Expected behavior:
   - All 4 GPUs show similar utilization (60-85%)
   - Memory usage balanced across GPUs
   - Power usage balanced across GPUs

3. Logs will show:
   [Rank 0], [Rank 1], [Rank 2], [Rank 3] prefixes
   Each process bound to its GPU
"""

# ============================================================================
# NCCL OPTIMIZATION FOR H100 NVL
# ============================================================================

NCCL_ENV_VARS = """
OPTIMAL ENVIRONMENT VARIABLES FOR 4x H100 NVL:
----------------------------------------------

# Set before running torchrun:
export NCCL_DEBUG=INFO  # Or WARN for less verbose
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback
export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA
export NCCL_P2P_LEVEL=NVL  # Use NVLink for peer-to-peer
export NCCL_NVLS_ENABLE=1  # Enable NVLink Sharp (H100-specific)
export CUDA_DEVICE_MAX_CONNECTIONS=1  # H100 optimization

These are included in launch_distributed.sh
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = """
COMMON ISSUES AND SOLUTIONS:
----------------------------

1. "Address already in use" error:
   - Change master_port in launch script (default 29500)
   - Or kill existing process: lsof -ti:29500 | xargs kill -9

2. Uneven GPU utilization:
   - Verify CUDA_VISIBLE_DEVICES shows all GPUs: echo $CUDA_VISIBLE_DEVICES
   - Check all processes started: ps aux | grep train_asr
   - Verify NCCL_P2P_LEVEL=NVL is set

3. Out of memory on one GPU:
   - Reduce per_device_train_batch_size in config
   - Enable gradient_checkpointing: true
   - Check if one GPU has extra processes

4. Slow initialization:
   - Normal for first run (model download)
   - Each process loads model to its GPU
   - Barrier synchronizes all processes

5. Process hangs at initialization:
   - Check firewall allows localhost communication
   - Verify all GPUs accessible: nvidia-smi
   - Check NCCL_DEBUG=INFO for detailed errors
"""

print("Distributed Training Configuration Guide")
print("=" * 80)
print("\nThis file documents all changes needed for distributed training.")
print("\nKey changes made:")
print("1. Added distributed setup functions")
print("2. Updated trainer class for multi-GPU support")
print("3. Created launch script for torchrun")
print("\nSee inline comments above for details.")
