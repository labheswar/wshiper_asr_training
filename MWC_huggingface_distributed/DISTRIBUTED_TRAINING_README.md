# Distributed Training Guide for 4x H100 NVL GPUs

## Overview

This guide explains how to run the MWC ASR Training System with proper distributed training across all 4 H100 NVL GPUs using PyTorch's DistributedDataParallel (DDP) with NCCL backend.

## Problem Addressed

**Root Cause**: Training was not running in fully distributed multi-process mode, causing compute to concentrate on GPU 0 (rank 0) while other GPUs only held model memory without performing active computation.

**Solution**: Implemented PyTorch Distributed Data Parallel (DDP) with proper rank-to-GPU binding, ensuring one process per GPU with equal compute distribution.

## Architecture

### NVLink Topology (Dell R760xa with H100 NVL)
```
GPU 0 ↔ GPU 1 (NVLink pair)
GPU 2 ↔ GPU 3 (NVLink pair)
All GPUs connected via NVSwitch
```

###Distributed Training Setup
```
Process 0 (Rank 0) → GPU 0
Process 1 (Rank 1) → GPU 1
Process 2 (Rank 2) → GPU 2
Process 3 (Rank 3) → GPU 3
```

## Quick Start

### Option 1: Using start_training.sh with Distributed Mode (Recommended)

The main training script now supports distributed training with simple flags:

```bash
# Distributed training on 4 GPUs
./start_training.sh --distributed --gpus=4

# Auto-detect all available GPUs
./start_training.sh --distributed

# Test with 2 GPUs and limited data
./start_training.sh --distributed --gpus=2 --10
```

**Arguments**:
- `--distributed`: Enable distributed training mode
- `--gpus=N`: Number of GPUs to use (auto-detects if not specified)
- `--N`: Limit parquet files (e.g., `--10` for 10 files)
- `--all`: Use all parquet files
- `--clean`: Clean up old training runs

**Examples**:
```bash
# Single GPU (default behavior, no changes needed)
./start_training.sh

# Distributed on all GPUs with all data
./start_training.sh --distributed --all

# Distributed on 2 GPUs with 5 parquet files (testing)
./start_training.sh --distributed --gpus=2 --5
```

### Option 2: Direct torchrun Command

For manual control, you can use torchrun directly:

```bash
torchrun \
    --nproc_per_node=4 \
    --standalone \
    scripts/train_asr.py \
    --config config/training_config.yaml
```

**For testing with limited data**:
```bash
torchrun \
    --nproc_per_node=4 \
    --standalone \
    scripts/train_asr.py \
    --config config/training_config.yaml \
    --max-parquet-files 10
```

## Configuration

### training_config.yaml

The configuration file now includes distributed training settings:

```yaml
training:
  # ... existing settings ...
  
  # Distributed training configuration
  distributed_strategy: "ddp"  # Options: ddp, fsdp
  ddp_backend: "nccl"  # NCCL for NVIDIA GPUs
  ddp_find_unused_parameters: false  # Performance optimization
  
  # Optional FSDP config (for very large models)
  # fsdp_config:
  #   fsdp_transformer_layer_cls_to_wrap: "WhisperEncoderLayer"
  #   fsdp_backward_prefetch: "backward_pre"
  #   fsdp_state_dict_type: "full_state_dict"
```

### Effective Batch Size

With distributed training, the effective global batch size is:
```
Effective Batch Size = per_device_batch_size × gradient_accumulation_steps × num_gpus
```

**Example** (default config):
- Per-device batch size: 4
- Gradient accumulation: 4
- Number of GPUs: 4
- **Effective global batch size: 64**

## Environment Variables

The `start_training.sh` script automatically sets optimal NCCL variables for H100 NVL when using `--distributed` flag:

```bash
export NCCL_DEBUG=INFO  # Logging level
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback
export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA
export NCCL_P2P_LEVEL=NVL  # Use NVLink for peer-to-peer
export NCCL_NVLS_ENABLE=1  # Enable NVLink Sharp (H100-specific)
export CUDA_DEVICE_MAX_CONNECTIONS=1  # H100 optimization
```

These are automatically configured - no manual setup required when using `./start_training.sh --distributed`.

## Verification

### 1. Check GPU Utilization

```bash
watch -n 1 nvidia-smi
```

**Expected behavior**:
- All 4 GPUs show similar utilization (60-85%)
- Memory usage balanced across GPUs
- Power draw balanced across GPUs

### 2. Monitor Training Logs

Logs will show rank prefixes:
```
[Rank 0] Distributed Training Initialized: Rank: 0/4, Local Rank (GPU): 0
[Rank 1] Distributed Training Initialized: Rank: 1/4, Local Rank (GPU): 1
[Rank 2] Distributed Training Initialized: Rank: 2/4, Local Rank (GPU): 2
[Rank 3] Distributed Training Initialized: Rank: 3/4, Local Rank (GPU): 3
```

### 3. Verify Process Distribution

```bash
ps aux | grep train_asr
```

Should show 4 Python processes (one per GPU).

### 4. Check CUDA_VISIBLE_DEVICES

Inside the container/pod:
```bash
echo $CUDA_VISIBLE_DEVICES
```

Expected: `0,1,2,3`

## Code Changes Summary

### 1. New Functions
- `setup_distributed()`: Initializes distributed training environment
- `cleanup_distributed()`: Cleanup on shutdown
- Updated `setup_logging()`: Multi-process logging with rank prefixes

### 2. Trainer Class Updates
- **Initialization**: Added `self.dist_info = setup_distributed()`
- **Device Setup**: Rank-to-GPU binding with `torch.cuda.set_device(local_rank)`
- **Model Loading**: Updated `device_map` for distributed mode
- **Training Arguments**: Added DDP-specific parameters
- **Synchronization**: Added `dist.barrier()` calls for process coordination

### 3. New Files
- `launch_distributed.sh`: Bash launcher with optimal NCCL settings
- `DISTRIBUTED_TRAINING_GUIDE.py`: Configuration guide
- `MANUAL_UPDATES_train_asr.py`: Manual update instructions

## Troubleshooting

### Issue: "Address already in use"
**Solution**: Change master port or kill existing process
```bash
# Change port in launch script (default 29500)
# Or kill existing process
lsof -ti:29500 | xargs kill -9
```

### Issue: Uneven GPU utilization
**Solutions**:
1. Verify `CUDA_VISIBLE_DEVICES`: `echo $CUDA_VISIBLE_DEVICES`
2. Check all processes started: `ps aux | grep train_asr`
3. Verify NCCL_P2P_LEVEL=NVL is set
4. Check NCCL logs for errors

### Issue: Out of memory on one GPU
**Solutions**:
1. Reduce `per_device_train_batch_size` in config
2. Enable `gradient_checkpointing: true`
3. Check if one GPU has extra processes: `nvidia-smi`

### Issue: Process hangs at initialization
**Solutions**:
1. Check firewall allows localhost communication
2. Verify all GPUs accessible: `nvidia-smi`
3. Set `NCCL_DEBUG=TRACE` for detailed debugging
4. Check network interfaces: `ifconfig`

### Issue: NCCL errors
**Solutions**:
1. Ensure NCCL compatible with CUDA version
2. Check NCCL version: `python -c "import torch; print(torch.cuda.nccl.version())"`
3. Update PyTorch if needed: `pip install --upgrade torch`

## Performance Expectations

### Single GPU (Baseline)
- GPU Utilization: ~70-80%
- Effective Batch Size: 16 (4 × 4)
- Training Speed: X samples/sec

### 4 GPUs (Distributed)
- GPU Utilization per GPU: ~60-85%
- Effective Batch Size: 64 (4 × 4 × 4)
- Training Speed: ~3.5-3.8X faster
- Linear scaling efficiency: 87-95%

### NVLink Benefits
- GPU-to-GPU bandwidth: ~900 GB/s (NVLink 4.0)
- Reduced communication overhead
- Faster gradient synchronization

## AI3 / Kubernetes Configuration

Ensure your pod specification includes:

```yaml
resources:
  limits:
    nvidia.com/gpu: 4  # Request all 4 GPUs
  requests:
    nvidia.com/gpu: 4

env:
  - name: NCCL_DEBUG
    value: "INFO"
  - name: NCCL_SOCKET_IFNAME
    value: "^lo,docker"
```

No GPU affinity or masking should be applied.

## Advanced: FSDP for Very Large Models

For models too large for single GPU memory, use Fully Sharded Data Parallel (FSDP):

1. Update `training_config.yaml`:
```yaml
training:
  distributed_strategy: "fsdp"
  fsdp_config:
    fsdp_transformer_layer_cls_to_wrap: "WhisperEncoderLayer"
    fsdp_backward_prefetch: "backward_pre"
    fsdp_state_dict_type: "full_state_dict"
    fsdp_offload_params: false
```

2. Launch same way with torchrun

FSDP shards model parameters, gradients, and optimizer states across GPUs.

## Monitoring

### Tensorboard (if enabled)
```bash
tensorboard --logdir logs/training/
```

### Custom Metrics
Check `logs/training/training_results.json` for:
- Total training time
- GPU count used
- Per-epoch time
- Performance summaries

### Real-time GPU Monitoring
```bash
# Terminal 1: Training
./launch_distributed.sh

# Terminal 2: GPU monitoring
watch -n 0.5 nvidia-smi

# Terminal 3: Process monitoring
watch -n 1 'ps aux | grep train_asr | grep -v grep'
```

## References

- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [H100 NVL Specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)

## Support

For issues specific to distributed training, check:
1. NCCL logs (set `NCCL_DEBUG=TRACE`)
2. PyTorch distributed logs
3. GPU topology: `nvidia-smi topo -m`
4. Process status: `ps aux | grep train_asr`

---

**Last Updated**: January 6, 2026  
**Tested On**: Dell R760xa with 4x H100 NVL GPUs  
**PyTorch Version**: 2.0+  
**NCCL Version**: 2.18+
