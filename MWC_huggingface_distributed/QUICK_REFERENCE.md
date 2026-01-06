# Quick Reference: Distributed Training on 4x H100 NVL

## TL;DR

```bash
# 1. Test distributed setup (IMPORTANT - do this first!)
torchrun --nproc_per_node=4 --standalone validate_distributed.py

# 2. Run distributed training on 4 GPUs
./start_training.sh --distributed --gpus=4

# 3. Or auto-detect all GPUs
./start_training.sh --distributed
```

## One-Liners

### Single GPU (default)
```bash
./start_training.sh
```

### Distributed training on all available GPUs
```bash
./start_training.sh --distributed
```

### Distributed training on 4 specific GPUs
```bash
./start_training.sh --distributed --gpus=4
```

### Distributed with limited data (testing)
```bash
./start_training.sh --distributed --gpus=2 --10
```

### Direct torchrun (manual control)
```bash
torchrun --nproc_per_node=4 --standalone scripts/train_asr.py --config config/training_config.yaml
```

### Monitor GPUs in real-time
```bash
watch -n 1 nvidia-smi
```

### Check running processes
```bash
ps aux | grep train_asr | grep -v grep
```

### Kill stuck training
```bash
pkill -f train_asr
```

## What Changed?

| File | Changes |
|------|---------|
| `scripts/train_asr.py` | ✅ Added distributed functions<br>⚠️ Needs manual updates (see MANUAL_UPDATES_train_asr.py) |
| `config/training_config.yaml` | ✅ Added distributed config |
| `start_training.sh` | ✅ UPDATED - Now supports distributed mode |
| `validate_distributed.py` | ✅ NEW - Test script |

## Expected GPU Utilization

**Before:**
- GPU 0: 70-80% ⚡
- GPU 1-3: 0-10% 💤

**After:**
- GPU 0-3: 60-85% ⚡⚡⚡⚡

## Key Environment Variables

Auto-set by `launch_distributed.sh`:
- `NCCL_P2P_LEVEL=NVL` - Use NVLink
- `NCCL_NVLS_ENABLE=1` - H100 optimization
- `NCCL_DEBUG=INFO` - Logging

## Effective Batch Size

```
Global Batch Size = per_device (4) × grad_accum (4) × GPUs (4) = 64
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Address in use | `lsof -ti:29500 \| xargs kill -9` |
| Uneven GPU usage | Check `CUDA_VISIBLE_DEVICES` |
| NCCL timeout | `export NCCL_TIMEOUT=1800` |
| Import errors | Apply manual updates from MANUAL_UPDATES_train_asr.py |

## Files to Review

1. **IMPLEMENTATION_SUMMARY.md** - What was changed
2. **DISTRIBUTED_TRAINING_README.md** - Complete guide
3. **MANUAL_UPDATES_train_asr.py** - Code changes needed
4. **validate_distributed.py** - Test script

## Validation Steps

- [ ] Run `validate_distributed.py` with 4 GPUs
- [ ] All processes show correct rank and GPU binding
- [ ] All-reduce test passes
- [ ] Test training with 2 parquet files completes
- [ ] All 4 GPUs show balanced utilization
- [ ] Training speed 3-4x faster than single GPU

## Support

Check logs in:
- Console output (prefixed with [Rank N])
- `logs/training/training_run_*.log` (main process only)

---

**Need Help?** See DISTRIBUTED_TRAINING_README.md for full documentation
