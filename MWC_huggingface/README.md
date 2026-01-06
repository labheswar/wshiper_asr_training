# MWC ASR Training System - Production Ready

High-performance Whisper ASR training system optimized for large-scale parquet datasets.

## Quick Start

```bash
# 1. Place your parquet files in the data directory
# You can use a symbolic link to point to your data on another server:
ln -s /path/to/your/parquet/files data/audio/parquet_files

# 2. Run training (creates fresh environment, installs dependencies, trains, cleans up)
./start_training.sh
```

## System Requirements

- **OS**: Linux (Ubuntu 20.04+)
- **Conda**: Miniconda or Anaconda
- **GPU**: CUDA-capable GPU (Tesla T4, V100, A100, etc.)
- **RAM**: 32GB+ recommended
- **VRAM**: 16GB+ recommended

## Features

✅ Auto Environment Management (creates fresh conda env)
✅ Auto Cleanup (removes env after training)  
✅ 4-bit Quantization (QLoRA)
✅ Auto CPU/GPU Detection
✅ Parquet Directory Mode (recursive loading)
✅ Production Ready

## Configuration

Edit `config/training_config.yaml`:
- Model: whisper-large-v3
- Language: spanish
- Batch size: 4
- Data path: data/audio/parquet_files/

## Using Data from Another Server

```bash
# Option 1: Symbolic link (recommended)
ln -s /mnt/nfs/parquet_data data/audio/parquet_files

# Option 2: Update config file
# Edit config/training_config.yaml -> data.parquet.file_path

# Option 3: Copy data
rsync -avz user@server:/path/to/parquet/ data/audio/parquet_files/
```

## Outputs

- Final Model: `models/final/`
- Checkpoints: `models/checkpoints/`
- Logs: `logs/training/`

## Data Format

Parquet files should have:
- `audio` column: dict with audio bytes/array
- `text` column: transcription string

---
**Production Ready** ✅ | December 22, 2025
