# MWC ASR Training System - Deployment Guide

## Quick Setup on New Server

### Prerequisites
- Linux server with GPU (NVIDIA recommended)
- CUDA drivers installed
- Conda/Anaconda installed
- SSH access to transfer files

---

## 📦 STEP 1: Transfer Files to New Server

```bash
# On source server - tar file created at:
/data/V2V/en_sp/V4_Arpit/MWC_deployment_20251221.tar.gz

# Transfer using SCP
scp /data/V2V/en_sp/V4_Arpit/MWC_deployment_20251221.tar.gz user@new-server:/path/to/destination/

# Or use rsync
rsync -avz --progress /data/V2V/en_sp/V4_Arpit/MWC_deployment_20251221.tar.gz user@new-server:/path/to/destination/
```

---

## 🛠️ STEP 2: Extract on New Server

```bash
# Extract tar file
cd /path/to/destination/
tar -xzf MWC_deployment_20251221.tar.gz
cd MWC/
```

---

## ⚙️ STEP 3: Update Configuration File

**File to Edit:** `config/training_config.yaml`

### 3.1 Data Configuration (REQUIRED)

```yaml
data:
  # CHANGE THIS - Point to your new audio data location
  audio_path: "/path/to/your/new/audio/data"
  
  # OR use HuggingFace dataset
  use_huggingface: true
  huggingface_dataset: "YourUsername/YourDataset"  # CHANGE THIS
  huggingface_split: "train"
```

**Choose ONE option:**
- **Option A:** Local audio files → Set `use_huggingface: false` and update `audio_path`
- **Option B:** HuggingFace dataset → Set `use_huggingface: true` and update `huggingface_dataset`

### 3.2 Model Configuration (Optional)

```yaml
model:
  base_model: "openai/whisper-large-v3"  # Change if using different model
  language: "spanish"                     # Change target language
  task: "transcribe"                      # Or "translate"
```

### 3.3 Output Paths (Optional - Auto-created if not exists)

```yaml
training:
  output_dir: "./models/checkpoints"  # Checkpoints saved here
  logging_dir: "./logs/training"      # Logs saved here
```

**No need to change unless you want different paths**

### 3.4 Training Parameters (Optional - Already optimized)

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 1.0e-05
  gradient_accumulation_steps: 2
```

**Only adjust if you know your GPU memory limitations**

---

## 🚀 STEP 4: Run Training

```bash
# Make script executable
chmod +x start_training.sh

# Start training (interactive)
./start_training.sh

# Or run in background
nohup ./start_training.sh > training_output.log 2>&1 &
```

---

## 📝 What the Script Does Automatically

1. ✅ Checks for GPU and CUDA
2. ✅ Creates conda environment `mwc_exp`
3. ✅ Installs all dependencies
4. ✅ Creates required folders (logs/, models/, data/)
5. ✅ Validates configuration
6. ✅ Downloads base model
7. ✅ Loads and preprocesses data
8. ✅ Starts training with progress monitoring
9. ✅ Saves checkpoints automatically

---

## 🔧 Configuration Summary

### Mandatory Changes
| Parameter | Location | Action |
|-----------|----------|--------|
| **Data Source** | `config/training_config.yaml` → `data.audio_path` OR `data.huggingface_dataset` | Update to new data location |

### Optional Changes
| Parameter | Location | Default | When to Change |
|-----------|----------|---------|----------------|
| **Language** | `config/training_config.yaml` → `model.language` | `spanish` | Different language |
| **Base Model** | `config/training_config.yaml` → `model.base_model` | `openai/whisper-large-v3` | Different model size |
| **Batch Size** | `config/training_config.yaml` → `training.per_device_train_batch_size` | `8` | GPU memory issues |
| **Epochs** | `config/training_config.yaml` → `training.num_train_epochs` | `3` | Longer/shorter training |

---

## 📊 Monitoring Training

### Check Progress
```bash
# View live logs
tail -f logs/training/*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Output Locations
- **Checkpoints:** `models/checkpoints/checkpoint-{step}/`
- **Final Model:** `models/final/`
- **Logs:** `logs/training/training_run_*.log`

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size in `config/training_config.yaml`
```yaml
training:
  per_device_train_batch_size: 4  # Reduce from 8 to 4 or 2
```

### Issue: Data Not Found
**Solution:** Verify path in config file
```bash
# Check if path exists
ls -la /path/to/your/new/audio/data
```

### Issue: HuggingFace Authentication
**Solution:** Login to HuggingFace
```bash
# In conda environment
conda activate mwc_exp
huggingface-cli login
```

### Issue: Permission Denied
**Solution:** Make script executable
```bash
chmod +x start_training.sh
```

---

## ✅ Verification Checklist

Before running training, verify:

- [ ] Tar file extracted successfully
- [ ] `config/training_config.yaml` updated with new data path
- [ ] Data path exists and contains audio files (if using local data)
- [ ] GPU is accessible (`nvidia-smi` works)
- [ ] Internet connection available (for downloading base model)
- [ ] Sufficient disk space (at least 50GB recommended)
- [ ] Script is executable (`chmod +x start_training.sh`)

---

## 📞 Quick Reference

**Single command to change data source:**
```bash
# Edit config file
nano config/training_config.yaml

# Find line 10 and change:
audio_path: "/data/V2V/en_sp/V4_Arpit/data/asr_samples"
# TO:
audio_path: "/your/new/data/path"
```

**That's it! Run training:**
```bash
./start_training.sh
```

---

## 🎯 Summary

**Minimum steps to get running:**
1. Transfer and extract tar file
2. Edit `config/training_config.yaml` - Update data path (line 10)
3. Run `./start_training.sh`

**Everything else is handled automatically!**
