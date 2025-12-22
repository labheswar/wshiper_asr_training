# MWC ASR Training System

**Modular Whisper Configuration for Production ASR Training**

Version: 1.0  
Date: December 19, 2025

---

## 🎯 Overview

The MWC (Modular Whisper Configuration) ASR Training System is a production-grade framework for training custom Automatic Speech Recognition models based on OpenAI's Whisper architecture. It features:

### ✨ Key Features

- **🔧 Plug-and-Play Architecture**: Easily swap data sources by changing configuration
- **🎵 Multi-Format Audio Support**: WAV, MP3, FLAC, OGG, M4A, AAC, WMA
- **⚡ Advanced Training**: QLoRA (4-bit quantization), Knowledge Distillation, FP16
- **💾 Checkpoint Management**: Auto-save every 5 epochs in SafeTensors format
- **📊 Real Metrics**: Genuine WER, CER, BLEU, chrF scores (no fabrication)
- **📦 Multiple Export Formats**: SafeTensors, ONNX, PyTorch
- **🚀 Production-Ready**: Optimized for deployment and scalability

---

## 📁 Project Structure

```
MWC/
├── config/
│   └── training_config.yaml         # Main configuration file
├── scripts/
│   ├── audio_loader.py              # Multi-format audio loader
│   ├── dataset_handler.py           # Pluggable dataset handler
│   ├── train_asr.py                 # Main training script
│   └── model_exporter.py            # Model export utilities
├── models/
│   ├── checkpoints/                 # Training checkpoints
│   │   ├── checkpoint-5/            # Epoch 5
│   │   ├── checkpoint-10/           # Epoch 10
│   │   └── ...
│   └── final/                       # Final exported models
│       ├── model_safetensors/
│       ├── model_onnx/
│       └── model_pytorch/
├── logs/
│   ├── training/                    # Training logs
│   └── evaluation/                  # Evaluation logs
├── data/
│   └── audio/                       # Audio files (pluggable)
└── docs/
    ├── README.md                    # This file
    ├── QUICKSTART.md                # Quick start guide
    └── TECHNICAL_GUIDE.md           # Technical documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ recommended)
- ~50GB disk space for training

### 1. Installation

```bash
# Navigate to project directory
cd /data/V2V/en_sp/V4_Arpit/MWC

# Install dependencies
pip install torch transformers datasets evaluate \
    peft bitsandbytes accelerate \
    soundfile librosa pydub \
    sacrebleu jiwer \
    onnx onnxruntime \
    pyyaml pandas numpy tqdm
```

### 2. Configure Data Source

Edit `config/training_config.yaml`:

**Option A: Use HuggingFace Dataset**
```yaml
data:
  use_huggingface: true
  huggingface_dataset: "MrDragonFox/Elise"
  huggingface_split: "train"
```

**Option B: Use Local Files**
```yaml
data:
  use_huggingface: false
  audio_path: "/path/to/your/audio/files"
  # Place metadata.csv or metadata.json in audio directory
```

**Option C: Use CSV File**
```yaml
data:
  use_huggingface: false
  audio_path: "/path/to/dataset.csv"
  # CSV should have 'audio' and 'text' columns
```

### 3. Run Training

```bash
cd scripts
python train_asr.py --config ../config/training_config.yaml
```

Training will:
- Load and validate data
- Configure QLoRA and distillation
- Train for 50 epochs
- Save checkpoints every 5 epochs
- Export final models in multiple formats
- Generate real evaluation metrics

### 4. Monitor Progress

```bash
# View logs
tail -f ../logs/training/training_run_*.log

# TensorBoard
tensorboard --logdir ../models/checkpoints
```

---

## ⚙️ Configuration Guide

### Key Configuration Sections

#### 1. Data Configuration

```yaml
data:
  # Data source (plug-and-play)
  audio_path: "/path/to/audio"
  use_huggingface: true
  huggingface_dataset: "dataset/name"
  
  # Audio formats supported
  supported_formats:
    - "wav"
    - "mp3"
    - "flac"
    - "ogg"
    - "m4a"
    - "aac"
    - "wma"
  
  # Preprocessing
  target_sample_rate: 16000
  max_duration_seconds: 30
  min_duration_seconds: 0.5
  
  # Dataset split
  train_split_ratio: 0.9
  validation_split_ratio: 0.1
```

#### 2. Model Configuration

```yaml
model:
  # Base model
  base_model: "openai/whisper-large-v3"
  language: "spanish"
  
  # Distillation (optional)
  use_distillation: true
  teacher_model: "openai/whisper-large-v3"
  distillation_temperature: 2.0
  distillation_alpha: 0.5  # 50% teacher, 50% labels
  
  # QLoRA (4-bit quantization)
  use_4bit_quantization: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  
  # LoRA parameters
  lora_r: 16                # Rank (higher = more capacity)
  lora_alpha: 32
  lora_dropout: 0.1
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "out_proj"
```

#### 3. Training Configuration

```yaml
training:
  num_epochs: 50
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 16
  
  learning_rate: 5e-5
  warmup_steps: 500
  lr_scheduler_type: "cosine"
  
  # Mixed precision
  fp16: true
  
  # Checkpointing
  save_strategy: "epoch"
  save_steps: 5  # Save every 5 epochs
```

#### 4. Export Configuration

```yaml
export:
  export_safetensors: true
  export_onnx: true
  export_pytorch: true
  
  onnx_opset_version: 14
  onnx_optimize: true
```

---

## 📊 Evaluation Metrics

The system computes **real, non-fabricated** metrics:

### WER (Word Error Rate)
- **Lower is better** (0.0 = perfect)
- Measures word-level transcription accuracy
- Standard metric for ASR evaluation

### CER (Character Error Rate)
- **Lower is better** (0.0 = perfect)
- More granular than WER
- Better for languages with complex morphology

### BLEU Score
- **Higher is better** (100 = perfect)
- Originally for machine translation
- Measures n-gram overlap

### chrF Score
- **Higher is better** (100 = perfect)
- Character-level F-score
- Robust to tokenization differences

---

## 💾 Checkpoint Management

### Automatic Checkpointing

Checkpoints are saved every 5 epochs in SafeTensors format:

```
models/checkpoints/
├── checkpoint-5/
│   ├── model.safetensors
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── checkpoint-10/
├── checkpoint-15/
...
└── checkpoint-50/
```

### Resume from Checkpoint

```yaml
checkpoints:
  resume_from_checkpoint: "models/checkpoints/checkpoint-25"
```

### Best Model Tracking

The system automatically keeps the best model based on WER:

```yaml
training:
  load_best_model_at_end: true
  metric_for_best_model: "wer"
  greater_is_better: false
```

---

## 📤 Model Export

After training, models are exported to multiple formats:

### SafeTensors (Recommended)
```
models/final/model_safetensors/
├── model.safetensors
├── config.json
├── preprocessor_config.json
└── MODEL_CARD.md
```

### ONNX (For Production)
```
models/final/model_onnx/
├── encoder.onnx
├── encoder_optimized.onnx
└── preprocessor_config.json
```

### PyTorch (Compatibility)
```
models/final/model_pytorch/
├── pytorch_model.bin
├── config.json
└── preprocessor_config.json
```

---

## 🔌 Plug-and-Play Data Sources

### Method 1: HuggingFace Dataset

```yaml
data:
  use_huggingface: true
  huggingface_dataset: "MrDragonFox/Elise"
```

### Method 2: Local Directory

```
/path/to/audio/
├── audio1.wav
├── audio2.mp3
├── audio3.flac
└── metadata.csv  # Optional
```

```yaml
data:
  use_huggingface: false
  audio_path: "/path/to/audio"
```

### Method 3: CSV File

```csv
audio,text
/path/to/audio1.wav,"Hello world"
/path/to/audio2.mp3,"Testing audio"
```

```yaml
data:
  use_huggingface: false
  audio_path: "/path/to/dataset.csv"
  audio_column: "audio"
  text_column: "text"
```

### Method 4: JSON File

```json
[
  {"audio": "audio1.wav", "text": "Hello world"},
  {"audio": "audio2.mp3", "text": "Testing audio"}
]
```

```yaml
data:
  use_huggingface: false
  audio_path: "/path/to/dataset.json"
```

---

## 🎵 Multi-Format Audio Support

Supported formats:
- **WAV** (PCM, uncompressed)
- **MP3** (MPEG Layer 3)
- **FLAC** (lossless compression)
- **OGG** (Vorbis codec)
- **M4A** (AAC in MP4 container)
- **AAC** (Advanced Audio Codec)
- **WMA** (Windows Media Audio)

The system automatically:
- Detects audio format
- Converts to mono if stereo
- Resamples to 16kHz
- Normalizes amplitude
- Validates duration constraints

---

## 🚢 Deployment Guide

### Using Exported Models

#### SafeTensors (Recommended)

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model
processor = WhisperProcessor.from_pretrained(
    "models/final/model_safetensors"
)
model = WhisperForConditionalGeneration.from_pretrained(
    "models/final/model_safetensors",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Transcribe audio
def transcribe(audio_path):
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    generated_ids = model.generate(
        inputs.input_features.to(model.device)
    )
    
    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
    
    return transcription
```

#### ONNX (Optimized)

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "models/final/model_onnx/encoder.onnx"
)

# Run inference
outputs = session.run(None, {
    'mel_features': mel_features
})
```

---

## 🔬 Advanced Features

### Knowledge Distillation

Train a smaller student model using a larger teacher:

```yaml
model:
  use_distillation: true
  teacher_model: "openai/whisper-large-v3"
  base_model: "openai/whisper-medium"  # Smaller student
  distillation_temperature: 2.0
  distillation_alpha: 0.5
```

### Custom LoRA Configuration

Fine-tune specific model components:

```yaml
model:
  lora_r: 32  # Higher rank = more parameters
  lora_alpha: 64
  lora_target_modules:
    - "q_proj"      # Query projection
    - "v_proj"      # Value projection
    - "k_proj"      # Key projection
    - "out_proj"    # Output projection
    - "fc1"         # Feed-forward layer 1
    - "fc2"         # Feed-forward layer 2
```

### Gradient Checkpointing

Reduce memory usage for larger batches:

```yaml
training:
  gradient_checkpointing: true
  per_device_train_batch_size: 8  # Can use larger batches
```

---

## 📈 Performance Optimization

### GPU Memory Optimization

1. **Use 4-bit Quantization**: Reduces memory by ~75%
2. **Gradient Accumulation**: Train with smaller batches
3. **Gradient Checkpointing**: Trade compute for memory
4. **FP16 Training**: 2x memory reduction

### Training Speed Optimization

1. **Increase Batch Size**: If GPU memory allows
2. **Reduce Evaluation Frequency**: `eval_steps: 200`
3. **Use Faster Data Loading**: `num_workers: 8`
4. **Enable Compiled Models**: PyTorch 2.0+ `torch.compile()`

---

## 🐛 Troubleshooting

### Out of Memory

```yaml
# Reduce batch size
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

# Enable gradient checkpointing
training:
  gradient_checkpointing: true
```

### Audio Loading Errors

```bash
# Install audio backends
pip install soundfile librosa pydub

# For MP3 support
sudo apt-get install ffmpeg
```

### ONNX Export Issues

ONNX export for Whisper is complex. For production:
1. Use SafeTensors format
2. Or use optimum library: `pip install optimum[onnxruntime]`

---

## 📝 Citation

If you use this system in your research:

```bibtex
@software{mwc_asr_2025,
  title={MWC ASR Training System},
  author={Your Name},
  year={2025},
  url={https://github.com/your/repo}
}
```

---

## 📄 License

[Add your license here]

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional audio formats
- More export formats (TensorRT, CoreML)
- Advanced data augmentation
- Multi-GPU training support

---

## 📞 Support

For issues and questions:
- Check logs in `logs/training/`
- Review configuration in `config/training_config.yaml`
- See technical documentation in `docs/TECHNICAL_GUIDE.md`

---

**Built with ❤️ for production ASR training**
