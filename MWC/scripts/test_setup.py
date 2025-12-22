#!/usr/bin/env python3
"""
Example script to test the MWC ASR training system
Run this to verify everything is set up correctly
"""

import os
import sys
import logging
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required libraries are installed"""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    imports = {
        'Core': [
            ('torch', 'PyTorch'),
            ('transformers', 'Transformers'),
            ('datasets', 'Datasets'),
            ('peft', 'PEFT'),
            ('bitsandbytes', 'BitsAndBytes'),
        ],
        'Audio': [
            ('soundfile', 'SoundFile'),
            ('librosa', 'Librosa'),
        ],
        'Evaluation': [
            ('evaluate', 'Evaluate'),
            ('jiwer', 'Jiwer'),
        ],
        'Utilities': [
            ('yaml', 'PyYAML'),
            ('pandas', 'Pandas'),
            ('numpy', 'NumPy'),
        ]
    }
    
    all_passed = True
    
    for category, libs in imports.items():
        print(f"\n{category}:")
        for module, name in libs:
            try:
                __import__(module)
                print(f"  ✅ {name}")
            except ImportError as e:
                print(f"  ❌ {name} - FAILED ({e})")
                all_passed = False
    
    return all_passed


def test_gpu():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("TESTING GPU")
    print("="*60)
    
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA available")
        print(f"   Device: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available - will use CPU (slow)")
        return False


def test_audio_loader():
    """Test audio loader"""
    print("\n" + "="*60)
    print("TESTING AUDIO LOADER")
    print("="*60)
    
    try:
        from audio_loader import AudioLoader
        
        loader = AudioLoader(target_sample_rate=16000)
        print(f"✅ AudioLoader initialized")
        print(f"   Supported formats: {', '.join(loader.SUPPORTED_FORMATS)}")
        print(f"   Has soundfile: {loader.has_soundfile}")
        print(f"   Has librosa: {loader.has_librosa}")
        print(f"   Has pydub: {loader.has_pydub}")
        return True
        
    except Exception as e:
        print(f"❌ AudioLoader failed: {e}")
        return False


def test_dataset_handler():
    """Test dataset handler"""
    print("\n" + "="*60)
    print("TESTING DATASET HANDLER")
    print("="*60)
    
    try:
        from dataset_handler import PluggableDatasetHandler, DataSourceConfig
        print("✅ Dataset handler imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Dataset handler failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    try:
        import yaml
        
        config_path = script_dir.parent / "config" / "training_config.yaml"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded")
        print(f"   Base model: {config['model']['base_model']}")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Batch size: {config['training']['per_device_train_batch_size']}")
        print(f"   LoRA rank: {config['model']['lora_r']}")
        print(f"   Use 4-bit: {config['model']['use_4bit_quantization']}")
        print(f"   Use distillation: {config['model'].get('use_distillation', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_model_loading():
    """Test model loading (optional, requires download)"""
    print("\n" + "="*60)
    print("TESTING MODEL LOADING (Optional)")
    print("="*60)
    
    try:
        from transformers import WhisperProcessor
        
        print("Testing processor loading...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        print("✅ Model processor loaded (whisper-tiny)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Model loading skipped: {e}")
        print("   (This is OK - models will be downloaded during training)")
        return True  # Don't fail on this


def test_directory_structure():
    """Test that required directories exist"""
    print("\n" + "="*60)
    print("TESTING DIRECTORY STRUCTURE")
    print("="*60)
    
    base_dir = script_dir.parent
    
    required_dirs = [
        "config",
        "scripts",
        "models",
        "models/checkpoints",
        "models/final",
        "logs",
        "logs/training",
        "logs/evaluation",
        "docs",
        "data",
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}")
        else:
            print(f"  ❌ {dir_name} - MISSING")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MWC ASR TRAINING SYSTEM - VERIFICATION")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("GPU", test_gpu),
        ("Audio Loader", test_audio_loader),
        ("Dataset Handler", test_dataset_handler),
        ("Configuration", test_config),
        ("Directory Structure", test_directory_structure),
        ("Model Loading", test_model_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nSystem is ready for training!")
        print("\nNext steps:")
        print("1. Review config/training_config.yaml")
        print("2. Prepare your dataset")
        print("3. Run: python train_asr.py --config ../config/training_config.yaml")
        print("\nFor help: See docs/QUICKSTART.md")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        print("See docs/README.md for troubleshooting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
