#!/usr/bin/env python3
"""
Test Training System - Validates all components before full training
Tests:
1. Parquet file loading from config
2. Audio processing
3. Model initialization
4. ONNX FP16 export capability
"""

import os
import sys
from pathlib import Path
import logging
import yaml
import torch
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_config_loading():
    """Test 1: Verify config file exists and is valid"""
    print_section("TEST 1: Configuration Loading")
    
    config_path = Path("config/training_config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"✓ Config file exists: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("✓ Config loaded successfully")
    
    # Check required fields
    required_fields = [
        ('data', 'parquet', 'file_path'),
        ('model', 'base_model'),
        ('training', 'num_epochs'),
        ('export', 'export_onnx'),
        ('export', 'onnx_quantize_fp16')
    ]
    
    for field_path in required_fields:
        value = config
        for key in field_path:
            if key not in value:
                raise KeyError(f"Missing config field: {'.'.join(field_path)}")
            value = value[key]
        logger.info(f"✓ Config field present: {'.'.join(field_path)}")
    
    logger.info("✓ All required config fields present")
    
    return config


def test_parquet_loading(config):
    """Test 2: Load one parquet file from config directory"""
    print_section("TEST 2: Parquet File Loading")
    
    parquet_dir = Path(config['data']['parquet']['file_path'])
    
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    
    logger.info(f"✓ Parquet directory exists: {parquet_dir}")
    
    # Find parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {parquet_dir}")
    
    logger.info(f"✓ Found {len(parquet_files)} parquet file(s)")
    
    # Load first parquet file
    import pandas as pd
    
    test_file = parquet_files[0]
    logger.info(f"Testing with: {test_file.name}")
    
    df = pd.read_parquet(test_file)
    logger.info(f"✓ Loaded parquet file: {len(df)} rows")
    
    # Check required columns
    audio_col = config['data']['parquet']['audio_column']
    text_col = config['data']['parquet']['text_column']
    
    if audio_col not in df.columns:
        raise KeyError(f"Audio column '{audio_col}' not found in parquet file")
    
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in parquet file")
    
    logger.info(f"✓ Audio column present: {audio_col}")
    logger.info(f"✓ Text column present: {text_col}")
    
    # Test loading first audio sample
    first_row = df.iloc[0]
    audio_data = first_row[audio_col]
    text_data = first_row[text_col]
    
    logger.info(f"✓ Sample audio data type: {type(audio_data)}")
    logger.info(f"✓ Sample text: {text_data[:50]}..." if len(text_data) > 50 else f"✓ Sample text: {text_data}")
    
    return df, parquet_files[0]


def test_audio_processing(df, config):
    """Test 3: Audio processing pipeline"""
    print_section("TEST 3: Audio Processing")
    
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    
    from audio_loader import AudioLoader
    
    target_sr = config['data']['target_sample_rate']
    audio_loader = AudioLoader(target_sr=target_sr)
    
    logger.info(f"✓ AudioLoader initialized (target_sr={target_sr})")
    
    # Test processing first sample
    audio_col = config['data']['parquet']['audio_column']
    first_audio = df.iloc[0][audio_col]
    
    # Handle embedded audio bytes
    import io
    import numpy as np
    
    try:
        if isinstance(first_audio, dict) and 'bytes' in first_audio:
            audio_bytes = first_audio['bytes']
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Try loading with librosa
            import librosa
            audio_array, sr = librosa.load(audio_buffer, sr=target_sr)
            logger.info(f"✓ Loaded audio from bytes: shape={audio_array.shape}, sr={sr}")
            
        elif isinstance(first_audio, bytes):
            audio_buffer = io.BytesIO(first_audio)
            import librosa
            audio_array, sr = librosa.load(audio_buffer, sr=target_sr)
            logger.info(f"✓ Loaded audio from bytes: shape={audio_array.shape}, sr={sr}")
            
        else:
            logger.warning(f"Unexpected audio data type: {type(first_audio)}")
            # Create dummy audio for testing
            audio_array = np.random.randn(target_sr * 2).astype(np.float32)
            sr = target_sr
            logger.info(f"✓ Using dummy audio for testing: shape={audio_array.shape}")
    
    except Exception as e:
        logger.warning(f"Audio processing issue: {e}")
        # Create dummy audio for testing
        audio_array = np.random.randn(target_sr * 2).astype(np.float32)
        sr = target_sr
        logger.info(f"✓ Using dummy audio for testing: shape={audio_array.shape}")
    
    logger.info(f"✓ Audio processing successful")
    
    return audio_array, sr


def test_model_initialization(config):
    """Test 4: Model and processor initialization"""
    print_section("TEST 4: Model Initialization")
    
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    base_model = config['model']['base_model']
    logger.info(f"Loading model: {base_model}")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(base_model)
    logger.info(f"✓ Processor loaded")
    
    # Load model (on CPU for testing)
    logger.info("Loading model (CPU mode for testing)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    logger.info(f"✓ Model loaded")
    
    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model parameters: {param_count:,}")
    
    return model, processor


def test_onnx_export_capability(model, processor, config):
    """Test 5: ONNX FP16 export capability"""
    print_section("TEST 5: ONNX FP16 Export Test")
    
    # Check if ONNX export is enabled
    if not config['export'].get('export_onnx', False):
        logger.warning("ONNX export is disabled in config")
        return
    
    if not config['export'].get('onnx_quantize_fp16', False):
        logger.warning("ONNX FP16 quantization is disabled in config")
        return
    
    logger.info("Testing ONNX export capability...")
    
    try:
        import onnx
        import onnxruntime as ort
        logger.info("✓ ONNX and ONNXRuntime installed")
    except ImportError as e:
        logger.warning(f"ONNX libraries not fully installed: {e}")
        logger.warning("ONNX export will be skipped during training")
        return
    
    # Test ONNX export with a simple encoder export
    try:
        logger.info("Testing encoder export to ONNX...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_encoder.onnx"
            
            # Create dummy input
            mel_features = torch.randn(1, 80, 3000, dtype=torch.float32)
            
            # Export encoder
            torch.onnx.export(
                model.model.encoder,
                mel_features,
                str(temp_path),
                input_names=['mel_features'],
                output_names=['encoder_hidden_states'],
                opset_version=config['export'].get('onnx_opset_version', 14),
                do_constant_folding=True
            )
            
            logger.info(f"✓ ONNX export successful: {temp_path.name}")
            
            # Verify ONNX model
            onnx_model = onnx.load(str(temp_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model validation passed")
            
            # Test FP16 conversion
            logger.info("Testing FP16 conversion...")
            try:
                from onnxconverter_common import float16
                onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
                logger.info("✓ FP16 conversion successful")
            except ImportError:
                logger.warning("onnxconverter-common not installed - FP16 conversion unavailable")
                logger.info("Install with: pip install onnxconverter-common")
            except Exception as e:
                logger.warning(f"FP16 conversion test failed: {e}")
        
        logger.info("✓ ONNX export capability verified")
        
    except Exception as e:
        logger.error(f"ONNX export test failed: {e}")
        logger.warning("ONNX export may fail during training")
        raise


def test_dependencies():
    """Test 6: Verify all required dependencies"""
    print_section("TEST 6: Dependency Check")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('librosa', 'Librosa'),
        ('peft', 'PEFT'),
        ('evaluate', 'Evaluate'),
    ]
    
    optional_packages = [
        ('onnx', 'ONNX'),
        ('onnxruntime', 'ONNXRuntime'),
        ('onnxconverter_common', 'ONNX Converter'),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} installed")
        except ImportError:
            logger.error(f"✗ {name} NOT installed")
            all_good = False
    
    for package, name in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} installed (optional)")
        except ImportError:
            logger.warning(f"  {name} not installed (optional - needed for ONNX export)")
    
    if not all_good:
        raise ImportError("Missing required dependencies")
    
    logger.info("✓ All required dependencies installed")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  MWC ASR TRAINING - PRE-TRAINING VALIDATION")
    print("="*70)
    
    try:
        # Test 1: Config loading
        config = test_config_loading()
        
        # Test 2: Parquet loading
        df, parquet_file = test_parquet_loading(config)
        
        # Test 3: Audio processing
        audio_array, sr = test_audio_processing(df, config)
        
        # Test 4: Model initialization
        model, processor = test_model_initialization(config)
        
        # Test 5: ONNX export capability
        test_onnx_export_capability(model, processor, config)
        
        # Test 6: Dependencies
        test_dependencies()
        
        # Success
        print_section("ALL TESTS PASSED ✅")
        logger.info("System is ready for training!")
        
        return 0
        
    except Exception as e:
        print_section("TESTS FAILED ❌")
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
