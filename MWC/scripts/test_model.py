#!/usr/bin/env python3
"""
Quick test script to verify your trained MWC ASR model works correctly
Usage: python test_model.py [audio_file]
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(model_path: str, audio_file: str = None):
    """Test a trained ASR model"""
    
    print("="*60)
    print("MWC ASR MODEL TESTING")
    print("="*60)
    print()
    
    # Import libraries
    print("Loading libraries...")
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install torch transformers librosa")
        return False
    
    # Check model path
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"✅ Model path: {model_path}")
    
    # Load model
    print("\nLoading model...")
    try:
        processor = WhisperProcessor.from_pretrained(str(model_path))
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {model.device}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test with audio file
    if audio_file:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            return False
        
        print(f"\nTesting with audio: {audio_path}")
        
        try:
            # Load audio
            print("Loading audio...")
            audio, sr = librosa.load(str(audio_path), sr=16000)
            duration = len(audio) / sr
            print(f"✅ Audio loaded: {duration:.2f}s, {sr}Hz")
            
            # Transcribe
            print("Transcribing...")
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(inputs.input_features)
            
            # Decode
            transcription = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            print("\n" + "="*60)
            print("TRANSCRIPTION RESULT")
            print("="*60)
            print(f"{transcription}")
            print("="*60)
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    else:
        print("\n✅ Model test passed (no audio file provided)")
        print("\nTo test with audio:")
        print(f"  python {sys.argv[0]} <audio_file>")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test trained MWC ASR model"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Audio file to transcribe (optional)"
    )
    parser.add_argument(
        "--model",
        default="../models/final/model_safetensors",
        help="Path to model directory (default: ../models/final/model_safetensors)"
    )
    
    args = parser.parse_args()
    
    success = test_model(args.model, args.audio_file)
    
    if success:
        print("\n✅ Test completed successfully!")
        return 0
    else:
        print("\n❌ Test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
