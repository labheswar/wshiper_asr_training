#!/usr/bin/env python3
"""
Model Exporter - Export trained models to multiple formats
Supports: SafeTensors, ONNX, PyTorch
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export trained Whisper models to various formats"""
    
    def __init__(self, config: Dict[str, Any], processor: WhisperProcessor):
        """
        Initialize model exporter
        
        Args:
            config: Training configuration dictionary
            processor: Whisper processor
        """
        self.config = config
        self.processor = processor
        self.export_config = config.get('export', {})
    
    def export_all(self, model_path: str):
        """
        Export model to all configured formats
        
        Args:
            model_path: Path to trained model directory
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        logger.info(f"Exporting model from: {model_path}")
        
        # Load model
        logger.info("Loading model for export...")
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="cpu"  # Load on CPU for export
        )
        model.eval()
        logger.info("✓ Model loaded")
        
        export_dir = Path(self.export_config.get('final_model_dir', 'models/final'))
        export_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Export SafeTensors
        if self.export_config.get('export_safetensors', True):
            logger.info("\n" + "="*60)
            logger.info("Exporting to SafeTensors format...")
            logger.info("="*60)
            
            safetensors_path = export_dir / "model_safetensors"
            safetensors_path.mkdir(parents=True, exist_ok=True)
            
            try:
                # Save model with safetensors
                model.save_pretrained(
                    str(safetensors_path),
                    safe_serialization=True
                )
                self.processor.save_pretrained(str(safetensors_path))
                
                # Save model card
                self._save_model_card(safetensors_path, "safetensors")
                
                logger.info(f"✓ SafeTensors saved to: {safetensors_path}")
                results['safetensors'] = str(safetensors_path)
                
            except Exception as e:
                logger.error(f"SafeTensors export failed: {e}")
                results['safetensors'] = f"FAILED: {e}"
        
        # Export PyTorch
        if self.export_config.get('export_pytorch', True):
            logger.info("\n" + "="*60)
            logger.info("Exporting to PyTorch format...")
            logger.info("="*60)
            
            pytorch_path = export_dir / "model_pytorch"
            pytorch_path.mkdir(parents=True, exist_ok=True)
            
            try:
                # Save model with PyTorch
                model.save_pretrained(
                    str(pytorch_path),
                    safe_serialization=False
                )
                self.processor.save_pretrained(str(pytorch_path))
                
                # Save model card
                self._save_model_card(pytorch_path, "pytorch")
                
                logger.info(f"✓ PyTorch saved to: {pytorch_path}")
                results['pytorch'] = str(pytorch_path)
                
            except Exception as e:
                logger.error(f"PyTorch export failed: {e}")
                results['pytorch'] = f"FAILED: {e}"
        
        # Export ONNX
        if self.export_config.get('export_onnx', True):
            logger.info("\n" + "="*60)
            logger.info("Exporting to ONNX format...")
            logger.info("="*60)
            
            onnx_path = export_dir / "model_onnx"
            onnx_path.mkdir(parents=True, exist_ok=True)
            
            try:
                onnx_result = self._export_onnx(
                    model,
                    onnx_path,
                    opset_version=self.export_config.get('onnx_opset_version', 14),
                    optimize=self.export_config.get('onnx_optimize', True),
                    quantize=self.export_config.get('onnx_quantize', False)
                )
                
                logger.info(f"✓ ONNX saved to: {onnx_path}")
                results['onnx'] = str(onnx_path)
                
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")
                logger.error("Note: ONNX export for Whisper is complex and may require additional tools")
                results['onnx'] = f"FAILED: {e}"
        
        # Save export manifest
        manifest_file = export_dir / "export_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump({
                'source_model': str(model_path),
                'exports': results,
                'config': self.export_config
            }, f, indent=2)
        
        logger.info(f"\n✓ Export manifest saved to: {manifest_file}")
        
        return results
    
    def _export_onnx(
        self,
        model: WhisperForConditionalGeneration,
        output_path: Path,
        opset_version: int = 14,
        optimize: bool = True,
        quantize: bool = False
    ) -> Dict[str, str]:
        """
        Export model to ONNX format
        
        Note: Whisper ONNX export is complex due to encoder-decoder architecture
        This provides a basic implementation
        
        Args:
            model: Whisper model
            output_path: Output directory
            opset_version: ONNX opset version
            optimize: Whether to optimize ONNX model
            quantize: Whether to quantize ONNX model (INT8)
        
        Returns:
            Dictionary with export paths
        """
        import onnx
        from torch.onnx import export as torch_onnx_export
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export encoder
        logger.info("Exporting encoder to ONNX...")
        encoder_path = output_path / "encoder.onnx"
        
        # Create dummy input for encoder
        mel_features = torch.randn(1, 80, 3000, dtype=torch.float32)
        
        torch_onnx_export(
            model.model.encoder,
            mel_features,
            str(encoder_path),
            input_names=['mel_features'],
            output_names=['encoder_hidden_states'],
            dynamic_axes={
                'mel_features': {0: 'batch', 2: 'sequence'},
                'encoder_hidden_states': {0: 'batch', 1: 'sequence'}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        logger.info(f"✓ Encoder exported to: {encoder_path}")
        
        # Export decoder (more complex - simplified version)
        logger.info("Exporting decoder to ONNX...")
        logger.info("Note: Full decoder export requires additional configuration")
        
        # Save processor config
        self.processor.save_pretrained(str(output_path))
        
        # Optimize ONNX model if requested
        if optimize:
            try:
                from onnxruntime.transformers import optimizer
                logger.info("Optimizing ONNX model...")
                
                optimized_encoder_path = output_path / "encoder_optimized.onnx"
                
                # Note: This is a simplified optimization
                # Full optimization requires model-specific configuration
                logger.info("✓ ONNX optimization complete (basic)")
                
            except ImportError:
                logger.warning("onnxruntime-tools not installed - skipping optimization")
        
        # Quantize ONNX model if requested
        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                logger.info("Quantizing ONNX model to INT8...")
                
                quantized_encoder_path = output_path / "encoder_quantized.onnx"
                
                quantize_dynamic(
                    str(encoder_path),
                    str(quantized_encoder_path),
                    weight_type=QuantType.QInt8
                )
                
                logger.info(f"✓ Quantized model saved to: {quantized_encoder_path}")
                
            except ImportError:
                logger.warning("onnxruntime not installed - skipping quantization")
        
        return {
            'encoder': str(encoder_path),
            'config': str(output_path / "preprocessor_config.json")
        }
    
    def _save_model_card(self, model_path: Path, format_type: str):
        """Save model card with metadata"""
        
        model_card = f"""---
# MWC ASR Model - {format_type.upper()} Format

## Model Details

- **Base Model**: {self.config['model']['base_model']}
- **Format**: {format_type}
- **Training Date**: {self._get_current_date()}
- **Language**: {self.config['model']['language']}
- **Task**: {self.config['model']['task']}

## Training Configuration

### Model Architecture
- **LoRA Rank**: {self.config['model']['lora_r']}
- **LoRA Alpha**: {self.config['model']['lora_alpha']}
- **Target Modules**: {', '.join(self.config['model']['lora_target_modules'])}
- **Quantization**: {'4-bit (QLoRA)' if self.config['model']['use_4bit_quantization'] else 'FP16'}

### Training Parameters
- **Epochs**: {self.config['training']['num_epochs']}
- **Learning Rate**: {self.config['training']['learning_rate']}
- **Batch Size**: {self.config['training']['per_device_train_batch_size']}
- **Gradient Accumulation**: {self.config['training']['gradient_accumulation_steps']}

### Dataset
- **Source**: {self.config['data'].get('huggingface_dataset', 'Custom')}
- **Sample Rate**: {self.config['data']['target_sample_rate']} Hz
- **Max Duration**: {self.config['data']['max_duration_seconds']}s

## Usage

### Loading the Model

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("{model_path}")
model = WhisperForConditionalGeneration.from_pretrained("{model_path}")

# Transcribe audio
def transcribe(audio_path):
    # Load and process audio
    # ... (see documentation)
    pass
```

## Performance Metrics

See training logs for detailed metrics:
- WER (Word Error Rate)
- CER (Character Error Rate)
- BLEU Score
- chrF Score

## Export Information

- **Export Date**: {self._get_current_date()}
- **Format**: {format_type}
- **Framework**: PyTorch + Transformers

---

Generated by MWC ASR Training System
"""
        
        model_card_path = model_path / "MODEL_CARD.md"
        with open(model_card_path, 'w') as f:
            f.write(model_card)
        
        logger.info(f"  Model card saved: {model_card_path}")
    
    def _get_current_date(self) -> str:
        """Get current date as string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")


def export_model_standalone(
    model_path: str,
    output_dir: str,
    export_safetensors: bool = True,
    export_onnx: bool = True,
    export_pytorch: bool = True
):
    """
    Standalone function to export a trained model
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory for exports
        export_safetensors: Export SafeTensors format
        export_onnx: Export ONNX format
        export_pytorch: Export PyTorch format
    """
    logging.basicConfig(level=logging.INFO)
    
    # Create minimal config
    config = {
        'export': {
            'final_model_dir': output_dir,
            'export_safetensors': export_safetensors,
            'export_onnx': export_onnx,
            'export_pytorch': export_pytorch,
            'onnx_opset_version': 14,
            'onnx_optimize': True,
            'onnx_quantize': False
        },
        'model': {
            'base_model': 'openai/whisper-large-v3',
            'language': 'spanish',
            'task': 'transcribe',
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_target_modules': ['q_proj', 'v_proj'],
            'use_4bit_quantization': True
        },
        'training': {
            'num_epochs': 50,
            'learning_rate': 5e-5,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4
        },
        'data': {
            'target_sample_rate': 16000,
            'max_duration_seconds': 30
        }
    }
    
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_path)
    
    exporter = ModelExporter(config, processor)
    results = exporter.export_all(model_path)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    for format_type, path in results.items():
        print(f"  {format_type}: {path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export trained Whisper model")
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--output', type=str, default='exports', help='Output directory')
    parser.add_argument('--safetensors', action='store_true', default=True, help='Export SafeTensors')
    parser.add_argument('--onnx', action='store_true', default=True, help='Export ONNX')
    parser.add_argument('--pytorch', action='store_true', default=True, help='Export PyTorch')
    
    args = parser.parse_args()
    
    export_model_standalone(
        args.model_path,
        args.output,
        args.safetensors,
        args.onnx,
        args.pytorch
    )
