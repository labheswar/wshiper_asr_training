#!/usr/bin/env python3
"""
MWC ASR Training - Production-Grade Whisper Fine-tuning
Features:
- QLoRA with 4-bit quantization
- Knowledge Distillation
- FP16 mixed precision
- Checkpoint saving every 5 epochs (safetensors)
- Real evaluation metrics (WER, CER, BLEU, chrF)
- Multi-format audio support
- Plug-and-play data sources
- ONNX and safetensors export
"""

#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add scripts directory to Python path FIRST
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Now do all other imports
import logging
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
from torch.utils.data import random_split

# Import custom modules (now Python can find them)
import audio_loader
import dataset_handler
from audio_loader import AudioLoader
from dataset_handler import PluggableDatasetHandler, DataSourceConfig


def get_optimal_num_workers():
    """Auto-detect optimal number of CPU cores for data loading"""
    try:
        cpu_count = os.cpu_count() or 4
        # Use 75% of available cores, minimum 2, maximum 16
        optimal = max(2, min(int(cpu_count * 0.75), 16))
        return optimal
    except:
        return 4  # Safe default


def resolve_config_value(value, key_name=""):
    """Resolve 'auto' values in config to actual system values"""
    if isinstance(value, str) and value.lower() == "auto":
        if "worker" in key_name.lower():
            return get_optimal_num_workers()
    return value


# Setup logging
def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper training
    Handles audio features and text labels with proper padding
    """
    processor: Any
    fp16: bool = False  # Track if FP16 training is enabled
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features"""
        processed_features = []
        
        for feature in features:
            try:
                # Extract audio and text
                audio = feature['audio']
                text = feature['text']
                sampling_rate = feature.get('sampling_rate', 16000)
                
                # Convert to numpy if tensor
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                
                # Compute input features from audio
                input_features = self.processor.feature_extractor(
                    audio,
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                ).input_features[0]
                
                # Convert to fp16 if needed for FP16 training
                if self.fp16:
                    input_features = input_features.half()
                
                # Tokenize text to get labels
                labels = self.processor.tokenizer(
                    text,
                    return_tensors="pt"
                ).input_ids[0]
                
                processed_features.append({
                    "input_features": input_features,
                    "labels": labels
                })
                
            except Exception as e:
                # Skip malformed samples
                logging.warning(f"Skipping malformed feature: {e}")
                continue
        
        if not processed_features:
            raise ValueError("No valid features to collate in batch")
        
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in processed_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Ensure input_features are in correct dtype for FP16
        if self.fp16:
            batch["input_features"] = batch["input_features"].half()
        
        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in processed_features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


class DistillationTrainer(Seq2SeqTrainer):
    """
    Custom trainer with knowledge distillation support
    Combines hard labels with soft labels from teacher model
    """
    
    def __init__(
        self,
        teacher_model: Optional[WhisperForConditionalGeneration] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        *args,
        **kwargs
    ):
        """
        Initialize distillation trainer
        
        Args:
            teacher_model: Teacher model for distillation (optional)
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (0.5 = 50% distillation, 50% hard labels)
            *args, **kwargs: Arguments for Seq2SeqTrainer
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        if teacher_model is not None:
            self.teacher_model.eval()
            # Freeze teacher
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            logger.info(f"✓ Distillation enabled (T={temperature}, α={alpha})")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with optional distillation"""
        
        # Standard cross-entropy loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add distillation loss if teacher model is provided
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            
            # Compute distillation loss (KL divergence)
            student_logits = outputs.logits / self.temperature
            teacher_logits = teacher_outputs.logits / self.temperature
            
            distillation_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Combine losses
            loss = self.alpha * distillation_loss + (1 - self.alpha) * loss
        
        return (loss, outputs) if return_outputs else loss


class MWCASRTrainer:
    """Main training class for MWC ASR system"""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Resolve 'auto' values in hardware config
        if 'hardware' in self.config:
            for key, value in self.config['hardware'].items():
                self.config['hardware'][key] = resolve_config_value(value, key)
        
        # Setup paths - use simplified path structure
        if 'paths' in self.config:
            self.project_root = Path(__file__).resolve().parent.parent  # MWC_huggingface root
            self.logs_dir = self.project_root / self.config['paths']['logs_dir']
            self.checkpoints_dir = self.project_root / self.config['paths']['output_dir']
            self.final_models_dir = self.project_root / self.config['paths']['final_model_dir']
        else:
            # Fallback to default paths
            self.project_root = Path(__file__).resolve().parent.parent
            self.logs_dir = self.project_root / "logs" / "training"
            self.checkpoints_dir = self.project_root / "models" / "checkpoints"
            self.final_models_dir = self.project_root / "models" / "final"
        
        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.final_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        global logger
        logger = setup_logging(
            self.logs_dir,
            f"training_run"
        )
        
        logger.info("="*80)
        logger.info("MWC ASR TRAINING SYSTEM")
        logger.info("="*80)
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Project root: {self.project_root}")
        
        # Log hardware configuration
        logger.info("="*80)
        logger.info("HARDWARE CONFIGURATION")
        logger.info("="*80)
        logger.info(f"CPU cores available: {os.cpu_count()}")
        logger.info(f"DataLoader workers: {self.config['hardware'].get('num_workers', 4)}")
        
        # Device setup
        self.device = torch.device(self.config['hardware']['device'])
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            num_gpus = torch.cuda.device_count()
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"Number of GPUs: {num_gpus}")
        else:
            logger.info("Device: CPU")
        
        # Load metrics
        logger.info("="*80)
        logger.info("Loading evaluation metrics...")
        self.metrics = {}
        metrics_config = self.config.get('metrics', {})
        if metrics_config.get('compute_wer', True):
            self.metrics['wer'] = evaluate.load("wer")
        if metrics_config.get('compute_cer', True):
            self.metrics['cer'] = evaluate.load("cer")
        if metrics_config.get('compute_bleu', False):
            self.metrics['bleu'] = evaluate.load("sacrebleu")
        if metrics_config.get('compute_chrf', False):
            self.metrics['chrf'] = evaluate.load("chrf")
        
        logger.info(f"✓ Loaded metrics: {list(self.metrics.keys())}")
        
        # Initialize components
        self.processor = None
        self.model = None
        self.teacher_model = None
        self.dataset_handler = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def load_processor(self):
        """Load Whisper processor"""
        logger.info("Loading Whisper processor...")
        model_name = self.config['model']['base_model']
        self.processor = WhisperProcessor.from_pretrained(model_name)
        logger.info(f"✓ Loaded processor from {model_name}")
    
    def load_model(self):
        """Load and configure Whisper model with QLoRA"""
        logger.info("="*80)
        logger.info("MODEL SETUP")
        logger.info("="*80)
        
        model_name = self.config['model']['base_model']
        logger.info(f"Base model: {model_name}")
        
        # 4-bit quantization config
        if self.config['model']['use_4bit_quantization']:
            logger.info("Configuring 4-bit quantization (QLoRA)...")
            
            compute_dtype = getattr(torch, self.config['model']['bnb_4bit_compute_dtype'])
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config['model']['bnb_4bit_quant_type'],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config['model']['bnb_4bit_use_double_quant']
            )
            
            logger.info(f"  Quant type: {self.config['model']['bnb_4bit_quant_type']}")
            logger.info(f"  Compute dtype: {self.config['model']['bnb_4bit_compute_dtype']}")
            logger.info(f"  Double quant: {self.config['model']['bnb_4bit_use_double_quant']}")
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # CRITICAL: Convert all non-quantized layers to float32 to match input dtype
            # This fixes dtype mismatch when fp16=false
            for param in self.model.parameters():
                if param.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)
            
            logger.info("✓ Model prepared for 4-bit training (non-quantized layers in float32)")
        
        else:
            # Load in FP16
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("✓ Model loaded in FP16")
        
        # Configure LoRA
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=self.config['model']['lora_r'],
            lora_alpha=self.config['model']['lora_alpha'],
            target_modules=self.config['model']['lora_target_modules'],
            lora_dropout=self.config['model']['lora_dropout'],
            bias=self.config['model']['lora_bias']
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.config.use_cache = False
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"  LoRA rank (r): {self.config['model']['lora_r']}")
        logger.info(f"  LoRA alpha: {self.config['model']['lora_alpha']}")
        logger.info(f"  Target modules: {', '.join(self.config['model']['lora_target_modules'])}")
        logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"  Total params: {total_params:,}")
    
    def load_teacher_model(self):
        """Load teacher model for distillation (optional)"""
        if not self.config['model'].get('use_distillation', False):
            logger.info("Distillation disabled")
            return
        
        logger.info("="*80)
        logger.info("TEACHER MODEL SETUP (Distillation)")
        logger.info("="*80)
        
        teacher_name = self.config['model']['teacher_model']
        logger.info(f"Loading teacher model: {teacher_name}")
        
        self.teacher_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Convert teacher model to float32 to match input dtype when fp16=false
        if not self.config['training']['fp16']:
            logger.info("Converting teacher model to float32 to match training dtype...")
            for param in self.teacher_model.parameters():
                if param.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)
        
        self.teacher_model.eval()
        logger.info(f"✓ Teacher model loaded and frozen")
        logger.info(f"  Temperature: {self.config['model']['distillation_temperature']}")
        logger.info(f"  Alpha: {self.config['model']['distillation_alpha']}")
    
    def prepare_dataset(self):
        """Prepare dataset using pluggable data handler"""
        logger.info("="*80)
        logger.info("DATASET PREPARATION")
        logger.info("="*80)
        
        # Create audio loader
        audio_loader = AudioLoader(
            target_sr=self.config['data']['target_sample_rate']
        )
        
        # Create data source config based on source_type
        source_type = self.config['data'].get('source_type', 'local')
        
        logger.info(f"Data source type: {source_type}")
        
        if source_type == "huggingface":
            hf_config = self.config['data'].get('huggingface', {})
            data_config = DataSourceConfig(
                source_type="huggingface",
                dataset_name=hf_config.get('dataset_name', self.config['data'].get('huggingface_dataset')),
                split=hf_config.get('split', self.config['data'].get('huggingface_split', 'train')),
                audio_column=self.config['data'].get('audio_column', 'audio'),
                text_column=hf_config.get('text_column', self.config['data'].get('text_column', 'text'))
            )
        elif source_type == "parquet":
            parquet_config = self.config['data'].get('parquet', {})
            data_config = DataSourceConfig(
                source_type="parquet",
                path=parquet_config.get('file_path'),
                audio_column=parquet_config.get('audio_column', 'audio'),
                audio_path_column=parquet_config.get('audio_path_column', 'audio_path'),
                text_column=parquet_config.get('text_column', 'text')
            )
        elif source_type == "csv":
            csv_config = self.config['data'].get('csv', {})
            data_config = DataSourceConfig(
                source_type="csv",
                path=csv_config.get('file_path'),
                audio_column=csv_config.get('audio_path_column', 'audio_path'),
                text_column=csv_config.get('text_column', 'text')
            )
        elif source_type == "json":
            json_config = self.config['data'].get('json', {})
            data_config = DataSourceConfig(
                source_type="json",
                path=json_config.get('file_path'),
                audio_column=json_config.get('audio_path_column', 'audio_path'),
                text_column=json_config.get('text_column', 'text')
            )
        elif source_type == "local":
            local_config = self.config['data'].get('local', {})
            data_config = DataSourceConfig(
                source_type="local",
                path=local_config.get('audio_path', self.config['data'].get('audio_path')),
                audio_column=self.config['data'].get('audio_column', 'audio'),
                text_column=self.config['data'].get('text_column', 'text')
            )
        else:
            # Fallback to legacy configuration
            if self.config['data'].get('use_huggingface', False):
                data_config = DataSourceConfig(
                    source_type="huggingface",
                    dataset_name=self.config['data']['huggingface_dataset'],
                    split=self.config['data']['huggingface_split'],
                    audio_column=self.config['data']['audio_column'],
                    text_column=self.config['data']['text_column']
                )
            else:
                data_config = DataSourceConfig(
                    source_type="local",
                    path=self.config['data']['audio_path'],
                    audio_column=self.config['data']['audio_column'],
                    text_column=self.config['data']['text_column']
                )
        
        logger.info(f"Loading data from: {data_config.path or data_config.dataset_name}")
        
        # Load dataset
        self.dataset_handler = PluggableDatasetHandler(
            config=data_config,
            audio_loader=audio_loader,
            processor=self.processor,
            max_duration=self.config['data']['max_duration_seconds'],
            min_duration=self.config['data']['min_duration_seconds']
        )
        
        full_dataset = self.dataset_handler.load_dataset()
        logger.info(f"✓ Loaded {len(full_dataset)} total samples")
        
        # Split into train/eval
        train_size = int(self.config['data']['train_split_ratio'] * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(self.config['data']['random_seed'])
        )
        
        logger.info(f"  Train samples: {len(self.train_dataset)}")
        logger.info(f"  Eval samples: {len(self.eval_dataset)}")
    
    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute metrics
        results = {}
        
        if 'wer' in self.metrics:
            wer = self.metrics['wer'].compute(predictions=pred_str, references=label_str)
            results['wer'] = wer
        
        if 'cer' in self.metrics:
            cer = self.metrics['cer'].compute(predictions=pred_str, references=label_str)
            results['cer'] = cer
        
        if 'bleu' in self.metrics:
            bleu = self.metrics['bleu'].compute(
                predictions=pred_str,
                references=[[ref] for ref in label_str]
            )
            results['bleu'] = bleu['score']
        
        if 'chrf' in self.metrics:
            chrf = self.metrics['chrf'].compute(
                predictions=pred_str,
                references=label_str
            )
            results['chrf'] = chrf['score']
        
        # Log some predictions
        if self.config['metrics'].get('log_predictions', False):
            num_to_log = min(
                self.config['metrics'].get('num_predictions_to_log', 5),
                len(pred_str)
            )
            logger.info("\n" + "="*80)
            logger.info("SAMPLE PREDICTIONS")
            logger.info("="*80)
            for i in range(num_to_log):
                logger.info(f"\nSample {i+1}:")
                logger.info(f"  Reference: {label_str[i]}")
                logger.info(f"  Predicted: {pred_str[i]}")
        
        return results
    
    def train(self):
        """Execute training"""
        logger.info("="*80)
        logger.info("TRAINING CONFIGURATION")
        logger.info("="*80)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.checkpoints_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            warmup_steps=self.config['training']['warmup_steps'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            weight_decay=self.config['training']['weight_decay'],
            optim=self.config['training']['optimizer'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            fp16=self.config['training']['fp16'],
            bf16=self.config['training']['bf16'],
            eval_strategy=self.config['training']['eval_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            logging_steps=self.config['training']['logging_steps'],
            logging_first_step=self.config['training']['logging_first_step'],
            report_to=self.config['training']['report_to'],
            predict_with_generate=self.config['training']['predict_with_generate'],
            generation_max_length=self.config['training']['generation_max_length'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            seed=self.config['training']['seed'],
            dataloader_num_workers=self.config['hardware']['num_workers'],
            dataloader_pin_memory=self.config['hardware']['pin_memory'],
            remove_unused_columns=False,
            # Save in safetensors format
            save_safetensors=True,
        )
        
        logger.info(f"  Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size: {self.config['training']['per_device_train_batch_size']}")
        logger.info(f"  Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  Effective batch size: {self.config['training']['per_device_train_batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            fp16=False  # Disabled for 4-bit quantization
        )
        
        # Create trainer
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING TRAINER")
        logger.info("="*80)
        
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=self.config['model'].get('distillation_temperature', 2.0),
            alpha=self.config['model'].get('distillation_alpha', 0.5),
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.tokenizer,
        )
        
        logger.info("✓ Trainer initialized")
        
        # Train!
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING TRAINING")
        logger.info("="*80)
        
        train_start = datetime.now()
        
        train_result = trainer.train()
        
        train_end = datetime.now()
        train_duration = (train_end - train_start).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Training duration: {train_duration / 3600:.2f} hours")
        
        # Save final model
        logger.info("\nSaving final model...")
        final_model_path = self.final_models_dir / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(final_model_path))
        trainer.save_state()
        
        # Save training results
        results = {
            "model": self.config['model']['base_model'],
            "training_duration_hours": train_duration / 3600,
            "metrics": train_result.metrics,
            "training_date": datetime.now().isoformat(),
            "configuration": self.config
        }
        
        results_file = self.logs_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Results saved to: {results_file}")
        
        return trainer, results
    
    def run(self):
        """Execute full training pipeline"""
        try:
            # Load all components
            self.load_processor()
            self.load_model()
            self.load_teacher_model()
            self.prepare_dataset()
            
            # Train
            trainer, results = self.train()
            
            # Export models
            logger.info("\n" + "="*80)
            logger.info("EXPORTING MODELS")
            logger.info("="*80)
            
            from model_exporter import ModelExporter
            exporter = ModelExporter(self.config, self.processor)
            
            final_model_path = self.final_models_dir / "final_model"
            exporter.export_all(str(final_model_path))
            
            logger.info("\n" + "="*80)
            logger.info("🎉 ALL DONE!")
            logger.info("="*80)
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MWC ASR Training System")
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Run training
    trainer = MWCASRTrainer(config_path=args.config)
    results = trainer.run()
