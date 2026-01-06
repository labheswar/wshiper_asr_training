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

# Limit threading in libraries to prevent thread exhaustion with multiple workers
# Set these BEFORE importing any other libraries
os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # NumExpr threads

# Add scripts directory to Python path FIRST
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# MUST be done before importing torch or any other libraries
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Now do all other imports
import logging
import yaml
import json
import torch
import torch.distributed as dist
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import threading
from contextlib import contextmanager

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
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


# ============================================================================
# TIMING AND PERFORMANCE MONITORING
# ============================================================================

class TimingContext:
    """Context manager for timing code blocks"""
    def __init__(self, logger, name, log_level=logging.INFO):
        self.logger = logger
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        self.logger.log(self.log_level, f"⏱️  {self.name}: {elapsed:.4f}s")
        return False
    
    @property
    def elapsed(self):
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None


@contextmanager
def timer(logger, name, log_level=logging.INFO):
    """Simple timing context manager"""
    start = time.perf_counter()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        logger.log(log_level, f"⏱️  {name}: {elapsed:.4f}s")


class PerformanceMonitor:
    """Monitor training performance metrics"""
    def __init__(self, logger):
        self.logger = logger
        self.timings = {}
        self.counters = {}
        self.last_gpu_check = 0
    
    def add_timing(self, name: str, duration: float):
        """Add a timing measurement"""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def log_summary(self):
        """Log performance summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 PERFORMANCE SUMMARY")
        self.logger.info("="*80)
        
        for name, times in sorted(self.timings.items()):
            if times:
                avg = sum(times) / len(times)
                total = sum(times)
                min_t = min(times)
                max_t = max(times)
                self.logger.info(f"{name}:")
                self.logger.info(f"  Avg: {avg:.4f}s | Total: {total:.2f}s | Min: {min_t:.4f}s | Max: {max_t:.4f}s | Calls: {len(times)}")
        
        if self.counters:
            self.logger.info("\nCounters:")
            for name, value in sorted(self.counters.items()):
                self.logger.info(f"  {name}: {value}")
    
    def log_gpu_memory(self, prefix=""):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            current_time = time.time()
            # Throttle GPU memory checks (expensive operation)
            if current_time - self.last_gpu_check < 1.0:  # Max once per second
                return
            self.last_gpu_check = current_time
            
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            self.logger.info(f"🎮 GPU Memory {prefix}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")


# Global performance monitor
perf_monitor = None


# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

def setup_distributed():
    """
    Setup distributed training for multi-GPU training with torchrun.
    Detects and configures:
    - LOCAL_RANK: GPU device for this process
    - RANK: Global rank across all nodes
    - WORLD_SIZE: Total number of processes
    
    Returns:
        dict with distributed info (rank, local_rank, world_size, is_distributed)
    """
    # Check if running with torchrun/distributed launcher
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    is_distributed = local_rank != -1
    
    dist_info = {
        'is_distributed': is_distributed,
        'local_rank': local_rank if is_distributed else 0,
        'rank': rank if is_distributed else 0,
        'world_size': world_size,
        'is_main_process': (rank == 0 if is_distributed else True)
    }
    
    if is_distributed:
        # Initialize process group
        if not dist.is_initialized():
            # Set device BEFORE init_process_group to avoid CUDA errors
            torch.cuda.set_device(local_rank)
            
            # Initialize distributed backend
            dist.init_process_group(
                backend='nccl',  # NVIDIA GPUs
                init_method='env://',  # Use environment variables
            )
            
            print(f"🌐 [Rank {rank}] Distributed Training Initialized:")
            print(f"   Rank: {rank}/{world_size}")
            print(f"   Local Rank (GPU): {local_rank}")
            print(f"   Device: cuda:{local_rank}")
            print(f"   Master: {rank == 0}")
        
        # Synchronize all processes
        dist.barrier()
    
    return dist_info


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


# Setup logging
def setup_logging(log_dir: Path, run_name: str, is_main_process: bool = True) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Only main process logs to file, all processes log to stdout with rank prefix
    handlers = []
    
    if is_main_process:
        log_file = log_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    
    handlers.append(logging.StreamHandler(sys.stdout))
    
    # Get rank for logging prefix
    rank = int(os.environ.get('RANK', 0))
    log_format = f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    if is_main_process:
        logger.info(f"Logging to: {log_file}")
    return logger


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    GPU-Optimized Data collator for Whisper training
    - Uses pinned memory for faster CPU->GPU transfer
    - Detailed timing logs
    - Pre-allocates tensors on GPU when possible
    """
    processor: Any
    fp16: bool = False
    device: str = "cuda"  # Target device for tensors
    use_pinned_memory: bool = True  # Use pinned memory for faster transfer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features with GPU optimization"""
        global perf_monitor
        
        batch_start = time.perf_counter()
        processed_features = []
        
        # PHASE 1: Audio processing (CPU-bound)
        audio_processing_start = time.perf_counter()
        
        for idx, feature in enumerate(features):
            try:
                # Extract audio and text
                audio = feature['audio']
                text = feature['text']
                sampling_rate = feature.get('sampling_rate', 16000)
                
                # Ensure text is a string (not list or other type)
                if isinstance(text, list):
                    if len(text) > 0:
                        text = text[0]  # Take first element if it's a list
                    else:
                        raise ValueError("Text is an empty list")
                elif not isinstance(text, str):
                    text = str(text)  # Convert to string if it's something else
                
                # Handle GPU tensors (already on GPU from dataset)
                if isinstance(audio, torch.Tensor):
                    # Move to CPU for feature extraction (Whisper processor expects numpy/CPU tensors)
                    if audio.is_cuda:
                        audio = audio.cpu().numpy()
                    else:
                        audio = audio.numpy()
                elif not isinstance(audio, np.ndarray):
                    # If not tensor or numpy array, something is wrong
                    raise TypeError(f"Audio data must be tensor or numpy array, got {type(audio)}")
                
                # Compute input features from audio
                input_features = self.processor.feature_extractor(
                    audio,
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                ).input_features[0]
                
                # Convert to fp16 if needed
                if self.fp16:
                    input_features = input_features.half()
                
                # Tokenize text to get labels
                # Don't use return_tensors="pt" here to avoid nesting issues
                tokenized = self.processor.tokenizer(
                    text,
                    padding=False,  # Don't pad here, will pad in batch
                    truncation=True,  # Truncate if text is too long
                    max_length=448,  # Whisper's max text length
                    add_special_tokens=True
                )
                # Convert to tensor manually
                labels = torch.tensor(tokenized['input_ids'])
                
                processed_features.append({
                    "input_features": input_features,
                    "labels": labels
                })
                
            except Exception as e:
                # Log detailed error for debugging
                logging.error(f"Skipping sample {idx} in batch: {type(e).__name__}: {str(e)}")
                logging.error(f"  Audio type: {type(feature.get('audio'))}, Text type: {type(feature.get('text'))}, Text: {str(feature.get('text', 'N/A'))[:50]}")
                continue
        
        audio_processing_time = time.perf_counter() - audio_processing_start
        if perf_monitor:
            perf_monitor.add_timing("data_collator/audio_processing", audio_processing_time)
        
        if not processed_features:
            # Log the full error details
            logging.error(f"ALL {len(features)} samples in batch failed processing!")
            logging.error(f"Sample types: {[type(f.get('audio')) for f in features]}")
            raise ValueError(f"No valid features to collate in batch (all {len(features)} samples failed)")
        
        # PHASE 2: Padding and batching (CPU)
        padding_start = time.perf_counter()
        
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in processed_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Ensure correct dtype
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
        
        padding_time = time.perf_counter() - padding_start
        if perf_monitor:
            perf_monitor.add_timing("data_collator/padding", padding_time)
        
        # PHASE 3: Transfer to GPU with pinned memory (if CUDA available)
        if torch.cuda.is_available() and self.device == "cuda":
            transfer_start = time.perf_counter()
            
            if self.use_pinned_memory:
                # Use pinned memory for faster transfer
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].pin_memory()
            
            # Transfer happens asynchronously when fed to model
            # But we log the pinning time
            transfer_time = time.perf_counter() - transfer_start
            if perf_monitor:
                perf_monitor.add_timing("data_collator/pin_memory", transfer_time)
        
        batch_time = time.perf_counter() - batch_start
        if perf_monitor:
            perf_monitor.add_timing("data_collator/total", batch_time)
            perf_monitor.increment_counter("batches_collated")
        
        return batch


class DistillationTrainer(Seq2SeqTrainer):
    """
    Enhanced trainer with:
    - Knowledge distillation support
    - Async checkpoint saving (non-blocking)
    - Detailed timing logs
    - GPU memory monitoring
    """
    
    def __init__(
        self,
        teacher_model: Optional[WhisperForConditionalGeneration] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        async_save: bool = True,
        *args,
        **kwargs
    ):
        """
        Initialize enhanced trainer
        
        Args:
            teacher_model: Teacher model for distillation (optional)
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss
            async_save: Use async checkpoint saving (non-blocking)
            *args, **kwargs: Arguments for Seq2SeqTrainer
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.async_save = async_save
        self.save_thread = None
        
        if teacher_model is not None:
            self.teacher_model.eval()
            # Freeze teacher
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            logger.info(f"✓ Distillation enabled (T={temperature}, α={alpha})")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with timing and optional distillation"""
        global perf_monitor
        
        loss_start = time.perf_counter()
        
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
        
        loss_time = time.perf_counter() - loss_start
        if perf_monitor:
            perf_monitor.add_timing("training/compute_loss", loss_time)
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with detailed timing"""
        global perf_monitor
        
        step_start = time.perf_counter()
        
        # Time CPU->GPU transfer
        if torch.cuda.is_available():
            transfer_start = time.perf_counter()
            inputs = self._prepare_inputs(inputs)
            torch.cuda.synchronize()
            transfer_time = time.perf_counter() - transfer_start
            if perf_monitor:
                perf_monitor.add_timing("training/cpu_to_gpu_transfer", transfer_time)
        else:
            inputs = self._prepare_inputs(inputs)
        
        # Forward + backward pass
        forward_start = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        forward_time = time.perf_counter() - forward_start
        
        if perf_monitor:
            perf_monitor.add_timing("training/forward_backward", forward_time)
        
        step_time = time.perf_counter() - step_start
        if perf_monitor:
            perf_monitor.add_timing("training/step_total", step_time)
            perf_monitor.increment_counter("training_steps")
        
        return loss
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Save checkpoint with async support, timing, and metrics reporting"""
        global perf_monitor
        
        checkpoint_start = time.perf_counter()
        
        # Extract current metrics from state if available
        current_metrics = {}
        if hasattr(self.state, 'log_history') and self.state.log_history:
            # Get latest evaluation metrics from log history
            for log_entry in reversed(self.state.log_history):
                if 'eval_wer' in log_entry or 'eval_cer' in log_entry:
                    current_metrics = {k: v for k, v in log_entry.items() if k.startswith('eval_')}
                    break
        
        if self.async_save:
            # Wait for previous save to complete
            if self.save_thread and self.save_thread.is_alive():
                logger.info("⏳ Waiting for previous checkpoint save to complete...")
                wait_start = time.perf_counter()
                self.save_thread.join()
                wait_time = time.perf_counter() - wait_start
                logger.info(f"✓ Previous save completed in {wait_time:.2f}s")
                if perf_monitor:
                    perf_monitor.add_timing("checkpoint/wait_for_previous", wait_time)
            
            # Save checkpoint in background thread
            logger.info("💾 Starting async checkpoint save (non-blocking)...")
            self.save_thread = threading.Thread(
                target=self._save_checkpoint_sync,
                args=(model, trial, current_metrics, checkpoint_start)
            )
            self.save_thread.start()
            logger.info("✓ Checkpoint save started in background")
        else:
            # Synchronous save (blocking)
            logger.info("💾 Saving checkpoint (blocking)...")
            self._save_checkpoint_sync(model, trial, current_metrics, checkpoint_start)
        
        # Log time for checkpoint initiation (not full save if async)
        init_time = time.perf_counter() - checkpoint_start
        if perf_monitor:
            perf_monitor.add_timing("checkpoint/initiate", init_time)
    
    def _save_checkpoint_sync(self, model, trial, current_metrics, start_time):
        """Synchronous checkpoint save with timing and metrics reporting"""
        global perf_monitor
        
        try:
            # Call parent save method - Seq2SeqTrainer._save_checkpoint takes only model and trial
            super()._save_checkpoint(model, trial)
            
            save_time = time.perf_counter() - start_time
            logger.info(f"✓ Checkpoint saved in {save_time:.2f}s")
            
            # Save comprehensive metrics report
            if current_metrics and self.args.output_dir:
                try:
                    self._save_metrics_report(current_metrics, save_time)
                except Exception as e:
                    logger.warning(f"Failed to save metrics report: {e}")
            
            if perf_monitor:
                perf_monitor.add_timing("checkpoint/save_total", save_time)
                perf_monitor.increment_counter("checkpoints_saved")
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _save_metrics_report(self, current_metrics, checkpoint_time):
        """Save comprehensive metrics report to JSON and text files"""
        import json
        from datetime import datetime
        from pathlib import Path
        
        # Create metrics directory
        metrics_dir = Path(self.args.output_dir) / "metrics_reports"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dataset sizes
        train_samples = len(self.train_dataset) if self.train_dataset else 0
        eval_samples = len(self.eval_dataset) if self.eval_dataset else 0
        
        # Build comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_step": self.state.global_step,
            "epoch": self.state.epoch,
            "checkpoint_save_time_seconds": round(checkpoint_time, 2),
            "dataset": {
                "train_samples": train_samples,
                "eval_samples": eval_samples,
                "total_samples": train_samples + eval_samples
            },
            "training_stats": {
                "steps_completed": self.state.global_step,
                "epochs_completed": round(self.state.epoch, 2) if self.state.epoch else 0,
                "total_epochs_planned": self.args.num_train_epochs
            },
            "metrics": {}
        }
        
        # Add all evaluation metrics
        for key, value in current_metrics.items():
            # Remove 'eval_' prefix for cleaner report
            metric_name = key.replace('eval_', '')
            if isinstance(value, (int, float)):
                report["metrics"][metric_name] = round(float(value), 4)
            else:
                report["metrics"][metric_name] = value
        
        # Save JSON report
        json_file = metrics_dir / f"metrics_step_{self.state.global_step}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable text report
        txt_file = metrics_dir / f"metrics_step_{self.state.global_step}.txt"
        with open(txt_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"METRICS REPORT - Step {self.state.global_step}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Epoch: {report['epoch']:.2f} / {report['training_stats']['total_epochs_planned']}\n")
            f.write(f"Checkpoint Save Time: {report['checkpoint_save_time_seconds']:.2f}s\n\n")
            
            f.write("DATASET INFO:\n")
            f.write(f"  Train Samples: {report['dataset']['train_samples']:,}\n")
            f.write(f"  Eval Samples: {report['dataset']['eval_samples']:,}\n")
            f.write(f"  Total Samples: {report['dataset']['total_samples']:,}\n\n")
            
            f.write("EVALUATION METRICS:\n")
            for metric_name, value in report['metrics'].items():
                f.write(f"  {metric_name.upper()}: {value}\n")
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"📊 Metrics report saved:")
        logger.info(f"   JSON: {json_file}")
        logger.info(f"   Text: {txt_file}")
        if report['metrics']:
            logger.info(f"   Metrics: {', '.join([f'{k}={v}' for k, v in report['metrics'].items()])}")


class MWCASRTrainer:
    """Main training class for MWC ASR system"""
    
    def __init__(self, config_path: str, max_parquet_files_override: Optional[int] = None):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to YAML configuration file
            max_parquet_files_override: Override max_parquet_files from config (for command-line testing)
        """
        global logger, perf_monitor
        
        # Setup distributed training FIRST
        self.dist_info = setup_distributed()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Apply max_parquet_files override if provided
        self.max_parquet_files_override = max_parquet_files_override
        if max_parquet_files_override is not None and logger is not None:
            logger.info(f"⚠️  Overriding max_parquet_files with command-line value: {max_parquet_files_override}")
        
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
        
        # Setup logging (reinitialize global logger)
        logger = setup_logging(
            self.logs_dir,
            f"training_run",
            is_main_process=self.dist_info['is_main_process']
        )
        
        # Initialize performance monitor
        perf_monitor = PerformanceMonitor(logger)
        
        # Only main process prints detailed startup info
        if self.dist_info['is_main_process']:
            logger.info("="*80)
            logger.info("MWC ASR TRAINING SYSTEM - DISTRIBUTED GPU OPTIMIZED")
            logger.info("="*80)
            logger.info(f"Configuration: {config_path}")
            logger.info(f"Project root: {self.project_root}")
        
        # All processes log distributed info
        logger.info("="*80)
        logger.info("DISTRIBUTED TRAINING CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Distributed Mode: {self.dist_info['is_distributed']}")
        logger.info(f"World Size: {self.dist_info['world_size']}")
        logger.info(f"Global Rank: {self.dist_info['rank']}")
        logger.info(f"Local Rank (GPU): {self.dist_info['local_rank']}")
        logger.info(f"Main Process: {self.dist_info['is_main_process']}")
        
        # Log hardware configuration
        if self.dist_info['is_main_process']:
            logger.info("="*80)
            logger.info("HARDWARE CONFIGURATION")
            logger.info("="*80)
        logger.info(f"CPU cores available: {os.cpu_count()}")
        logger.info(f"DataLoader workers: {self.config['hardware'].get('num_workers', 4)}")
        
        # Device setup
                # Device setup - CRITICAL for distributed training
        if self.dist_info['is_distributed']:
            # In distributed mode, each process gets its own GPU based on local_rank
            local_rank = self.dist_info['local_rank']
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)  # CRITICAL: Bind this process to its GPU
            logger.info(f"✓ Device bound to: cuda:{local_rank}")
        else:
            # Single GPU or CPU mode
            self.device = torch.device(self.config['hardware']['device'])
            if torch.cuda.is_available():
                torch.cuda.set_device(0)

        if torch.cuda.is_available():
            local_rank = self.dist_info['local_rank']
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
            num_gpus = torch.cuda.device_count()
            
            logger.info(f"GPU {local_rank}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if self.dist_info['is_main_process']:
                logger.info(f"Total GPUs Available: {num_gpus}")
                logger.info(f"GPUs in Use: {self.dist_info['world_size']}")
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"PyTorch Version: {torch.__version__}")
                logger.info(f"NCCL Available: {dist.is_nccl_available()}")
                
                # Log NVLink topology for H100 NVL
                logger.info("\\nGPU Topology (NVLink):")
                logger.info("  GPU 0 ↔ GPU 1 (NVLink pair)")
                logger.info("  GPU 2 ↔ GPU 3 (NVLink pair)")
                logger.info("  All GPUs connected via NVSwitch")
                
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch CUDA Enabled: {torch.cuda.is_available()}")
            
            # Force all operations to use GPU
            torch.set_default_device(self.device)
            logger.info(f"✓ Default device set to: {self.device}")
            
            # Enable TF32 for better performance on Ampere GPUs (A100, H100)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ TF32 enabled for Ampere/Hopper GPUs (H100)")
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            logger.info("✓ CuDNN benchmark mode enabled")
            
            perf_monitor.log_gpu_memory("Initial")
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
        """Load Whisper processor with timing"""
        with timer(logger, "load_processor"):
            logger.info("Loading Whisper processor...")
            model_name = self.config['model']['base_model']
            
            # FORCE language to English if dataset is English
            language = self.config['model'].get('language', 'english')
            logger.info(f"Target language: {language}")
            
            self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
            logger.info(f"✓ Loaded processor from {model_name}")
    
    def load_model(self):
        """Load and configure Whisper model with QLoRA and timing"""
        with timer(logger, "load_model"):
            logger.info("="*80)
            logger.info("MODEL SETUP")
            logger.info("="*80)
            
            model_name = self.config['model']['base_model']
            logger.info(f"Base model: {model_name}")
            
            # Set language and task tokens
            language = self.config['model'].get('language', 'english')
            task = self.config['model'].get('task', 'transcribe')
            logger.info(f"Language: {language} | Task: {task}")

            # In distributed mode, we need to be careful with device_map
            # Each process should load model to its assigned GPU
            if self.dist_info['is_distributed']:
                # Load to specific GPU for this process
                device_map = {'': self.dist_info['local_rank']}
            else:
                device_map = "auto"            

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
                    device_map=device_map
                )
                
                # Prepare for k-bit training
                self.model = prepare_model_for_kbit_training(self.model)
                
                logger.info("✓ Model prepared for 4-bit training with QLoRA")
            
            else:
                # Load in FP16
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=device_map
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

        # In distributed mode, we need to be careful with device_map
        # Each process should load model to its assigned GPU
        if self.dist_info['is_distributed']:
            # Load to specific GPU for this process
            device_map = {'': self.dist_info['local_rank']}
        else:
            device_map = "auto"          
        
        self.teacher_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_name,
            torch_dtype=torch.float16,
            device_map=device_map
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
        """Prepare dataset using pluggable data handler with timing"""
        with timer(logger, "prepare_dataset"):
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
                # Use override if provided, otherwise use config value
                max_parquet_files = self.max_parquet_files_override if self.max_parquet_files_override is not None else parquet_config.get('max_parquet_files')
                data_config = DataSourceConfig(
                    source_type="parquet",
                    path=parquet_config.get('file_path'),
                    audio_column=parquet_config.get('audio_column', 'audio'),
                    audio_path_column=parquet_config.get('audio_path_column', 'audio_path'),
                    text_column=parquet_config.get('text_column', 'text'),
                    max_parquet_files=max_parquet_files
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
            
            # Load dataset with timing and GPU optimization
            load_start = time.perf_counter()
            self.dataset_handler = PluggableDatasetHandler(
                config=data_config,
                audio_loader=audio_loader,
                processor=self.processor,
                max_duration=self.config['data']['max_duration_seconds'],
                min_duration=self.config['data']['min_duration_seconds'],
                device=self.config['hardware']['device'],
                use_pinned_memory=self.config['hardware'].get('pin_memory', True)
            )
            
            full_dataset = self.dataset_handler.load_dataset()
            load_time = time.perf_counter() - load_start
            logger.info(f"✓ Loaded {len(full_dataset)} total samples in {load_time:.2f}s")
            perf_monitor.add_timing("dataset/load_full", load_time)
            
            # Check if dataset is empty
            if len(full_dataset) == 0:
                raise ValueError("Dataset is empty! No samples loaded. Please check your data configuration.")
            
            # Split into train/eval
            split_start = time.perf_counter()
            train_size = int(self.config['data']['train_split_ratio'] * len(full_dataset))
            eval_size = len(full_dataset) - train_size
            
            # Ensure at least 1 sample in eval if eval_strategy is not 'no'
            if eval_size == 0 and self.config['training'].get('eval_strategy', 'no') != 'no':
                logger.warning(f"Dataset has only {len(full_dataset)} samples. Adjusting split to ensure at least 1 eval sample.")
                if len(full_dataset) == 1:
                    logger.warning("Only 1 sample available - disabling evaluation")
                    train_size = 1
                    eval_size = 0
                    # Will set eval_dataset to None and disable evaluation
                else:
                    # Ensure at least 1 sample in eval
                    eval_size = max(1, int(len(full_dataset) * 0.1))  # At least 10% or 1 sample
                    train_size = len(full_dataset) - eval_size
            
            if eval_size > 0:
                # Create generator on the correct device
                device = self.config['hardware']['device']
                generator = torch.Generator(device=device if device != 'cpu' else None)
                generator.manual_seed(self.config['data']['random_seed'])
                
                self.train_dataset, self.eval_dataset = random_split(
                    full_dataset,
                    [train_size, eval_size],
                    generator=generator
                )
            else:
                # No eval split - use entire dataset for training
                self.train_dataset = full_dataset
                self.eval_dataset = None
                logger.warning("Evaluation disabled - eval_dataset set to None")
            
            split_time = time.perf_counter() - split_start
            perf_monitor.add_timing("dataset/split", split_time)
            
            logger.info(f"  Train samples: {len(self.train_dataset)}")
            logger.info(f"  Eval samples: {len(self.eval_dataset) if self.eval_dataset else 0}")
    
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
        metrics_config = self.config.get('metrics', {})
        if metrics_config.get('log_predictions', False):
            num_to_log = min(
                metrics_config.get('num_predictions_to_log', 5),
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
        
        # Determine eval strategy based on eval_dataset availability
        eval_strategy = self.config['training']['eval_strategy']
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            logger.warning("No eval dataset available - disabling evaluation")
            eval_strategy = "no"
        
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
            eval_strategy=eval_strategy,
            eval_steps=self.config['training']['eval_steps'] if eval_strategy != "no" else None,
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            logging_steps=self.config['training']['logging_steps'],
            logging_first_step=self.config['training']['logging_first_step'],
            report_to=self.config['training']['report_to'],
            predict_with_generate=self.config['training']['predict_with_generate'],
            generation_max_length=self.config['training']['generation_max_length'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'] if eval_strategy != "no" else False,
            metric_for_best_model=self.config['training']['metric_for_best_model'] if eval_strategy != "no" else None,
            greater_is_better=self.config['training']['greater_is_better'],
            seed=self.config['training']['seed'],
            dataloader_num_workers=self.config['hardware']['num_workers'],
            dataloader_pin_memory=self.config['hardware']['pin_memory'],
            remove_unused_columns=False,
            # Save in safetensors format
            save_safetensors=True,
            # Distributed training settings
            local_rank=self.dist_info['local_rank'],
            ddp_backend='nccl',  # NVIDIA GPUs
            ddp_find_unused_parameters=False,  # Optimization for fixed computation graph
        )
        
        if self.dist_info['is_main_process']:
            effective_batch_size = (
                self.config['training']['per_device_train_batch_size'] * 
                self.config['training']['gradient_accumulation_steps'] *
                self.dist_info['world_size']  # Account for distributed training
            )
            logger.info(f"  Epochs: {self.config['training']['num_epochs']}")
            logger.info(f"  Per-device batch size: {self.config['training']['per_device_train_batch_size']}")
            logger.info(f"  Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}")
            logger.info(f"  World size (GPUs): {self.dist_info['world_size']}")
            logger.info(f"  Effective global batch size: {effective_batch_size}")
            logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")

        logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        
        # Data collator with GPU optimization
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            fp16=False,  # Disabled for 4-bit quantization
            device=self.config['hardware']['device'],
            use_pinned_memory=self.config['hardware'].get('pin_memory', True)
        )
        logger.info(f"✓ Data collator configured (device={self.config['hardware']['device']}, pinned_memory={self.config['hardware'].get('pin_memory', True)})")
        
        # Create trainer with async checkpoint saving
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING TRAINER")
        logger.info("="*80)
        
        async_save = self.config.get('training', {}).get('async_checkpoint_save', True)
        logger.info(f"Async checkpoint saving: {async_save}")
        
        # Prepare callbacks
        callbacks = []
        
        # Add early stopping callback if enabled
        early_stopping_config = self.config['training'].get('early_stopping', {})
        if early_stopping_config.get('enabled', False) and eval_strategy != "no":
            early_stopping_patience = early_stopping_config.get('patience', 3)
            early_stopping_threshold = early_stopping_config.get('threshold', 0.0)
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold
                )
            )
            logger.info(f"✓ Early stopping enabled (patience={early_stopping_patience}, threshold={early_stopping_threshold})")
        else:
            if early_stopping_config.get('enabled', False) and eval_strategy == "no":
                logger.warning("Early stopping disabled - evaluation strategy is 'no'")
        
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=self.config['model'].get('distillation_temperature', 2.0),
            alpha=self.config['model'].get('distillation_alpha', 0.5),
            async_save=async_save,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.tokenizer,
            callbacks=callbacks,
        )
        
        logger.info("✓ Trainer initialized")
        perf_monitor.log_gpu_memory("After Trainer Init")
        
        # Synchronize all processes before training
        if self.dist_info['is_distributed']:
            dist.barrier()
            logger.info("✓ All processes synchronized")
        
        # Train!
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING TRAINING")
        logger.info("="*80)
        logger.info(f"Total training steps: {len(self.train_dataset) // (self.config['training']['per_device_train_batch_size'] * self.config['training']['gradient_accumulation_steps']) * self.config['training']['num_epochs']}")
        perf_monitor.log_gpu_memory("Before Training")
        
        train_start = datetime.now()
        train_start_perf = time.perf_counter()
        
        train_result = trainer.train()
        
        train_end = datetime.now()
        train_end_perf = time.perf_counter()
        train_duration = (train_end - train_start).total_seconds()
        
        # Synchronize all processes after training
        if self.dist_info['is_distributed']:
            dist.barrier()
        
        if self.dist_info['is_main_process']:
            logger.info("\\n" + "="*80)
            logger.info("✅ DISTRIBUTED TRAINING COMPLETE")
            logger.info("="*80)

        logger.info(f"Training duration: {train_duration / 3600:.2f} hours ({train_duration:.2f} seconds)")
        logger.info(f"Training duration (perf_counter): {train_end_perf - train_start_perf:.2f} seconds")
        logger.info(f"Average seconds per epoch: {train_duration / self.config['training']['num_epochs']:.2f}s")
        perf_monitor.log_gpu_memory("After Training")
        
        # Log performance summary
        perf_monitor.log_summary()
        
        # Save final model
        logger.info("\nSaving final model...")
        save_start = time.perf_counter()
        final_model_path = self.final_models_dir / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(final_model_path))
        trainer.save_state()
        save_time = time.perf_counter() - save_start
        logger.info(f"✓ Final model saved in {save_time:.2f}s")
        
        # Save training results with timing data
        results = {
            "model": self.config['model']['base_model'],
            "language": self.config['model'].get('language', 'english'),
            "training_duration_seconds": train_duration,
            "training_duration_hours": train_duration / 3600,
            "metrics": train_result.metrics,
            "training_date": datetime.now().isoformat(),
            "configuration": self.config,
            "performance_summary": {
                "timings": {k: {"avg": sum(v)/len(v), "total": sum(v), "count": len(v)} 
                           for k, v in perf_monitor.timings.items() if v},
                "counters": perf_monitor.counters
            }
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
            
            # Cleanup distributed training
            if self.dist_info['is_distributed']:
                cleanup_distributed()
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            # Cleanup on error
            if self.dist_info['is_distributed']:
                cleanup_distributed()
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
    parser.add_argument(
        '--max-parquet-files',
        type=int,
        default=None,
        help='Maximum number of parquet files to load (overrides config). Use for testing with subset of data.'
    )
    
    args = parser.parse_args()
    
    # Run training with optional override
    trainer = MWCASRTrainer(config_path=args.config, max_parquet_files_override=args.max_parquet_files)
    results = trainer.run()
