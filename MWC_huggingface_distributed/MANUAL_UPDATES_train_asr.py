"""
Remaining Manual Updates for train_asr.py
==========================================

Due to file size and complexity, some changes need to be applied manually.
This file documents all remaining changes with exact line numbers and code.

SEARCH AND REPLACE INSTRUCTIONS:
--------------------------------

1. Update device setup (around line 760):
   FIND:
        logger.info(f"CPU cores available: {os.cpu_count()}")
        logger.info(f"DataLoader workers: {self.config['hardware'].get('num_workers', 4)}")
        
        # Device setup
        self.device = torch.device(self.config['hardware']['device'])

   REPLACE WITH:
        logger.info(f"CPU cores available: {os.cpu_count()}")
        logger.info(f"DataLoader workers: {self.config['hardware'].get('num_workers', 4)}")
        
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

2. Update GPU logging (around line 766):
   FIND:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            num_gpus = torch.cuda.device_count()
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"Number of GPUs: {num_gpus}")

   REPLACE WITH:
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

3. Update load_model() method (around line 890):
   FIND:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )

   REPLACE WITH:
                # In distributed mode, we need to be careful with device_map
                # Each process should load model to its assigned GPU
                if self.dist_info['is_distributed']:
                    # Load to specific GPU for this process
                    device_map = {'': self.dist_info['local_rank']}
                else:
                    device_map = "auto"
                
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map=device_map
                )

4. Update train() method arguments (around line 1070):
   FIND:
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.checkpoints_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            ...
            save_safetensors=True,
        )

   ADD AFTER save_safetensors=True,:
            # Distributed training settings
            local_rank=self.dist_info['local_rank'],
            ddp_backend='nccl',  # NVIDIA GPUs
            ddp_find_unused_parameters=False,  # Optimization for fixed computation graph

5. Update batch size logging (around line 1103):
   FIND:
        logger.info(f"  Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size: {self.config['training']['per_device_train_batch_size']}")
        logger.info(f"  Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  Effective batch size: {self.config['training']['per_device_train_batch_size'] * self.config['training']['gradient_accumulation_steps']}")

   REPLACE WITH:
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

6. Add synchronization before training (around line 1145):
   FIND:
        logger.info("✓ Trainer initialized")
        perf_monitor.log_gpu_memory("After Trainer Init")
        
        # Train!

   REPLACE WITH:
        logger.info("✓ Trainer initialized")
        perf_monitor.log_gpu_memory("After Trainer Init")
        
        # Synchronize all processes before training
        if self.dist_info['is_distributed']:
            dist.barrier()
            logger.info("✓ All processes synchronized")
        
        # Train!

7. Update training completion logging (around line 1160):
   FIND:
        logger.info("\\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*80)

   REPLACE WITH:
        # Synchronize all processes after training
        if self.dist_info['is_distributed']:
            dist.barrier()
        
        if self.dist_info['is_main_process']:
            logger.info("\\n" + "="*80)
            logger.info("✅ DISTRIBUTED TRAINING COMPLETE")
            logger.info("="*80)

8. Update run() method cleanup (around line 1240):
   FIND:
    def run(self):
        \"\"\"Execute full training pipeline\"\"\"
        try:
            # Load all components
            ...
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

   REPLACE WITH:
    def run(self):
        \"\"\"Execute full training pipeline\"\"\"
        try:
            # Load all components
            ...
            
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

VALIDATION:
-----------
After applying all changes, run:
python scripts/train_asr.py --help

Expected output should not show any import errors.

Then test distributed training:
torchrun --nproc_per_node=2 --standalone scripts/train_asr.py --config config/training_config.yaml --max-parquet-files 2
"""
