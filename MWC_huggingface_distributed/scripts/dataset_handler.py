#!/usr/bin/env python3
"""
GPU-Optimized Dataset Handler - Zero CPU Bottlenecks
Features:
- GPU-accelerated audio resampling (torchaudio CUDA)
- Pinned memory for fast CPU→GPU transfer
- Direct GPU tensor returns
- Parallel audio decoding
- Comprehensive timing logs
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
import json
import pandas as pd
from dataclasses import dataclass
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset, Dataset as HFDataset, Features, Value, Audio
import numpy as np
import io
import soundfile as sf
import time
from multiprocessing import Pool, cpu_count
from functools import partial

from audio_loader import AudioLoader

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data source"""
    source_type: str  # 'local', 'huggingface', 'csv', 'json', 'parquet'
    path: Optional[str] = None
    dataset_name: Optional[str] = None
    split: str = "train"
    audio_column: str = "audio"
    text_column: str = "text"
    sample_rate_column: Optional[str] = "sampling_rate"
    audio_path_column: Optional[str] = "audio_path"  # For parquet with separate audio files
    max_parquet_files: Optional[int] = None  # Limit number of parquet files to load (None = all)


class PluggableDatasetHandler:
    """
    GPU-Optimized dataset handler - minimizes CPU usage
    """
    
    def __init__(
        self,
        config: DataSourceConfig,
        audio_loader: AudioLoader,
        processor: Any,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        device: str = "cuda",
        use_pinned_memory: bool = True
    ):
        """
        Initialize GPU-optimized dataset handler
        
        Args:
            config: Data source configuration
            audio_loader: AudioLoader instance
            processor: Whisper processor for feature extraction
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            device: Target device ('cuda' or 'cpu')
            use_pinned_memory: Use pinned memory for faster transfer
        """
        self.config = config
        self.audio_loader = audio_loader
        self.processor = processor
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.device = device
        self.use_pinned_memory = use_pinned_memory and torch.cuda.is_available()
        
        self.dataset = None
        self.raw_data = None
        
        logger.info(f"🎮 Dataset Handler initialized:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Pinned Memory: {self.use_pinned_memory}")
        logger.info(f"   Target Sample Rate: {self.audio_loader.target_sr}")
    
    def load_dataset(self) -> Dataset:
        """
        Load dataset from configured source
        
        Returns:
            PyTorch Dataset ready for training
        """
        logger.info(f"Loading dataset from {self.config.source_type} source...")
        
        load_start = time.perf_counter()
        
        if self.config.source_type == "huggingface":
            dataset = self._load_huggingface_dataset()
        elif self.config.source_type == "local":
            dataset = self._load_local_dataset()
        elif self.config.source_type == "csv":
            dataset = self._load_csv_dataset()
        elif self.config.source_type == "json":
            dataset = self._load_json_dataset()
        elif self.config.source_type == "parquet":
            dataset = self._load_parquet_dataset()
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")
        
        load_time = time.perf_counter() - load_start
        logger.info(f"⏱️  Dataset loaded in {load_time:.2f}s")
        
        return dataset
    
    def _load_huggingface_dataset(self) -> Dataset:
        """Load dataset from HuggingFace"""
        logger.info(f"Loading HuggingFace dataset: {self.config.dataset_name}")
        
        try:
            # Load from HuggingFace without decoding audio
            hf_dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.split
            )
            
            # CRITICAL: Set format to None to disable automatic feature decoding
            # This prevents the Audio feature from trying to decode and requiring torchcodec
            hf_dataset = hf_dataset.with_format(None)
            
            logger.info(f"✓ Loaded {len(hf_dataset)} samples from HuggingFace")
            
            # Convert to GPU-optimized dataset
            return GPUOptimizedWhisperDataset(
                hf_dataset=hf_dataset,
                audio_loader=self.audio_loader,
                processor=self.processor,
                audio_column=self.config.audio_column,
                text_column=self.config.text_column,
                max_duration=self.max_duration,
                min_duration=self.min_duration,
                device=self.device,
                use_pinned_memory=self.use_pinned_memory
            )
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise
    
    def _load_local_dataset(self) -> Dataset:
        """Load dataset from local directory"""
        audio_path = Path(self.config.path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio path not found: {audio_path}")
        
        logger.info(f"Loading local dataset from: {audio_path}")
        
        # Find all audio files
        audio_files = []
        for fmt in self.audio_loader.SUPPORTED_FORMATS:
            audio_files.extend(list(audio_path.rglob(f"*{fmt}")))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Look for corresponding text files or metadata
        metadata = self._find_metadata(audio_path)
        
        if metadata is None:
            logger.warning("No metadata found - using filename as text")
            data = [
                {
                    'audio': str(f),
                    'text': f.stem.replace('_', ' ')
                }
                for f in audio_files
            ]
        else:
            data = metadata
        
        # Create HuggingFace dataset
        hf_dataset = HFDataset.from_dict({
            'audio': [d['audio'] for d in data],
            'text': [d['text'] for d in data]
        })
        
        # Cast audio column to Audio type
        hf_dataset = hf_dataset.cast_column(
            'audio', 
            Audio(sampling_rate=self.audio_loader.target_sr)
        )
        
        return WhisperASRDataset(
            hf_dataset=hf_dataset,
            audio_loader=self.audio_loader,
            processor=self.processor,
            audio_column='audio',
            text_column='text',
            max_duration=self.max_duration,
            min_duration=self.min_duration
        )
    
    def _load_csv_dataset(self) -> Dataset:
        """Load dataset from CSV file"""
        csv_path = Path(self.config.path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading CSV dataset from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate columns
        if self.config.audio_column not in df.columns:
            raise ValueError(f"Audio column '{self.config.audio_column}' not found in CSV")
        if self.config.text_column not in df.columns:
            raise ValueError(f"Text column '{self.config.text_column}' not found in CSV")
        
        # Create dataset
        data = []
        base_path = csv_path.parent
        
        for _, row in df.iterrows():
            audio_file = row[self.config.audio_column]
            
            # Handle relative paths
            if not Path(audio_file).is_absolute():
                audio_file = str(base_path / audio_file)
            
            data.append({
                'audio': audio_file,
                'text': row[self.config.text_column]
            })
        
        hf_dataset = HFDataset.from_dict({
            'audio': [d['audio'] for d in data],
            'text': [d['text'] for d in data]
        })
        
        hf_dataset = hf_dataset.cast_column(
            'audio',
            Audio(sampling_rate=self.audio_loader.target_sample_rate)
        )
        
        logger.info(f"✓ Loaded {len(data)} samples from CSV")
        
        return WhisperASRDataset(
            hf_dataset=hf_dataset,
            audio_loader=self.audio_loader,
            processor=self.processor,
            audio_column='audio',
            text_column='text',
            max_duration=self.max_duration,
            min_duration=self.min_duration
        )
    
    def _load_parquet_dataset(self) -> Dataset:
        """Load dataset from Parquet file(s) with optimized parallel loading"""
        parquet_path = Path(self.config.path)
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet path not found: {parquet_path}")
        
        logger.info(f"Loading Parquet dataset from: {parquet_path}")
        
        # Handle both single file and directory of parquet files
        if parquet_path.is_file():
            #df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded parquet file: {parquet_path.name}")
        elif parquet_path.is_dir():
            # Load all parquet files in directory (recursively)
            #parquet_files = list(parquet_path.rglob("*.parquet"))
            parquet_files = [str(f) for f in parquet_path.rglob("*.parquet")]
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {parquet_path}")
            
            logger.info(f"Found {len(parquet_files)} parquet files")
            
            # Apply max_parquet_files limit if specified
            if self.config.max_parquet_files is not None and self.config.max_parquet_files > 0:
                if self.config.max_parquet_files < len(parquet_files):
                    parquet_files = parquet_files[:self.config.max_parquet_files]
                    logger.info(f"⚠️  Limited to {len(parquet_files)} parquet files (max_parquet_files={self.config.max_parquet_files})")
            
            # Parallel loading with multiprocessing for speed
            logger.info(f"Loading {len(parquet_files)} parquet files in parallel...")
            load_start = time.perf_counter()
            
            # Determine optimal number of workers
            num_workers = min(cpu_count(), len(parquet_files), 8)  # Cap at 8 to avoid memory issues
            logger.info(f"Using {num_workers} parallel workers")
            
            # Load files in parallel
            #with Pool(num_workers) as pool:
            #    dfs = pool.map(pd.read_parquet, parquet_files)
            
            # Concatenate results
            #logger.info(f"Concatenating {len(dfs)} dataframes...")
            #df = pd.concat(dfs, ignore_index=True)
            #logger.info(f"Clearing {len(dfs)} dataframes...")
            del dfs
            
            load_time = time.perf_counter() - load_start
            #logger.info(f"✓ Loaded {len(df)} rows from {len(parquet_files)} files in {load_time:.2f}s")
        else:
            raise ValueError(f"Invalid parquet path: {parquet_path}")
        
        # Validate columns
        #if self.config.text_column not in df.columns:
        #    raise ValueError(f"Text column '{self.config.text_column}' not found in parquet")
        
        # Handle two cases:
        # Case 1: audio column contains binary audio data (bytes)
        # Case 2: separate audio_path column with paths to audio files
        
        data = []
        base_path = parquet_path.parent if parquet_path.is_file() else parquet_path
        
        hf_dataset = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",
            columns=[self.config.audio_column, self.config.text_column],
            #streaming=True,
            cache_dir="/opt/app-root/src/data/hf_cache"
        )

        #if self.config.audio_column in df.columns:
        #    # Audio data embedded in parquet
        #    logger.info(f"Using embedded audio from column: {self.config.audio_column}")
        #    #for _, row in df.iterrows():
        #    #    data.append({
        #    #        'audio': row[self.config.audio_column],
        #    #        'text': row[self.config.text_column]
        #    #    })
        #    # Rename columns within the HF Dataset (happens in-place/efficiently)
        #    hf_dataset = hf_dataset.rename_columns({
        #        self.config.audio_column: 'audio',
        #        self.config.text_column: 'text'
        #    })
        #elif self.config.audio_path_column and self.config.audio_path_column in df.columns:
        #    # Audio files referenced by path
        #    logger.info(f"Using audio files from column: {self.config.audio_path_column}")
        #    for _, row in df.iterrows():
        #        audio_file = row[self.config.audio_path_column]
        #        
        #        # Handle relative paths
        #        if not Path(audio_file).is_absolute():
        #            audio_file = str(base_path / audio_file)
        #        
        #        data.append({
        #            'audio': audio_file,
        #            'text': row[self.config.text_column]
        #        })
        #        #hf_dataset = HFDataset.from_dict({
        #        #    'audio': [d['audio'] for d in data],
        #        #    'text': [d['text'] for d in data]
        #        #})

        #else:
        #    raise ValueError(
        #        f"Parquet must contain either '{self.config.audio_column}' column (audio bytes) "
        #        f"or '{self.config.audio_path_column}' column (audio file paths)"
        #    )

        #logger.info(f"Clearing concatenated dataframe...")
        #del df
        
        #logger.info(f"✓ Loaded {len(data)} samples from Parquet")
        #logger.info(f"✓ Clearing {len(data)} samples from data obj Parquet")
        #del data
        # Rename columns within the HF Dataset (happens in-place/efficiently)
        hf_dataset = hf_dataset.rename_columns({
            self.config.audio_column: 'audio',
            self.config.text_column: 'text'
        })
        # Only cast to Audio if not already binary data
        if self.config.audio_path_column:
            hf_dataset = hf_dataset.cast_column(
                'audio',
                Audio(sampling_rate=self.audio_loader.target_sr)
            )
        
        return GPUOptimizedWhisperDataset(
            hf_dataset=hf_dataset,
            audio_loader=self.audio_loader,
            processor=self.processor,
            audio_column='audio',
            text_column='text',
            max_duration=self.max_duration,
            min_duration=self.min_duration,
            device=self.device,
            use_pinned_memory=self.use_pinned_memory
        )
    
    def _load_json_dataset(self) -> Dataset:
        """Load dataset from JSON file"""
        json_path = Path(self.config.path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        logger.info(f"Loading JSON dataset from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Support different JSON formats
        if isinstance(data, dict):
            # Format: {"audio": [...], "text": [...]}
            if self.config.audio_column in data and self.config.text_column in data:
                audio_files = data[self.config.audio_column]
                texts = data[self.config.text_column]
            else:
                raise ValueError(f"JSON must contain '{self.config.audio_column}' and '{self.config.text_column}' keys")
        elif isinstance(data, list):
            # Format: [{"audio": "...", "text": "..."}, ...]
            audio_files = [item[self.config.audio_column] for item in data]
            texts = [item[self.config.text_column] for item in data]
        else:
            raise ValueError("Unsupported JSON format")
        
        # Handle relative paths
        base_path = json_path.parent
        audio_files = [
            str(base_path / f) if not Path(f).is_absolute() else f
            for f in audio_files
        ]
        
        hf_dataset = HFDataset.from_dict({
            'audio': audio_files,
            'text': texts
        })
        
        hf_dataset = hf_dataset.cast_column(
            'audio',
            Audio(sampling_rate=self.audio_loader.target_sr)
        )
        
        logger.info(f"✓ Loaded {len(audio_files)} samples from JSON")
        
        return WhisperASRDataset(
            hf_dataset=hf_dataset,
            audio_loader=self.audio_loader,
            processor=self.processor,
            audio_column='audio',
            text_column='text',
            max_duration=self.max_duration,
            min_duration=self.min_duration
        )
    
    def _find_metadata(self, audio_path: Path) -> Optional[List[Dict]]:
        """Find metadata file in audio directory"""
        
        # Look for common metadata files
        metadata_files = [
            audio_path / "metadata.csv",
            audio_path / "metadata.json",
            audio_path / "transcripts.csv",
            audio_path / "transcripts.json",
        ]
        
        for meta_file in metadata_files:
            if meta_file.exists():
                logger.info(f"Found metadata file: {meta_file}")
                
                if meta_file.suffix == '.csv':
                    df = pd.read_csv(meta_file)
                    # Try to find audio and text columns
                    audio_col = self._find_column(df, ['audio', 'file', 'path', 'filename'])
                    text_col = self._find_column(df, ['text', 'transcript', 'transcription'])
                    
                    if audio_col and text_col:
                        return [
                            {
                                'audio': str(audio_path / row[audio_col]),
                                'text': row[text_col]
                            }
                            for _, row in df.iterrows()
                        ]
                
                elif meta_file.suffix == '.json':
                    with open(meta_file, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        # Convert dict format to list
                        audio_col = self._find_key(data, ['audio', 'file', 'path'])
                        text_col = self._find_key(data, ['text', 'transcript'])
                        
                        if audio_col and text_col:
                            return [
                                {
                                    'audio': str(audio_path / a),
                                    'text': t
                                }
                                for a, t in zip(data[audio_col], data[text_col])
                            ]
        
        return None
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find column in dataframe from list of candidates"""
        for col in df.columns:
            if col.lower() in candidates:
                return col
        return None
    
    def _find_key(self, data: dict, candidates: List[str]) -> Optional[str]:
        """Find key in dictionary from list of candidates"""
        for key in data.keys():
            if key.lower() in candidates:
                return key
        return None


class GPUOptimizedWhisperDataset(Dataset):
    """
    GPU-Optimized PyTorch Dataset for Whisper ASR
    
    Features:
    - GPU-accelerated resampling (torchaudio CUDA)
    - Pinned memory for fast CPU→GPU transfer
    - Direct GPU tensor returns
    - Comprehensive timing logs
    """
    
    def __init__(
        self,
        hf_dataset: HFDataset,
        audio_loader: AudioLoader,
        processor: Any,
        audio_column: str = "audio",
        text_column: str = "text",
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        device: str = "cuda",
        use_pinned_memory: bool = True
    ):
        """Initialize GPU-optimized dataset"""
        self.dataset = hf_dataset
        self.audio_loader = audio_loader
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.device = device
        self.use_pinned_memory = use_pinned_memory and torch.cuda.is_available()
        
        # Pre-create GPU resampler for faster processing
        self.gpu_resampler = None
        if torch.cuda.is_available():
            # Will be created on-demand with correct input sample rate
            pass
        
        # Timing statistics
        self.timing_stats = {
            'decode': [],
            'resample': [],
            'to_gpu': [],
            'total': []
        }
        self.log_every = 100  # Log timing every N samples
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        GPU-optimized sample loading
        
        Returns GPU tensors directly to minimize CPU→GPU transfer overhead
        """
        total_start = time.perf_counter()
        
        # ============ STEP 1: Get raw data from PyArrow (CPU) ============
        pa_table = self.dataset._data
        
        text_col_idx = self.dataset.column_names.index(self.text_column)
        text = pa_table.column(text_col_idx)[idx].as_py()
        
        audio_col_idx = self.dataset.column_names.index(self.audio_column)
        audio_struct = pa_table.column(audio_col_idx)[idx].as_py()
        
        # ============ STEP 2: Decode audio on CPU (unavoidable) ============
        decode_start = time.perf_counter()
        
        if isinstance(audio_struct, dict):
            if 'bytes' in audio_struct and audio_struct['bytes']:
                # Decode from bytes using soundfile (fastest CPU decoder)
                audio_bytes = audio_struct['bytes']
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                audio_array = audio_array.astype(np.float32)
            elif 'path' in audio_struct and audio_struct['path']:
                audio_path = audio_struct['path']
                if not Path(audio_path).exists():
                    raise FileNotFoundError(f"Audio path not accessible: {audio_path}")
                audio_array, sampling_rate = self.audio_loader.load(
                    audio_path,
                    normalize=True,
                    validate_duration=True,
                    min_duration=self.min_duration,
                    max_duration=self.max_duration
                )
            else:
                raise ValueError(f"Audio struct missing both path and bytes")
        elif isinstance(audio_struct, str):
            audio_array, sampling_rate = self.audio_loader.load(
                audio_struct,
                normalize=True,
                validate_duration=True,
                min_duration=self.min_duration,
                max_duration=self.max_duration
            )
        else:
            raise ValueError(f"Unexpected audio data type: {type(audio_struct)}")
        
        decode_time = time.perf_counter() - decode_start
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # ============ STEP 3: Convert to torch tensor with pinned memory ============
        to_gpu_start = time.perf_counter()
        
        # Convert to torch tensor on CPU first
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Use pinned memory for faster transfer (only if not in worker process)
        # Pin memory in worker processes can cause CUDA initialization issues
        worker_info = torch.utils.data.get_worker_info()
        in_worker = worker_info is not None
        
        if self.use_pinned_memory and not in_worker:
            try:
                audio_tensor = audio_tensor.pin_memory()
            except RuntimeError:
                # CUDA not available or initialization error - skip pinning
                pass
        
        # Move to GPU ONLY in main process (not in workers)
        # Workers return CPU tensors, main process handles GPU transfer
        if self.device != 'cpu' and not in_worker:
            audio_tensor = audio_tensor.to(self.device, non_blocking=self.use_pinned_memory)
        
        to_gpu_time = time.perf_counter() - to_gpu_start
        
        # ============ STEP 4: GPU-accelerated resampling ============
        resample_start = time.perf_counter()
        
        if sampling_rate != self.audio_loader.target_sr:
            # Use torchaudio resampler (CPU in workers, GPU in main process)
            if self.gpu_resampler is None or getattr(self, '_last_sr', None) != sampling_rate:
                self.gpu_resampler = T.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.audio_loader.target_sr
                )
                # Only move to GPU if device is not CPU AND not in worker process
                if self.device != 'cpu' and not in_worker:
                    self.gpu_resampler = self.gpu_resampler.to(self.device)
                self._last_sr = sampling_rate
            
            audio_tensor = self.gpu_resampler(audio_tensor)
            sampling_rate = self.audio_loader.target_sr
        
        resample_time = time.perf_counter() - resample_start
        
        # ============ STEP 5: Return GPU tensor ============
        total_time = time.perf_counter() - total_start
        
        # Update timing stats
        self.timing_stats['decode'].append(decode_time)
        self.timing_stats['resample'].append(resample_time)
        self.timing_stats['to_gpu'].append(to_gpu_time)
        self.timing_stats['total'].append(total_time)
        
        # Log timing periodically
        if idx % self.log_every == 0 and idx > 0:
            self._log_timing_stats()
        
        # Return GPU tensor directly (no CPU→GPU transfer in collate_fn)
        return {
            'audio': audio_tensor,  # Already on GPU!
            'text': text,
            'sampling_rate': sampling_rate
        }
    
    def _log_timing_stats(self):
        """Log timing statistics"""
        if not self.timing_stats['total']:
            return
        
        avg_decode = np.mean(self.timing_stats['decode'][-self.log_every:]) * 1000
        avg_resample = np.mean(self.timing_stats['resample'][-self.log_every:]) * 1000
        avg_to_gpu = np.mean(self.timing_stats['to_gpu'][-self.log_every:]) * 1000
        avg_total = np.mean(self.timing_stats['total'][-self.log_every:]) * 1000
        
        logger.info(
            f"⏱️  Dataset Loading (avg {self.log_every} samples): "
            f"decode={avg_decode:.2f}ms, resample={avg_resample:.2f}ms, "
            f"CPU→GPU={avg_to_gpu:.2f}ms, total={avg_total:.2f}ms"
        )


# Keep the old WhisperASRDataset for backward compatibility
class WhisperASRDataset(GPUOptimizedWhisperDataset):
    """Backward compatibility wrapper"""
    def __init__(self, *args, decode_audio=True, **kwargs):
        # Ignore decode_audio parameter
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    # Test GPU-optimized dataset handler
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Testing GPU-Optimized Dataset Handler")
    print("="*60)
    
    # Example: Test HuggingFace dataset
    config = DataSourceConfig(
        source_type="huggingface",
        dataset_name="MrDragonFox/Elise",
        split="train",
        audio_column="audio",
        text_column="text"
    )
    
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    audio_loader = AudioLoader(target_sample_rate=16000)
    
    handler = PluggableDatasetHandler(
        config=config,
        audio_loader=audio_loader,
        processor=processor,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_pinned_memory=True
    )
    
    dataset = handler.load_dataset()
    print(f"\n✓ Dataset loaded: {len(dataset)} samples")
    
    # Test first few samples to see GPU optimization
    for i in range(5):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Audio device: {sample['audio'].device}")
        print(f"  Audio shape: {sample['audio'].shape}")
        print(f"  Text: {sample['text'][:50]}...")
