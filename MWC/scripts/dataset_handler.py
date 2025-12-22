#!/usr/bin/env python3
"""
Modular Dataset Handler - Plug and Play Architecture
Supports multiple data sources: local files, HuggingFace datasets, custom formats
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
import json
import pandas as pd
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset, Dataset as HFDataset, Audio
import numpy as np

from audio_loader import AudioLoader

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data source"""
    source_type: str  # 'local', 'huggingface', 'csv', 'json'
    path: Optional[str] = None
    dataset_name: Optional[str] = None
    split: str = "train"
    audio_column: str = "audio"
    text_column: str = "text"
    sample_rate_column: Optional[str] = "sampling_rate"


class PluggableDatasetHandler:
    """
    Modular dataset handler that supports multiple data sources
    Easy to swap data sources by just changing configuration
    """
    
    def __init__(
        self,
        config: DataSourceConfig,
        audio_loader: AudioLoader,
        processor: Any,
        max_duration: float = 30.0,
        min_duration: float = 0.5
    ):
        """
        Initialize dataset handler
        
        Args:
            config: Data source configuration
            audio_loader: AudioLoader instance
            processor: Whisper processor for feature extraction
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
        """
        self.config = config
        self.audio_loader = audio_loader
        self.processor = processor
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        self.dataset = None
        self.raw_data = None
    
    def load_dataset(self) -> Dataset:
        """
        Load dataset from configured source
        
        Returns:
            PyTorch Dataset ready for training
        """
        logger.info(f"Loading dataset from {self.config.source_type} source...")
        
        if self.config.source_type == "huggingface":
            return self._load_huggingface_dataset()
        elif self.config.source_type == "local":
            return self._load_local_dataset()
        elif self.config.source_type == "csv":
            return self._load_csv_dataset()
        elif self.config.source_type == "json":
            return self._load_json_dataset()
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")
    
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
            
            # Convert to custom dataset that handles audio loading manually
            return WhisperASRDataset(
                hf_dataset=hf_dataset,
                audio_loader=self.audio_loader,
                processor=self.processor,
                audio_column=self.config.audio_column,
                text_column=self.config.text_column,
                max_duration=self.max_duration,
                min_duration=self.min_duration,
                decode_audio=False  # We'll handle audio loading manually with soundfile/librosa
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
            Audio(sampling_rate=self.audio_loader.target_sample_rate)
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
            Audio(sampling_rate=self.audio_loader.target_sample_rate)
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


class WhisperASRDataset(Dataset):
    """PyTorch Dataset for Whisper ASR training"""
    
    def __init__(
        self,
        hf_dataset: HFDataset,
        audio_loader: AudioLoader,
        processor: Any,
        audio_column: str = "audio",
        text_column: str = "text",
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        decode_audio: bool = True
    ):
        """
        Initialize Whisper ASR dataset
        
        Args:
            hf_dataset: HuggingFace dataset
            audio_loader: AudioLoader instance
            processor: Whisper processor
            audio_column: Name of audio column
            text_column: Name of text column
            max_duration: Maximum audio duration
            min_duration: Minimum audio duration
            decode_audio: Whether audio is pre-decoded or needs manual loading
        """
        self.dataset = hf_dataset
        self.audio_loader = audio_loader
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.decode_audio = decode_audio
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample - access raw PyArrow data to avoid audio decoding"""
        
        # Access raw PyArrow table directly to bypass formatters/decoders
        pa_table = self.dataset._data
        
        # Get text directly from PyArrow (no decoding needed for text)
        text_col_idx = self.dataset.column_names.index(self.text_column)
        text = pa_table.column(text_col_idx)[idx].as_py()
        
        # Get audio struct directly from PyArrow (avoid Audio feature decoder)
        audio_col_idx = self.dataset.column_names.index(self.audio_column)
        audio_struct = pa_table.column(audio_col_idx)[idx].as_py()
        
        # audio_struct is a dict with 'bytes' and/or 'path'
        # Priority: bytes > path (bytes is always available for HF datasets)
        if isinstance(audio_struct, dict):
            if 'bytes' in audio_struct and audio_struct['bytes']:
                # Decode from bytes using soundfile - THIS IS THE PRIMARY PATH FOR HF DATASETS
                import io
                import soundfile as sf
                audio_bytes = audio_struct['bytes']
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                audio_array = audio_array.astype(np.float32)
            elif 'path' in audio_struct and audio_struct['path']:
                # Try loading from path (might be cached file)
                from pathlib import Path
                audio_path = audio_struct['path']
                
                # Check if path exists, if not try to find it in HF cache
                if not Path(audio_path).exists():
                    # Path is relative or cached, skip for now
                    raise FileNotFoundError(f"Audio path not accessible: {audio_path}. Use bytes instead.")
                
                audio_array, sampling_rate = self.audio_loader.load(
                    audio_path,
                    normalize=True,
                    validate_duration=True,
                    min_duration=self.min_duration,
                    max_duration=self.max_duration
                )
            else:
                raise ValueError(f"Audio struct missing both valid path and bytes: {audio_struct}")
        elif isinstance(audio_struct, str):
            # Direct path string
            audio_array, sampling_rate = self.audio_loader.load(
                audio_struct,
                normalize=True,
                validate_duration=True,
                min_duration=self.min_duration,
                max_duration=self.max_duration
            )
        else:
            raise ValueError(f"Unexpected audio data type: {type(audio_struct)}")
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Normalize to target sample rate if needed
        if sampling_rate != self.audio_loader.target_sr:
            import librosa
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sampling_rate, 
                target_sr=self.audio_loader.target_sr
            )
            sampling_rate = self.audio_loader.target_sr
        
        return {
            'audio': audio_array,
            'text': text,
            'sampling_rate': sampling_rate
        }


if __name__ == "__main__":
    # Test dataset handler
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Testing Pluggable Dataset Handler")
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
        processor=processor
    )
    
    dataset = handler.load_dataset()
    print(f"\n✓ Dataset loaded: {len(dataset)} samples")
    
    # Test first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Sample rate: {sample['sampling_rate']}")
