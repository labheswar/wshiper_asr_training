#!/usr/bin/env python3
"""
Multi-format Audio Loader with fallback support
Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Optional

class AudioLoader:
    """
    Robust audio loader with multiple backend support
    Tries: soundfile -> librosa -> pydub
    """
    
    SUPPORTED_FORMATS = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma']
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize audio loader
        
        Args:
            target_sr: Target sampling rate (default 16kHz for Whisper)
        """
        self.target_sr = target_sr
        self._check_backends()
    
    def _check_backends(self):
        """Check available audio backends"""
        self.has_soundfile = False
        self.has_librosa = False
        self.has_pydub = False
        
        try:
            import soundfile
            self.has_soundfile = True
        except ImportError:
            pass
        
        try:
            import librosa
            self.has_librosa = True
        except ImportError:
            pass
        
        try:
            import pydub
            self.has_pydub = True
        except ImportError:
            pass
        
        if not any([self.has_soundfile, self.has_librosa, self.has_pydub]):
            raise RuntimeError(
                "No audio backend available! Install at least one of: "
                "soundfile, librosa, or pydub"
            )
    
    def load(
        self,
        file_path: str,
        normalize: bool = True,
        validate_duration: bool = True,
        min_duration: float = 0.5,
        max_duration: float = 30.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic format detection and fallback
        
        Args:
            file_path: Path to audio file
            normalize: Normalize amplitude to [-1, 1]
            validate_duration: Check duration constraints
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check format
        ext = file_path.suffix.lower().lstrip('.')
        if ext not in self.SUPPORTED_FORMATS:
            warnings.warn(f"Unsupported format: {ext}. Attempting to load anyway...")
        
        # Try loading with available backends
        audio = None
        sr = None
        errors = []
        
        # Try soundfile first (fastest)
        if self.has_soundfile and audio is None:
            try:
                audio, sr = self._load_soundfile(file_path)
            except Exception as e:
                errors.append(f"soundfile: {e}")
        
        # Try librosa (most compatible)
        if self.has_librosa and audio is None:
            try:
                audio, sr = self._load_librosa(file_path)
            except Exception as e:
                errors.append(f"librosa: {e}")
        
        # Try pydub (fallback for exotic formats)
        if self.has_pydub and audio is None:
            try:
                audio, sr = self._load_pydub(file_path)
            except Exception as e:
                errors.append(f"pydub: {e}")
        
        if audio is None:
            raise RuntimeError(
                f"Failed to load {file_path} with all backends:\n" + 
                "\n".join(errors)
            )
        
        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = self._resample(audio, sr, self.target_sr)
            sr = self.target_sr
        
        # Normalize
        if normalize:
            audio = self._normalize(audio)
        
        # Validate duration
        if validate_duration:
            duration = len(audio) / sr
            if duration < min_duration:
                raise ValueError(
                    f"Audio too short: {duration:.2f}s < {min_duration}s"
                )
            if duration > max_duration:
                raise ValueError(
                    f"Audio too long: {duration:.2f}s > {max_duration}s"
                )
        
        return audio.astype(np.float32), sr
    
    def _load_soundfile(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load using soundfile"""
        import soundfile as sf
        audio, sr = sf.read(str(file_path), dtype='float32')
        return audio, sr
    
    def _load_librosa(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load using librosa"""
        import librosa
        audio, sr = librosa.load(str(file_path), sr=None, mono=False)
        return audio, sr
    
    def _load_pydub(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load using pydub (for exotic formats)"""
        from pydub import AudioSegment
        
        # Load audio
        audio = AudioSegment.from_file(str(file_path))
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1]
        samples = samples / (2**15)
        
        # Handle stereo
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        return samples, audio.frame_rate
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        # Try librosa for resampling (best quality)
        if self.has_librosa:
            import librosa
            return librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type='kaiser_best'
            )
        
        # Fallback: simple linear interpolation
        import scipy.signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        return scipy.signal.resample(audio, num_samples)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def get_duration(self, file_path: str) -> float:
        """Get audio duration in seconds"""
        audio, sr = self.load(file_path, validate_duration=False)
        return len(audio) / sr
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if file format is supported"""
        ext = Path(file_path).suffix.lower().lstrip('.')
        return ext in AudioLoader.SUPPORTED_FORMATS


if __name__ == "__main__":
    # Test audio loader
    import sys
    
    if len(sys.argv) > 1:
        loader = AudioLoader(target_sr=16000)
        audio, sr = loader.load(sys.argv[1])
        print(f"✓ Loaded: {sys.argv[1]}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(audio)/sr:.2f} seconds")
        print(f"  Shape: {audio.shape}")
        print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
    else:
        print("Audio Loader Test")
        print("Usage: python audio_loader.py <audio_file>")
