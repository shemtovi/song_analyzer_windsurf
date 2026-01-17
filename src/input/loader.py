"""Audio loading and preprocessing utilities."""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional


class AudioLoader:
    """Handles audio file loading and preprocessing."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4"}

    def __init__(
        self,
        target_sr: int = 22050,
        mono: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize AudioLoader.

        Args:
            target_sr: Target sample rate for resampling
            mono: Convert to mono if True
            normalize: Normalize audio amplitude if True
        """
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize

    def load(self, path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and preprocess.

        Args:
            path: Path to audio file

        Returns:
            Tuple of (audio array, sample rate)

        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # Load with librosa (handles resampling and mono conversion)
        audio, sr = librosa.load(
            str(path),
            sr=self.target_sr,
            mono=self.mono,
        )

        if self.normalize:
            audio = self._normalize(audio)

        return audio, sr

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range using peak normalization."""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        return audio

    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: int = 30,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Trim leading and trailing silence.

        Args:
            audio: Audio array
            top_db: Threshold in dB below peak to consider as silence

        Returns:
            Tuple of (trimmed audio, (start_idx, end_idx))
        """
        trimmed, indices = librosa.effects.trim(audio, top_db=top_db)
        return trimmed, indices

    def get_duration(self, audio: np.ndarray, sr: Optional[int] = None) -> float:
        """Get duration in seconds."""
        sr = sr or self.target_sr
        return len(audio) / sr
