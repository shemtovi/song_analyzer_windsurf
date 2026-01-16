"""Feature extraction for audio analysis."""

import numpy as np
import librosa
from typing import Optional


class FeatureExtractor:
    """Extracts audio features for transcription."""

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 30.0,
        fmax: Optional[float] = 8000.0,
    ):
        """
        Initialize FeatureExtractor.

        Args:
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Samples between frames
            n_mels: Number of Mel bands
            fmin: Minimum frequency for Mel filterbank
            fmax: Maximum frequency for Mel filterbank
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def mel_spectrogram(
        self,
        audio: np.ndarray,
        log_scale: bool = True,
    ) -> np.ndarray:
        """
        Compute Mel spectrogram.

        Args:
            audio: Audio array
            log_scale: Apply log compression if True

        Returns:
            Mel spectrogram [n_mels, time_frames]
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        if log_scale:
            mel = librosa.power_to_db(mel, ref=np.max)

        return mel

    def stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.

        Returns:
            Complex STFT matrix [freq_bins, time_frames]
        """
        return librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def cqt(
        self,
        audio: np.ndarray,
        n_bins: int = 84,
        bins_per_octave: int = 12,
    ) -> np.ndarray:
        """
        Compute Constant-Q Transform.

        Args:
            audio: Audio array
            n_bins: Number of frequency bins (84 = 7 octaves)
            bins_per_octave: Bins per octave (12 = semitone resolution)

        Returns:
            CQT magnitude [n_bins, time_frames]
        """
        cqt = librosa.cqt(
            audio,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
        return np.abs(cqt)

    def chromagram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute chromagram (12 pitch classes).

        Returns:
            Chromagram [12, time_frames]
        """
        return librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def frames_to_time(self, frames: np.ndarray) -> np.ndarray:
        """Convert frame indices to time in seconds."""
        return librosa.frames_to_time(
            frames,
            sr=self.sr,
            hop_length=self.hop_length,
        )

    def time_to_frames(self, times: np.ndarray) -> np.ndarray:
        """Convert time in seconds to frame indices."""
        return librosa.time_to_frames(
            times,
            sr=self.sr,
            hop_length=self.hop_length,
        )
