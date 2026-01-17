"""Pitch analysis utilities."""

import numpy as np
import librosa
from typing import Tuple, Optional


class PitchAnalyzer:
    """Low-level pitch detection and analysis."""

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = 65.0,  # C2
        fmax: float = 2093.0,  # C7
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def detect_f0(
        self,
        audio: np.ndarray,
        method: str = "pyin",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect fundamental frequency (f0) over time.

        Args:
            audio: Audio array
            method: Detection method ('pyin', 'yin')

        Returns:
            Tuple of (f0 in Hz, voiced_flag, voiced_prob)
        """
        if method == "pyin":
            f0, voiced_flag, voiced_prob = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sr,
                hop_length=self.hop_length,
            )
        else:
            # YIN doesn't return voiced probability
            f0 = librosa.yin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sr,
                hop_length=self.hop_length,
            )
            voiced_flag = ~np.isnan(f0)
            voiced_prob = voiced_flag.astype(float)

        return f0, voiced_flag, voiced_prob

    def f0_to_midi(self, f0: np.ndarray) -> np.ndarray:
        """Convert f0 array to MIDI pitches."""
        with np.errstate(divide="ignore", invalid="ignore"):
            midi = 69 + 12 * np.log2(f0 / 440.0)
            midi = np.where(np.isfinite(midi), np.round(midi), np.nan)
        return midi

    def get_pitch_contour(
        self,
        audio: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get smoothed pitch contour.

        Returns:
            Tuple of (times, midi_pitches)
        """
        f0, voiced, _ = self.detect_f0(audio)
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sr,
            hop_length=self.hop_length,
        )
        midi = self.f0_to_midi(f0)
        return times, midi
