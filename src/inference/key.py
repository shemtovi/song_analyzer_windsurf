"""Key detection - Identify the tonal center of a piece."""

import numpy as np
import librosa
from typing import List, Tuple
from dataclasses import dataclass

from ..core import Note, PITCH_NAMES


@dataclass
class KeyInfo:
    """Container for key detection results."""
    
    root: str  # Key root note (e.g., "C", "F#")
    mode: str  # "major" or "minor"
    confidence: float  # 0.0 - 1.0
    pitch_class_distribution: np.ndarray = None  # 12-element array


class KeyDetector:
    """Detect musical key from audio or notes."""

    # Krumhansl-Schmuckler key profiles
    MAJOR_PROFILE = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    MINOR_PROFILE = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    def detect_from_audio(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[str, str, float]:
        """
        Detect key from audio using chromagram.

        Returns:
            Tuple of (key name, mode, correlation score)
        """
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        return self._find_best_key(chroma_mean)

    def detect_from_notes(self, notes: List[Note]) -> Tuple[str, str, float]:
        """
        Detect key from note list.

        Returns:
            Tuple of (key name, mode, correlation score)
        """
        # Build pitch class histogram
        pitch_classes = np.zeros(12)
        for note in notes:
            pc = note.pitch % 12
            pitch_classes[pc] += note.duration

        # Normalize
        if pitch_classes.sum() > 0:
            pitch_classes /= pitch_classes.sum()

        return self._find_best_key(pitch_classes)

    def analyze(self, notes: List[Note]) -> KeyInfo:
        """
        Perform full key analysis.

        Args:
            notes: List of notes

        Returns:
            KeyInfo with detailed analysis
        """
        # Build pitch class histogram
        pitch_classes = np.zeros(12)
        for note in notes:
            pc = note.pitch % 12
            pitch_classes[pc] += note.duration

        # Normalize
        if pitch_classes.sum() > 0:
            pitch_classes /= pitch_classes.sum()

        key, mode, confidence = self._find_best_key(pitch_classes)
        
        return KeyInfo(
            root=key,
            mode=mode,
            confidence=confidence,
            pitch_class_distribution=pitch_classes,
        )

    def _find_best_key(
        self, pitch_classes: np.ndarray
    ) -> Tuple[str, str, float]:
        """Find best matching key using correlation."""
        best_corr = -1
        best_key = "C"
        best_mode = "major"

        for shift in range(12):
            # Rotate pitch classes
            rotated = np.roll(pitch_classes, -shift)

            # Correlate with major profile
            major_corr = np.corrcoef(rotated, self.MAJOR_PROFILE)[0, 1]
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = PITCH_NAMES[shift]
                best_mode = "major"

            # Correlate with minor profile
            minor_corr = np.corrcoef(rotated, self.MINOR_PROFILE)[0, 1]
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = PITCH_NAMES[shift]
                best_mode = "minor"

        return best_key, best_mode, best_corr
