"""Base classes for transcription."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Note:
    """Represents a musical note."""

    pitch: int  # MIDI pitch (0-127)
    onset: float  # Start time in seconds
    offset: float  # End time in seconds
    velocity: int = 64  # MIDI velocity (0-127)
    instrument: Optional[str] = None

    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.offset - self.onset

    @property
    def pitch_name(self) -> str:
        """Get note name (e.g., 'C4', 'A#3')."""
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (self.pitch // 12) - 1
        name = names[self.pitch % 12]
        return f"{name}{octave}"

    @staticmethod
    def freq_to_midi(freq: float) -> int:
        """Convert frequency (Hz) to MIDI pitch."""
        if freq <= 0:
            return 0
        return int(round(69 + 12 * np.log2(freq / 440.0)))

    @staticmethod
    def midi_to_freq(midi: int) -> float:
        """Convert MIDI pitch to frequency (Hz)."""
        return 440.0 * (2 ** ((midi - 69) / 12.0))


class Transcriber(ABC):
    """Abstract base class for audio transcription."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Transcribe audio to notes.

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            List of detected notes
        """
        pass
