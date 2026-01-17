"""Note data class - the fundamental unit of musical transcription."""

from dataclasses import dataclass
from typing import Optional
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

    @property
    def pitch_class(self) -> int:
        """Get pitch class (0-11, where 0=C)."""
        return self.pitch % 12

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
