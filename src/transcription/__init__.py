"""Transcription layer - Note-level detection from audio.

This layer converts audio signals into discrete note events:
- Monophonic transcription (single melody line)
- Polyphonic transcription (multiple simultaneous notes)
- Multi-instrument transcription (source separation + per-stem)
"""

from .base import Transcriber
from .monophonic import MonophonicTranscriber
from .polyphonic import PolyphonicTranscriber
from .multi_instrument import (
    MultiInstrumentTranscriber,
    MultiInstrumentTranscription,
    StemTranscription,
)

__all__ = [
    "Transcriber",
    "MonophonicTranscriber",
    "PolyphonicTranscriber",
    "MultiInstrumentTranscriber",
    "MultiInstrumentTranscription",
    "StemTranscription",
]
