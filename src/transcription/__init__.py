"""Transcription layer - Note-level detection from audio.

This layer converts audio signals into discrete note events:
- Monophonic transcription (single melody line)
- Polyphonic transcription (multiple simultaneous notes)
"""

from .base import Transcriber
from .monophonic import MonophonicTranscriber
from .polyphonic import PolyphonicTranscriber

__all__ = [
    "Transcriber",
    "MonophonicTranscriber",
    "PolyphonicTranscriber",
]
