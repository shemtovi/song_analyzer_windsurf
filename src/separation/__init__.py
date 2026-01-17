"""Source separation module for multi-instrument transcription.

Uses Demucs to separate audio into stems:
- drums
- bass
- vocals
- other (guitars, keys, etc.)

This enables per-instrument transcription for polyphonic multi-instrument audio.
"""

from .demucs_separator import (
    SourceSeparator,
    SeparatedStems,
    StemType,
    StemAudio,
)
from .instrument_classifier import (
    InstrumentClassifier,
    InstrumentCategory,
    InstrumentInfo,
)

__all__ = [
    # Source separation
    "SourceSeparator",
    "SeparatedStems",
    "StemType",
    "StemAudio",
    # Instrument classification
    "InstrumentClassifier",
    "InstrumentCategory",
    "InstrumentInfo",
]
