"""Inference layer - Musical understanding and analysis.

This layer builds higher-level musical understanding from notes:
- Key detection (tonal center)
- Chord recognition and progression analysis
- Melody extraction and analysis
- Harmony analysis
- Song structure detection

Pipeline: Notes → [Key, Chords, Melody, Harmony] → Musical Structure
"""

from .key import KeyDetector
from .chords import ChordAnalyzer
from .melody import MelodyAnalyzer
from .harmony import HarmonyAnalyzer
from .structure import StructureAnalyzer

__all__ = [
    "KeyDetector",
    "ChordAnalyzer",
    "MelodyAnalyzer",
    "HarmonyAnalyzer",
    "StructureAnalyzer",
]
