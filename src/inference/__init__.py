"""Inference layer - Musical understanding and analysis.

This layer builds higher-level musical understanding from notes:
- Key detection (tonal center)
- Chord recognition and progression analysis
- Melody extraction and analysis
- Harmony analysis (integrated key + chords + melody)
- Song structure detection
- Resilience handling for noisy/imperfect input

Pipeline: Notes → [Key, Chords, Melody, Harmony] → Musical Structure
"""

from .key import KeyDetector, KeyInfo, KeyCandidate
from .chords import ChordAnalyzer, Chord, ChordProgression
from .melody import MelodyAnalyzer
from .harmony import HarmonyAnalyzer, HarmonyInfo, HarmonicSegment
from .structure import StructureAnalyzer
from .resilience import (
    ResilienceProcessor,
    NoteFilter,
    TimingCorrector,
    PitchCorrector,
    ConfidenceWeighter,
    apply_musical_rules,
)

__all__ = [
    # Key detection
    "KeyDetector",
    "KeyInfo",
    "KeyCandidate",
    # Chord analysis
    "ChordAnalyzer",
    "Chord",
    "ChordProgression",
    # Melody analysis
    "MelodyAnalyzer",
    # Harmony analysis
    "HarmonyAnalyzer",
    "HarmonyInfo",
    "HarmonicSegment",
    # Structure analysis
    "StructureAnalyzer",
    # Resilience
    "ResilienceProcessor",
    "NoteFilter",
    "TimingCorrector",
    "PitchCorrector",
    "ConfidenceWeighter",
    "apply_musical_rules",
]
