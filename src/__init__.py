"""Song Analyzer - Audio to Sheet Music Transcription System.

Architecture Layers:
    1. input/       - Audio loading and preprocessing
    2. analysis/    - Low-level signal analysis (features, tempo, pitch)
    3. transcription/ - Note-level detection (mono/poly)
    4. inference/   - Musical understanding (key, chords, melody, harmony, structure)
    5. processing/  - Note post-processing (quantize, cleanup)
    6. output/      - Export (MIDI, MusicXML, sheet)
"""

__version__ = "0.1.0"

# Core types
from .core import Note

# Input layer
from .input import AudioLoader

# Analysis layer
from .analysis import FeatureExtractor, TempoAnalyzer, PitchAnalyzer

# Transcription layer
from .transcription import MonophonicTranscriber, PolyphonicTranscriber

# Inference layer
from .inference import (
    KeyDetector,
    ChordAnalyzer,
    MelodyAnalyzer,
    HarmonyAnalyzer,
    StructureAnalyzer,
)

# Processing layer
from .processing import Quantizer, NoteCleanup

# Output layer
from .output import MIDIExporter, MusicXMLExporter

__all__ = [
    # Core
    "Note",
    # Input
    "AudioLoader",
    # Analysis
    "FeatureExtractor",
    "TempoAnalyzer",
    "PitchAnalyzer",
    # Transcription
    "MonophonicTranscriber",
    "PolyphonicTranscriber",
    # Inference
    "KeyDetector",
    "ChordAnalyzer",
    "MelodyAnalyzer",
    "HarmonyAnalyzer",
    "StructureAnalyzer",
    # Processing
    "Quantizer",
    "NoteCleanup",
    # Output
    "MIDIExporter",
    "MusicXMLExporter",
]
