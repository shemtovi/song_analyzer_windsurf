"""Song Analyzer - Audio to Sheet Music Transcription System.

Architecture Layers:
    1. input/       - Audio loading and preprocessing
    2. analysis/    - Low-level signal analysis (features, tempo, pitch)
    3. transcription/ - Note-level detection (mono/poly/multi-instrument)
    4. separation/  - Source separation for multi-instrument (Demucs)
    5. inference/   - Musical understanding (key, chords, melody, harmony, structure)
    6. processing/  - Note post-processing (quantize, cleanup)
    7. output/      - Export (MIDI, MusicXML, sheet)
"""

__version__ = "0.2.0"

# Core types
from .core import Note

# Input layer
from .input import AudioLoader

# Analysis layer
from .analysis import FeatureExtractor, TempoAnalyzer, PitchAnalyzer

# Transcription layer
from .transcription import (
    MonophonicTranscriber,
    PolyphonicTranscriber,
    MultiInstrumentTranscriber,
)

# Separation layer (Phase 3)
from .separation import (
    SourceSeparator,
    StemType,
    InstrumentClassifier,
)

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
    "MultiInstrumentTranscriber",
    # Separation
    "SourceSeparator",
    "StemType",
    "InstrumentClassifier",
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
