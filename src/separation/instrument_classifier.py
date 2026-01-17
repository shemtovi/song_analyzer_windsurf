"""Instrument classification for separated stems and audio.

Classifies the instrument type within audio stems to help
with appropriate transcription settings and output labeling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class InstrumentCategory(Enum):
    """High-level instrument categories."""
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    GUITAR = "guitar"
    PIANO = "piano"
    KEYS = "keys"  # synths, organs, etc.
    STRINGS = "strings"
    WINDS = "winds"
    BRASS = "brass"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class InstrumentInfo:
    """Information about a detected instrument."""
    category: InstrumentCategory
    confidence: float
    subcategory: Optional[str] = None  # e.g., "electric_guitar", "acoustic_piano"
    pitch_range: Optional[Tuple[int, int]] = None  # MIDI pitch range
    is_melodic: bool = True  # False for drums/percussion
    is_monophonic: bool = False  # True for vocals, flute, etc.
    
    @property
    def transcription_mode(self) -> str:
        """Suggested transcription mode for this instrument."""
        if not self.is_melodic:
            return "percussion"
        elif self.is_monophonic:
            return "monophonic"
        else:
            return "polyphonic"


# Instrument characteristics for classification
INSTRUMENT_PROFILES = {
    InstrumentCategory.DRUMS: {
        "is_melodic": False,
        "is_monophonic": False,
        "pitch_range": None,
        "spectral_centroid_range": (1000, 8000),
        "spectral_flatness_range": (0.1, 0.9),
    },
    InstrumentCategory.BASS: {
        "is_melodic": True,
        "is_monophonic": True,
        "pitch_range": (28, 67),  # E1 to G4
        "spectral_centroid_range": (50, 500),
        "spectral_flatness_range": (0.0, 0.3),
    },
    InstrumentCategory.VOCALS: {
        "is_melodic": True,
        "is_monophonic": True,
        "pitch_range": (40, 84),  # E2 to C6
        "spectral_centroid_range": (200, 4000),
        "spectral_flatness_range": (0.0, 0.2),
    },
    InstrumentCategory.GUITAR: {
        "is_melodic": True,
        "is_monophonic": False,
        "pitch_range": (40, 88),  # E2 to E6
        "spectral_centroid_range": (200, 3000),
        "spectral_flatness_range": (0.0, 0.4),
    },
    InstrumentCategory.PIANO: {
        "is_melodic": True,
        "is_monophonic": False,
        "pitch_range": (21, 108),  # A0 to C8
        "spectral_centroid_range": (200, 4000),
        "spectral_flatness_range": (0.0, 0.3),
    },
    InstrumentCategory.KEYS: {
        "is_melodic": True,
        "is_monophonic": False,
        "pitch_range": (24, 108),
        "spectral_centroid_range": (100, 5000),
        "spectral_flatness_range": (0.0, 0.5),
    },
    InstrumentCategory.STRINGS: {
        "is_melodic": True,
        "is_monophonic": False,
        "pitch_range": (28, 96),  # E1 to C7
        "spectral_centroid_range": (200, 3000),
        "spectral_flatness_range": (0.0, 0.2),
    },
    InstrumentCategory.WINDS: {
        "is_melodic": True,
        "is_monophonic": True,
        "pitch_range": (36, 96),  # C2 to C7
        "spectral_centroid_range": (200, 4000),
        "spectral_flatness_range": (0.0, 0.3),
    },
    InstrumentCategory.BRASS: {
        "is_melodic": True,
        "is_monophonic": True,
        "pitch_range": (34, 84),  # Bb1 to C6
        "spectral_centroid_range": (100, 3000),
        "spectral_flatness_range": (0.0, 0.2),
    },
}


class InstrumentClassifier:
    """
    Classify instruments in audio using spectral features.
    
    This is a lightweight classifier based on spectral analysis.
    For higher accuracy, a neural network classifier could be used.
    """
    
    def __init__(
        self,
        hop_length: int = 512,
        n_fft: int = 2048,
    ):
        """
        Initialize InstrumentClassifier.
        
        Args:
            hop_length: Hop length for spectral analysis
            n_fft: FFT size for spectral analysis
        """
        self.hop_length = hop_length
        self.n_fft = n_fft
    
    def classify(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stem_hint: Optional[str] = None,
    ) -> InstrumentInfo:
        """
        Classify the instrument in an audio signal.
        
        Args:
            audio: Audio array (mono)
            sample_rate: Sample rate
            stem_hint: Optional hint from source separation (e.g., "bass", "vocals")
            
        Returns:
            InstrumentInfo with classification results
        """
        import librosa
        
        # If we have a stem hint, use it as primary classification
        if stem_hint:
            return self._classify_from_hint(stem_hint, audio, sample_rate)
        
        # Compute spectral features
        features = self._extract_features(audio, sample_rate)
        
        # Score each instrument category
        scores = {}
        for category, profile in INSTRUMENT_PROFILES.items():
            score = self._compute_category_score(features, profile)
            scores[category] = score
        
        # Get best match
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        # Normalize confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5
        
        profile = INSTRUMENT_PROFILES.get(best_category, {})
        
        return InstrumentInfo(
            category=best_category,
            confidence=confidence,
            pitch_range=profile.get("pitch_range"),
            is_melodic=profile.get("is_melodic", True),
            is_monophonic=profile.get("is_monophonic", False),
        )
    
    def _classify_from_hint(
        self,
        stem_hint: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> InstrumentInfo:
        """Classify using stem hint as primary signal."""
        # Map stem hints to instrument categories
        # Supports both 4-stem (htdemucs) and 6-stem (htdemucs_6s) models
        hint_map = {
            "drums": InstrumentCategory.DRUMS,
            "bass": InstrumentCategory.BASS,
            "vocals": InstrumentCategory.VOCALS,
            "guitar": InstrumentCategory.GUITAR,  # 6-stem model
            "piano": InstrumentCategory.PIANO,    # 6-stem model
            "other": InstrumentCategory.OTHER,
        }
        
        category = hint_map.get(stem_hint.lower(), InstrumentCategory.UNKNOWN)
        
        # For "other" stem, try to determine more specific category
        if category == InstrumentCategory.OTHER:
            # Analyze to determine if it's more likely guitar, piano, keys, etc.
            category = self._classify_other_stem(audio, sample_rate)
        
        profile = INSTRUMENT_PROFILES.get(category, {})
        
        # Higher confidence for specific stem types from 6-stem model
        if stem_hint.lower() in ("guitar", "piano"):
            confidence = 0.9
        elif category != InstrumentCategory.OTHER:
            confidence = 0.8
        else:
            confidence = 0.5
        
        return InstrumentInfo(
            category=category,
            confidence=confidence,
            pitch_range=profile.get("pitch_range"),
            is_melodic=profile.get("is_melodic", True),
            is_monophonic=profile.get("is_monophonic", False),
        )
    
    def _classify_other_stem(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> InstrumentCategory:
        """Try to classify the 'other' stem more specifically."""
        import librosa
        
        features = self._extract_features(audio, sample_rate)
        
        # Check for guitar vs piano vs keys characteristics
        candidates = [
            InstrumentCategory.GUITAR,
            InstrumentCategory.PIANO,
            InstrumentCategory.KEYS,
            InstrumentCategory.STRINGS,
        ]
        
        scores = {}
        for category in candidates:
            profile = INSTRUMENT_PROFILES[category]
            score = self._compute_category_score(features, profile)
            scores[category] = score
        
        if not scores:
            return InstrumentCategory.OTHER
        
        best = max(scores, key=scores.get)
        
        # Only return specific category if confidence is reasonable
        if scores[best] > 0.3:
            return best
        
        return InstrumentCategory.OTHER
    
    def _extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract spectral features for classification."""
        import librosa
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        
        # Skip if too short
        if len(audio) < self.n_fft:
            return {
                "spectral_centroid": 1000,
                "spectral_flatness": 0.5,
                "spectral_bandwidth": 1000,
                "zero_crossing_rate": 0.1,
                "rms": 0.1,
            }
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spectral flatness (noisiness)
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Zero crossing rate (percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        
        return {
            "spectral_centroid": float(np.mean(centroid)),
            "spectral_flatness": float(np.mean(flatness)),
            "spectral_bandwidth": float(np.mean(bandwidth)),
            "zero_crossing_rate": float(np.mean(zcr)),
            "rms": float(np.mean(rms)),
        }
    
    def _compute_category_score(
        self,
        features: Dict[str, float],
        profile: Dict,
    ) -> float:
        """Compute how well features match an instrument profile."""
        score = 0.0
        count = 0
        
        # Check spectral centroid
        if "spectral_centroid_range" in profile:
            low, high = profile["spectral_centroid_range"]
            centroid = features.get("spectral_centroid", 1000)
            if low <= centroid <= high:
                score += 1.0
            elif centroid < low:
                score += max(0, 1 - (low - centroid) / low)
            else:
                score += max(0, 1 - (centroid - high) / high)
            count += 1
        
        # Check spectral flatness
        if "spectral_flatness_range" in profile:
            low, high = profile["spectral_flatness_range"]
            flatness = features.get("spectral_flatness", 0.5)
            if low <= flatness <= high:
                score += 1.0
            else:
                score += 0.5
            count += 1
        
        return score / count if count > 0 else 0.5
    
    def classify_batch(
        self,
        stems: Dict[str, np.ndarray],
        sample_rate: int,
    ) -> Dict[str, InstrumentInfo]:
        """
        Classify multiple stems at once.
        
        Args:
            stems: Dictionary mapping stem names to audio arrays
            sample_rate: Sample rate
            
        Returns:
            Dictionary mapping stem names to InstrumentInfo
        """
        results = {}
        for name, audio in stems.items():
            results[name] = self.classify(audio, sample_rate, stem_hint=name)
        return results
    
    def get_transcription_config(
        self,
        instrument_info: InstrumentInfo,
    ) -> Dict:
        """
        Get recommended transcription configuration for an instrument.
        
        Args:
            instrument_info: Classified instrument info
            
        Returns:
            Dictionary with transcription configuration
        """
        config = {
            "mode": instrument_info.transcription_mode,
            "min_pitch": None,
            "max_pitch": None,
        }
        
        if instrument_info.pitch_range:
            config["min_pitch"] = instrument_info.pitch_range[0]
            config["max_pitch"] = instrument_info.pitch_range[1]
        
        # Instrument-specific settings
        if instrument_info.category == InstrumentCategory.DRUMS:
            config["detect_drums"] = True
            config["detect_pitch"] = False
        elif instrument_info.category == InstrumentCategory.BASS:
            config["onset_sensitivity"] = 0.7
            config["min_note_duration"] = 0.1
        elif instrument_info.category == InstrumentCategory.VOCALS:
            config["pitch_continuity"] = True
            config["vibrato_tolerance"] = True
        elif instrument_info.category == InstrumentCategory.PIANO:
            config["velocity_sensitive"] = True
            config["sustain_detection"] = True
        
        return config
