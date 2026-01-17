"""Key detection - Identify the tonal center of a piece.

Implements robust key detection with:
- Krumhansl-Schmuckler key profiles
- Temperley key profiles (alternative weighting)
- Ambiguity detection (relative major/minor, parallel keys)
- Confidence scoring with multiple metrics
- Tolerance for noisy/imperfect input
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

from ..core import Note, PITCH_NAMES


class Mode(Enum):
    """Musical modes."""
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"


@dataclass
class KeyCandidate:
    """A candidate key with its score."""
    root: str
    mode: str
    correlation: float
    
    @property
    def name(self) -> str:
        return f"{self.root} {self.mode}"


@dataclass
class KeyInfo:
    """Container for key detection results."""
    
    root: str  # Key root note (e.g., "C", "F#")
    mode: str  # "major", "minor", or mode name
    confidence: float  # 0.0 - 1.0
    pitch_class_distribution: np.ndarray = None  # 12-element array
    alternatives: List[KeyCandidate] = field(default_factory=list)  # Other likely keys
    ambiguity_score: float = 0.0  # How ambiguous the detection is (0=clear, 1=very ambiguous)
    relative_key: Optional[str] = None  # Relative major/minor if applicable
    parallel_key: Optional[str] = None  # Parallel major/minor


class KeyDetector:
    """Detect musical key from audio or notes.
    
    Features:
    - Multiple key profile algorithms (Krumhansl-Schmuckler, Temperley)
    - Ambiguity detection for closely-scoring keys
    - Noise tolerance through weighted analysis
    - Support for modes beyond major/minor
    """

    # Krumhansl-Schmuckler key profiles (cognitive-based)
    KRUMHANSL_MAJOR = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    KRUMHANSL_MINOR = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    
    # Temperley key profiles (corpus-based, often more accurate for pop/rock)
    TEMPERLEY_MAJOR = np.array(
        [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
    )
    TEMPERLEY_MINOR = np.array(
        [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    )
    
    # Modal profiles (derived from scale degrees)
    MODE_PROFILES = {
        "dorian": np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 2.0, 3.5, 4.0, 2.0]),
        "phrygian": np.array([5.0, 4.0, 2.0, 3.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 4.0, 2.0]),
        "lydian": np.array([5.0, 2.0, 3.5, 2.0, 4.5, 2.0, 4.0, 4.5, 2.0, 3.5, 2.0, 4.0]),
        "mixolydian": np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 4.0, 2.0]),
        "locrian": np.array([5.0, 4.0, 2.0, 3.5, 2.0, 4.0, 4.0, 2.0, 3.5, 2.0, 4.0, 2.0]),
    }
    
    # Aliases for backward compatibility
    MAJOR_PROFILE = KRUMHANSL_MAJOR
    MINOR_PROFILE = KRUMHANSL_MINOR
    
    # Ambiguity threshold - if second best key is within this correlation of best, flag as ambiguous
    AMBIGUITY_THRESHOLD = 0.05
    
    # Minimum confidence to consider a key detection valid
    MIN_CONFIDENCE = 0.3

    def __init__(
        self,
        profile_type: str = "krumhansl",
        detect_modes: bool = False,
        ambiguity_threshold: float = 0.05,
        min_notes: int = 3,
    ):
        """
        Initialize KeyDetector.
        
        Args:
            profile_type: Key profile algorithm ("krumhansl" or "temperley")
            detect_modes: Whether to detect modes beyond major/minor
            ambiguity_threshold: Correlation difference to flag ambiguity
            min_notes: Minimum notes required for valid detection
        """
        self.profile_type = profile_type
        self.detect_modes = detect_modes
        self.ambiguity_threshold = ambiguity_threshold
        self.min_notes = min_notes
        
        # Select key profiles based on type
        if profile_type == "temperley":
            self.major_profile = self.TEMPERLEY_MAJOR
            self.minor_profile = self.TEMPERLEY_MINOR
        else:
            self.major_profile = self.KRUMHANSL_MAJOR
            self.minor_profile = self.KRUMHANSL_MINOR

    def detect_from_audio(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[str, str, float]:
        """
        Detect key from audio using chromagram.

        Returns:
            Tuple of (key name, mode, correlation score)
        """
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Normalize
        if chroma_mean.sum() > 0:
            chroma_mean = chroma_mean / chroma_mean.sum()

        return self._find_best_key(chroma_mean)

    def detect_from_notes(
        self, 
        notes: List[Note],
        weighted: bool = True,
    ) -> Tuple[str, str, float]:
        """
        Detect key from note list.
        
        Args:
            notes: List of Note objects
            weighted: Weight by duration and velocity

        Returns:
            Tuple of (key name, mode, correlation score)
        """
        pitch_classes = self._build_pitch_class_distribution(notes, weighted)
        return self._find_best_key(pitch_classes)
    
    def _build_pitch_class_distribution(
        self,
        notes: List[Note],
        weighted: bool = True,
    ) -> np.ndarray:
        """
        Build a pitch class distribution from notes.
        
        Args:
            notes: List of notes
            weighted: Weight by duration and velocity
            
        Returns:
            12-element numpy array of pitch class weights
        """
        pitch_classes = np.zeros(12)
        
        for note in notes:
            pc = note.pitch % 12
            if weighted:
                # Weight by duration and velocity
                weight = note.duration * (note.velocity / 127.0)
            else:
                weight = 1.0
            pitch_classes[pc] += weight

        # Normalize
        if pitch_classes.sum() > 0:
            pitch_classes /= pitch_classes.sum()

        return pitch_classes

    def analyze(
        self, 
        notes: List[Note],
        return_alternatives: bool = True,
    ) -> KeyInfo:
        """
        Perform full key analysis with ambiguity detection.

        Args:
            notes: List of notes
            return_alternatives: Include alternative key candidates

        Returns:
            KeyInfo with detailed analysis including alternatives and ambiguity
        """
        if len(notes) < self.min_notes:
            return KeyInfo(
                root="C",
                mode="major",
                confidence=0.0,
                pitch_class_distribution=np.zeros(12),
                ambiguity_score=1.0,
            )
        
        # Build pitch class histogram with weighting
        pitch_classes = self._build_pitch_class_distribution(notes, weighted=True)
        
        # Get all key candidates with scores
        candidates = self._get_all_candidates(pitch_classes)
        
        if not candidates:
            return KeyInfo(
                root="C",
                mode="major", 
                confidence=0.0,
                pitch_class_distribution=pitch_classes,
                ambiguity_score=1.0,
            )
        
        # Sort by correlation (descending)
        candidates.sort(key=lambda c: c.correlation, reverse=True)
        
        best = candidates[0]
        
        # Calculate confidence (scale correlation to 0-1 range)
        # Correlation can be negative to 1, map to 0-1
        confidence = max(0.0, min(1.0, (best.correlation + 1) / 2))
        
        # Calculate ambiguity score
        ambiguity_score = self._calculate_ambiguity(candidates)
        
        # Get relative and parallel keys
        relative_key = self._get_relative_key(best.root, best.mode)
        parallel_key = self._get_parallel_key(best.root, best.mode)
        
        # Get alternatives (top 3 after best)
        alternatives = candidates[1:4] if return_alternatives else []
        
        return KeyInfo(
            root=best.root,
            mode=best.mode,
            confidence=confidence,
            pitch_class_distribution=pitch_classes,
            alternatives=alternatives,
            ambiguity_score=ambiguity_score,
            relative_key=relative_key,
            parallel_key=parallel_key,
        )
    
    def _get_all_candidates(self, pitch_classes: np.ndarray) -> List[KeyCandidate]:
        """
        Get all key candidates with their correlation scores.
        
        Args:
            pitch_classes: 12-element pitch class distribution
            
        Returns:
            List of KeyCandidate objects
        """
        candidates = []
        
        for shift in range(12):
            root = PITCH_NAMES[shift]
            rotated = np.roll(pitch_classes, -shift)
            
            # Test major
            major_corr = self._correlate(rotated, self.major_profile)
            candidates.append(KeyCandidate(root, "major", major_corr))
            
            # Test minor
            minor_corr = self._correlate(rotated, self.minor_profile)
            candidates.append(KeyCandidate(root, "minor", minor_corr))
            
            # Test modes if enabled
            if self.detect_modes:
                for mode_name, profile in self.MODE_PROFILES.items():
                    mode_corr = self._correlate(rotated, profile)
                    candidates.append(KeyCandidate(root, mode_name, mode_corr))
        
        return candidates
    
    def _correlate(self, distribution: np.ndarray, profile: np.ndarray) -> float:
        """
        Calculate correlation between distribution and profile.
        
        Handles edge cases gracefully for noisy input.
        """
        if distribution.std() == 0 or profile.std() == 0:
            return 0.0
        
        # Use Pearson correlation
        corr = np.corrcoef(distribution, profile)[0, 1]
        
        # Handle NaN (can occur with degenerate input)
        if np.isnan(corr):
            return 0.0
            
        return float(corr)
    
    def _calculate_ambiguity(self, candidates: List[KeyCandidate]) -> float:
        """
        Calculate how ambiguous the key detection is.
        
        High ambiguity when multiple keys have similar correlation scores.
        
        Returns:
            Ambiguity score 0.0 (clear) to 1.0 (very ambiguous)
        """
        if len(candidates) < 2:
            return 0.0
        
        # Check how many keys are within threshold of best
        best_corr = candidates[0].correlation
        close_candidates = [
            c for c in candidates[1:]
            if abs(c.correlation - best_corr) < self.ambiguity_threshold
        ]
        
        # Also consider relative major/minor specifically
        relative_ambiguity = self._check_relative_ambiguity(candidates)
        
        # Combine: more close candidates = more ambiguous
        candidate_ambiguity = min(1.0, len(close_candidates) / 3.0)
        
        return max(candidate_ambiguity, relative_ambiguity)
    
    def _check_relative_ambiguity(self, candidates: List[KeyCandidate]) -> float:
        """
        Check if relative major/minor are both strong candidates.
        """
        if len(candidates) < 2:
            return 0.0
        
        best = candidates[0]
        relative = self._get_relative_key(best.root, best.mode)
        
        if relative:
            # Find the relative key in candidates
            rel_root, rel_mode = relative.split(" ")
            for c in candidates[1:6]:  # Check top 6
                if c.root == rel_root and c.mode == rel_mode:
                    # How close is the relative key?
                    diff = abs(best.correlation - c.correlation)
                    if diff < self.ambiguity_threshold:
                        return 0.8  # Relative key is very close
                    elif diff < self.ambiguity_threshold * 2:
                        return 0.5  # Relative key is somewhat close
        
        return 0.0
    
    def _get_relative_key(self, root: str, mode: str) -> Optional[str]:
        """
        Get the relative major/minor key.
        
        Relative minor is 3 semitones down from major.
        Relative major is 3 semitones up from minor.
        """
        root_idx = PITCH_NAMES.index(root)
        
        if mode == "major":
            # Relative minor is 3 semitones down
            rel_idx = (root_idx - 3) % 12
            return f"{PITCH_NAMES[rel_idx]} minor"
        elif mode == "minor":
            # Relative major is 3 semitones up
            rel_idx = (root_idx + 3) % 12
            return f"{PITCH_NAMES[rel_idx]} major"
        
        return None
    
    def _get_parallel_key(self, root: str, mode: str) -> Optional[str]:
        """
        Get the parallel major/minor key (same root, different mode).
        """
        if mode == "major":
            return f"{root} minor"
        elif mode == "minor":
            return f"{root} major"
        return None

    def _find_best_key(
        self, pitch_classes: np.ndarray
    ) -> Tuple[str, str, float]:
        """
        Find best matching key using correlation.
        
        This is the simplified interface returning just the top result.
        """
        candidates = self._get_all_candidates(pitch_classes)
        
        if not candidates:
            return "C", "major", 0.0
        
        best = max(candidates, key=lambda c: c.correlation)
        return best.root, best.mode, best.correlation
    
    def detect_key_segments(
        self,
        notes: List[Note],
        segment_duration: float = 4.0,
        overlap: float = 0.5,
    ) -> List[Tuple[float, float, KeyInfo]]:
        """
        Detect key changes over time by analyzing segments.
        
        Useful for pieces with modulations.
        
        Args:
            notes: List of all notes
            segment_duration: Duration of each analysis segment (seconds)
            overlap: Overlap between segments (0-1)
            
        Returns:
            List of (start_time, end_time, KeyInfo) tuples
        """
        if not notes:
            return []
        
        min_time = min(n.onset for n in notes)
        max_time = max(n.offset for n in notes)
        
        step = segment_duration * (1 - overlap)
        segments = []
        
        current_time = min_time
        while current_time < max_time:
            end_time = min(current_time + segment_duration, max_time)
            
            # Get notes in this segment
            segment_notes = [
                n for n in notes
                if n.onset < end_time and n.offset > current_time
            ]
            
            if len(segment_notes) >= self.min_notes:
                key_info = self.analyze(segment_notes)
                segments.append((current_time, end_time, key_info))
            
            current_time += step
        
        return segments
    
    def get_scale_notes(self, root: str, mode: str) -> List[int]:
        """
        Get the pitch classes (0-11) that belong to a scale.
        
        Args:
            root: Root note (e.g., "C", "G")
            mode: Scale mode ("major", "minor", etc.)
            
        Returns:
            List of pitch classes in the scale
        """
        # Scale intervals from root
        SCALE_INTERVALS = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
            "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
            "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "lydian": [0, 2, 4, 6, 7, 9, 11],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "locrian": [0, 1, 3, 5, 6, 8, 10],
        }
        
        root_idx = PITCH_NAMES.index(root)
        intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
        
        return [(root_idx + interval) % 12 for interval in intervals]
