"""Chord analysis - Detect and analyze chord progressions.

Implements robust chord detection with:
- Template matching for chord quality identification
- Probabilistic scoring with key context
- Inversion detection via bass note analysis
- Tolerance for missing or extra notes
- Roman numeral analysis in key context
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict

from ..core import Note, PITCH_NAMES


@dataclass
class ChordCandidate:
    """A candidate chord with its confidence score."""
    root: str
    quality: str
    score: float
    matched_intervals: List[int] = field(default_factory=list)
    missing_intervals: List[int] = field(default_factory=list)
    extra_intervals: List[int] = field(default_factory=list)


@dataclass
class Chord:
    """Represents a detected chord."""
    
    root: str  # Root note (e.g., "C", "F#")
    quality: str  # "major", "minor", "dim", "aug", "7", "maj7", "min7", etc.
    onset: float  # Start time in seconds
    offset: float  # End time in seconds
    bass: Optional[str] = None  # Bass note if different from root (inversions)
    notes: List[int] = field(default_factory=list)  # MIDI pitches in the chord
    confidence: float = 1.0  # Detection confidence (0-1)
    inversion: int = 0  # 0=root position, 1=first inversion, 2=second inversion
    
    @property
    def duration(self) -> float:
        return self.offset - self.onset
    
    @property
    def pitch_classes(self) -> Set[int]:
        """Get pitch classes in the chord."""
        return {p % 12 for p in self.notes}
    
    @property
    def root_pitch_class(self) -> int:
        """Get root as pitch class (0-11)."""
        return PITCH_NAMES.index(self.root)
    
    @property
    def symbol(self) -> str:
        """Get chord symbol (e.g., 'Cmaj7', 'Am', 'G/B')."""
        quality_map = {
            "major": "",
            "minor": "m",
            "diminished": "dim",
            "augmented": "aug",
            "dominant7": "7",
            "major7": "maj7",
            "minor7": "m7",
            "diminished7": "dim7",
            "half_diminished7": "m7b5",
            "augmented7": "aug7",
            "sus2": "sus2",
            "sus4": "sus4",
            "add9": "add9",
            "6": "6",
            "minor6": "m6",
        }
        suffix = quality_map.get(self.quality, self.quality)
        symbol = f"{self.root}{suffix}"
        if self.bass and self.bass != self.root:
            symbol += f"/{self.bass}"
        return symbol
    
    def get_roman_numeral(self, key_root: str, key_mode: str) -> str:
        """
        Get roman numeral representation in given key.
        
        Args:
            key_root: Key root (e.g., "C")
            key_mode: Key mode ("major" or "minor")
            
        Returns:
            Roman numeral (e.g., "IV", "ii", "V7")
        """
        key_root_pc = PITCH_NAMES.index(key_root)
        chord_root_pc = self.root_pitch_class
        
        # Calculate interval from key root
        interval = (chord_root_pc - key_root_pc) % 12
        
        # Map interval to scale degree
        if key_mode == "major":
            degree_map = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
        else:  # minor
            degree_map = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7}
        
        degree = degree_map.get(interval, 0)
        if degree == 0:
            # Non-diatonic chord
            return f"({self.symbol})"
        
        # Roman numerals
        numerals = ["", "I", "II", "III", "IV", "V", "VI", "VII"]
        numeral = numerals[degree]
        
        # Lowercase for minor chords
        if self.quality in ["minor", "minor7", "diminished", "diminished7", "half_diminished7"]:
            numeral = numeral.lower()
        
        # Add quality suffix
        if "7" in self.quality:
            numeral += "7"
        elif self.quality == "diminished":
            numeral += "°"
        elif self.quality == "augmented":
            numeral += "+"
        
        return numeral


@dataclass
class ChordProgression:
    """Container for chord progression analysis."""
    
    chords: List[Chord]
    key_root: Optional[str] = None
    key_mode: Optional[str] = None
    roman_numerals: List[str] = field(default_factory=list)  # e.g., ["I", "IV", "V", "I"]
    
    @property
    def key(self) -> Optional[str]:
        """Get key as string."""
        if self.key_root and self.key_mode:
            return f"{self.key_root} {self.key_mode}"
        return None
    
    @property
    def symbols(self) -> List[str]:
        """Get list of chord symbols."""
        return [c.symbol for c in self.chords]
    
    def get_common_progressions(self) -> List[Tuple[str, float]]:
        """
        Identify common chord progressions in the sequence.
        
        Returns:
            List of (progression_name, confidence) tuples
        """
        if not self.roman_numerals or len(self.roman_numerals) < 2:
            return []
        
        # Common progressions to look for
        COMMON_PROGRESSIONS = {
            "I-IV-V-I": ["I", "IV", "V", "I"],
            "I-V-vi-IV": ["I", "V", "vi", "IV"],
            "ii-V-I": ["ii", "V", "I"],
            "I-vi-IV-V": ["I", "vi", "IV", "V"],
            "I-IV-vi-V": ["I", "IV", "vi", "V"],
            "vi-IV-I-V": ["vi", "IV", "I", "V"],
            "I-V-vi-iii-IV": ["I", "V", "vi", "iii", "IV"],
            "I-IV": ["I", "IV"],
            "I-V": ["I", "V"],
            "IV-V-I": ["IV", "V", "I"],
        }
        
        found = []
        numerals_str = "-".join(self.roman_numerals)
        
        for name, pattern in COMMON_PROGRESSIONS.items():
            pattern_str = "-".join(pattern)
            if pattern_str in numerals_str:
                # Calculate how much of the progression this pattern covers
                coverage = len(pattern) / len(self.roman_numerals)
                found.append((name, min(1.0, coverage)))
        
        # Sort by confidence
        found.sort(key=lambda x: x[1], reverse=True)
        return found


class ChordAnalyzer:
    """Analyze chords and chord progressions from notes.
    
    Features:
    - Template matching for chord quality identification
    - Probabilistic scoring with tolerance for missing/extra notes
    - Key context integration for improved accuracy
    - Inversion detection
    - Roman numeral analysis
    """

    # Chord templates (intervals from root in semitones)
    # Ordered by priority (more common chords first)
    CHORD_TEMPLATES = {
        # Triads
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "diminished": [0, 3, 6],
        "augmented": [0, 4, 8],
        "sus4": [0, 5, 7],
        "sus2": [0, 2, 7],
        # Seventh chords
        "dominant7": [0, 4, 7, 10],
        "major7": [0, 4, 7, 11],
        "minor7": [0, 3, 7, 10],
        "diminished7": [0, 3, 6, 9],
        "half_diminished7": [0, 3, 6, 10],
        "augmented7": [0, 4, 8, 10],
        # Extended
        "add9": [0, 4, 7, 14],  # 14 = 2 + 12 (9th)
        "6": [0, 4, 7, 9],
        "minor6": [0, 3, 7, 9],
    }
    
    # Weights for chord template matching
    # Root and fifth are most important
    INTERVAL_WEIGHTS = {
        0: 2.0,   # Root - essential
        3: 1.5,   # Minor third
        4: 1.5,   # Major third
        5: 1.0,   # Perfect fourth (sus)
        6: 1.0,   # Tritone
        7: 1.8,   # Perfect fifth - very important
        8: 1.0,   # Augmented fifth
        9: 0.8,   # Sixth
        10: 0.9,  # Minor seventh
        11: 0.9,  # Major seventh
    }
    
    # Diatonic chord qualities for each scale degree in major/minor
    DIATONIC_CHORDS_MAJOR = {
        0: "major",    # I
        2: "minor",    # ii
        4: "minor",    # iii
        5: "major",    # IV
        7: "major",    # V
        9: "minor",    # vi
        11: "diminished",  # vii°
    }
    
    DIATONIC_CHORDS_MINOR = {
        0: "minor",    # i
        2: "diminished",  # ii°
        3: "major",    # III
        5: "minor",    # iv
        7: "minor",    # v (or major V in harmonic minor)
        8: "major",    # VI
        10: "major",   # VII
    }

    def __init__(
        self,
        min_chord_duration: float = 0.25,
        min_notes_for_chord: int = 2,
        tolerance_semitones: int = 0,
        use_key_context: bool = True,
        key_context_weight: float = 0.2,
    ):
        """
        Initialize ChordAnalyzer.

        Args:
            min_chord_duration: Minimum duration for a chord segment (seconds)
            min_notes_for_chord: Minimum simultaneous notes to form a chord
            tolerance_semitones: Allow pitch mismatches within this range
            use_key_context: Whether to use key context for chord scoring
            key_context_weight: How much to weight key context (0-1)
        """
        self.min_chord_duration = min_chord_duration
        self.min_notes_for_chord = min_notes_for_chord
        self.tolerance_semitones = tolerance_semitones
        self.use_key_context = use_key_context
        self.key_context_weight = key_context_weight

    def detect_chords(
        self,
        notes: List[Note],
        key_root: Optional[str] = None,
        key_mode: Optional[str] = None,
        segment_by: str = "onset",
    ) -> List[Chord]:
        """
        Detect chords from a list of notes.

        Args:
            notes: List of Note objects
            key_root: Optional key root for context (e.g., "C")
            key_mode: Optional key mode for context ("major" or "minor")
            segment_by: How to segment chords - "onset" (by note onsets) or 
                       "fixed" (fixed time windows)

        Returns:
            List of detected Chord objects
        """
        if not notes or len(notes) < self.min_notes_for_chord:
            return []
        
        # Segment notes into chord regions
        if segment_by == "onset":
            segments = self._segment_by_onset(notes)
        else:
            segments = self._segment_by_time(notes)
        
        chords = []
        prev_chord = None
        
        for onset, offset, segment_notes in segments:
            if len(segment_notes) < self.min_notes_for_chord:
                continue
            
            # Detect chord for this segment
            chord = self._detect_single_chord(
                segment_notes, onset, offset,
                key_root, key_mode, prev_chord
            )
            
            if chord:
                # Merge with previous if same chord
                if (prev_chord and 
                    prev_chord.root == chord.root and 
                    prev_chord.quality == chord.quality and
                    chord.onset - prev_chord.offset < 0.1):
                    # Extend previous chord
                    prev_chord.offset = chord.offset
                    prev_chord.notes = list(set(prev_chord.notes + chord.notes))
                else:
                    chords.append(chord)
                    prev_chord = chord
        
        return chords
    
    def _segment_by_onset(self, notes: List[Note]) -> List[Tuple[float, float, List[Note]]]:
        """
        Segment notes by onset times, grouping simultaneous notes.
        
        Returns:
            List of (onset, offset, notes) tuples
        """
        if not notes:
            return []
        
        # Group notes by onset time (with tolerance)
        onset_groups = defaultdict(list)
        tolerance = 0.05  # 50ms
        
        sorted_notes = sorted(notes, key=lambda n: n.onset)
        
        current_onset = sorted_notes[0].onset
        for note in sorted_notes:
            if note.onset - current_onset > tolerance:
                current_onset = note.onset
            onset_groups[current_onset].append(note)
        
        segments = []
        onsets = sorted(onset_groups.keys())
        
        for i, onset in enumerate(onsets):
            group_notes = onset_groups[onset]
            
            # Determine offset for this segment
            if i + 1 < len(onsets):
                offset = onsets[i + 1]
            else:
                offset = max(n.offset for n in group_notes)
            
            # Also include notes that are still sounding from previous onsets
            active_notes = [
                n for n in notes
                if n.onset <= onset and n.offset > onset
            ]
            
            if offset - onset >= self.min_chord_duration:
                segments.append((onset, offset, active_notes))
        
        return segments
    
    def _segment_by_time(self, notes: List[Note]) -> List[Tuple[float, float, List[Note]]]:
        """
        Segment notes by fixed time windows.
        """
        if not notes:
            return []
        
        min_time = min(n.onset for n in notes)
        max_time = max(n.offset for n in notes)
        
        segments = []
        current = min_time
        
        while current < max_time:
            end = current + self.min_chord_duration * 2
            
            active_notes = [
                n for n in notes
                if n.onset < end and n.offset > current
            ]
            
            if len(active_notes) >= self.min_notes_for_chord:
                segments.append((current, end, active_notes))
            
            current = end
        
        return segments
    
    def _detect_single_chord(
        self,
        notes: List[Note],
        onset: float,
        offset: float,
        key_root: Optional[str],
        key_mode: Optional[str],
        prev_chord: Optional[Chord],
    ) -> Optional[Chord]:
        """
        Detect a single chord from simultaneous notes.
        """
        # Get pitch classes present
        pitch_classes = self.notes_to_pitch_classes(notes)
        
        if len(pitch_classes) < self.min_notes_for_chord:
            return None
        
        # Find bass note (lowest pitch)
        bass_pitch = min(n.pitch for n in notes)
        bass_pc = bass_pitch % 12
        bass_name = PITCH_NAMES[bass_pc]
        
        # Try all possible roots and chord types
        candidates = self._get_chord_candidates(
            pitch_classes, key_root, key_mode, bass_pc, prev_chord
        )
        
        if not candidates:
            return None
        
        # Select best candidate
        best = max(candidates, key=lambda c: c.score)
        
        # Determine inversion
        inversion = 0
        if best.root != bass_name:
            root_pc = PITCH_NAMES.index(best.root)
            template = self.CHORD_TEMPLATES.get(best.quality, [0, 4, 7])
            if bass_pc == (root_pc + template[1]) % 12 if len(template) > 1 else False:
                inversion = 1
            elif bass_pc == (root_pc + template[2]) % 12 if len(template) > 2 else False:
                inversion = 2
        
        return Chord(
            root=best.root,
            quality=best.quality,
            onset=onset,
            offset=offset,
            bass=bass_name if bass_name != best.root else None,
            notes=[n.pitch for n in notes],
            confidence=min(1.0, best.score),
            inversion=inversion,
        )
    
    def _get_chord_candidates(
        self,
        pitch_classes: Set[int],
        key_root: Optional[str],
        key_mode: Optional[str],
        bass_pc: int,
        prev_chord: Optional[Chord],
    ) -> List[ChordCandidate]:
        """
        Get all chord candidates for a set of pitch classes.
        """
        candidates = []
        
        for root_pc in range(12):
            root_name = PITCH_NAMES[root_pc]
            
            for quality, template in self.CHORD_TEMPLATES.items():
                score, matched, missing, extra = self._score_chord_match(
                    pitch_classes, root_pc, template
                )
                
                if score > 0:
                    # Apply key context bonus
                    if self.use_key_context and key_root and key_mode:
                        key_bonus = self._get_key_context_bonus(
                            root_name, quality, key_root, key_mode
                        )
                        score += key_bonus * self.key_context_weight
                    
                    # Bonus if bass note is root (root position preferred)
                    if bass_pc == root_pc:
                        score += 0.1
                    
                    # Bonus for continuity with previous chord
                    if prev_chord:
                        continuity_bonus = self._get_continuity_bonus(
                            root_name, quality, prev_chord
                        )
                        score += continuity_bonus * 0.1
                    
                    candidates.append(ChordCandidate(
                        root=root_name,
                        quality=quality,
                        score=score,
                        matched_intervals=matched,
                        missing_intervals=missing,
                        extra_intervals=extra,
                    ))
        
        return candidates
    
    def _score_chord_match(
        self,
        pitch_classes: Set[int],
        root_pc: int,
        template: List[int],
    ) -> Tuple[float, List[int], List[int], List[int]]:
        """
        Score how well pitch classes match a chord template.
        
        Returns:
            (score, matched_intervals, missing_intervals, extra_intervals)
        """
        # Convert pitch classes to intervals from root
        intervals = {(pc - root_pc) % 12 for pc in pitch_classes}
        template_intervals = set(i % 12 for i in template)
        
        matched = list(intervals & template_intervals)
        missing = list(template_intervals - intervals)
        extra = list(intervals - template_intervals)
        
        # Calculate weighted score
        matched_score = sum(self.INTERVAL_WEIGHTS.get(i, 1.0) for i in matched)
        template_score = sum(self.INTERVAL_WEIGHTS.get(i, 1.0) for i in template_intervals)
        
        # Penalty for missing essential notes (root, third, fifth)
        missing_penalty = 0
        for interval in missing:
            if interval == 0:  # Root is essential
                missing_penalty += 0.5
            elif interval in [3, 4]:  # Third defines quality
                missing_penalty += 0.3
            elif interval == 7:  # Fifth is important
                missing_penalty += 0.2
            else:
                missing_penalty += 0.1
        
        # Small penalty for extra notes (could be extensions or passing tones)
        extra_penalty = len(extra) * 0.05
        
        # Final score
        if template_score > 0:
            score = (matched_score / template_score) - missing_penalty - extra_penalty
        else:
            score = 0
        
        return max(0, score), matched, missing, extra
    
    def _get_key_context_bonus(
        self,
        chord_root: str,
        chord_quality: str,
        key_root: str,
        key_mode: str,
    ) -> float:
        """
        Calculate bonus for chord being diatonic to the key.
        """
        key_root_pc = PITCH_NAMES.index(key_root)
        chord_root_pc = PITCH_NAMES.index(chord_root)
        interval = (chord_root_pc - key_root_pc) % 12
        
        # Get expected quality for this scale degree
        if key_mode == "major":
            expected_quality = self.DIATONIC_CHORDS_MAJOR.get(interval)
        else:
            expected_quality = self.DIATONIC_CHORDS_MINOR.get(interval)
        
        if expected_quality is None:
            return 0.0  # Non-diatonic root
        
        # Bonus for diatonic chord root
        bonus = 0.3
        
        # Extra bonus if quality matches expected
        if chord_quality == expected_quality:
            bonus += 0.2
        elif self._qualities_compatible(chord_quality, expected_quality):
            bonus += 0.1
        
        return bonus
    
    def _qualities_compatible(self, q1: str, q2: str) -> bool:
        """
        Check if two chord qualities are compatible (e.g., minor7 with minor).
        """
        compatible_groups = [
            {"major", "major7", "dominant7", "6", "add9"},
            {"minor", "minor7", "minor6"},
            {"diminished", "diminished7", "half_diminished7"},
            {"augmented", "augmented7"},
        ]
        
        for group in compatible_groups:
            if q1 in group and q2 in group:
                return True
        return False
    
    def _get_continuity_bonus(
        self,
        chord_root: str,
        chord_quality: str,
        prev_chord: Chord,
    ) -> float:
        """
        Calculate bonus for smooth chord progression.
        """
        if not prev_chord:
            return 0.0
        
        prev_root_pc = prev_chord.root_pitch_class
        curr_root_pc = PITCH_NAMES.index(chord_root)
        interval = (curr_root_pc - prev_root_pc) % 12
        
        # Common progressions (interval from previous chord root)
        COMMON_INTERVALS = {
            7: 0.3,   # Up a fifth (IV-I, V-I)
            5: 0.3,   # Down a fifth (I-IV, I-V) 
            2: 0.2,   # Step up (IV-V)
            10: 0.2,  # Step down (V-IV)
            3: 0.15,  # Minor third
            9: 0.15,  # Major sixth
            4: 0.1,   # Major third
            8: 0.1,   # Minor sixth
        }
        
        return COMMON_INTERVALS.get(interval, 0.0)

    def analyze_progression(
        self, 
        chords: List[Chord], 
        key_root: Optional[str] = None,
        key_mode: Optional[str] = None,
    ) -> ChordProgression:
        """
        Analyze chord progression in context of key.

        Args:
            chords: List of detected chords
            key_root: Key root (e.g., "C")
            key_mode: Key mode (e.g., "major")

        Returns:
            ChordProgression with roman numeral analysis
        """
        if not chords:
            return ChordProgression(chords=[], key_root=key_root, key_mode=key_mode)
        
        # Calculate roman numerals if key is provided
        roman_numerals = []
        if key_root and key_mode:
            for chord in chords:
                numeral = chord.get_roman_numeral(key_root, key_mode)
                roman_numerals.append(numeral)
        
        return ChordProgression(
            chords=chords,
            key_root=key_root,
            key_mode=key_mode,
            roman_numerals=roman_numerals,
        )
    
    def analyze(
        self,
        notes: List[Note],
        key_root: Optional[str] = None,
        key_mode: Optional[str] = None,
    ) -> ChordProgression:
        """
        Full chord analysis: detect chords and analyze progression.
        
        Args:
            notes: List of notes
            key_root: Optional key root
            key_mode: Optional key mode
            
        Returns:
            ChordProgression with detected chords and analysis
        """
        chords = self.detect_chords(notes, key_root, key_mode)
        return self.analyze_progression(chords, key_root, key_mode)

    def get_simultaneous_notes(
        self, 
        notes: List[Note], 
        time: float, 
        tolerance: float = 0.05
    ) -> List[Note]:
        """Get all notes sounding at a given time."""
        return [
            n for n in notes
            if n.onset <= time + tolerance and n.offset >= time - tolerance
        ]

    def notes_to_pitch_classes(self, notes: List[Note]) -> Set[int]:
        """Convert notes to pitch class set (0-11)."""
        return {n.pitch % 12 for n in notes}
    
    def identify_cadences(
        self,
        progression: ChordProgression,
    ) -> List[Tuple[int, str]]:
        """
        Identify cadences in a chord progression.
        
        Args:
            progression: Analyzed chord progression
            
        Returns:
            List of (chord_index, cadence_type) tuples
        """
        if not progression.roman_numerals or len(progression.roman_numerals) < 2:
            return []
        
        cadences = []
        numerals = progression.roman_numerals
        
        for i in range(1, len(numerals)):
            prev = numerals[i-1].upper().rstrip("7°+")
            curr = numerals[i].upper().rstrip("7°+")
            
            # Authentic cadence: V-I
            if prev == "V" and curr == "I":
                cadences.append((i, "authentic"))
            # Plagal cadence: IV-I
            elif prev == "IV" and curr == "I":
                cadences.append((i, "plagal"))
            # Half cadence: ?-V
            elif curr == "V":
                cadences.append((i, "half"))
            # Deceptive cadence: V-vi
            elif prev == "V" and curr == "VI":
                cadences.append((i, "deceptive"))
        
        return cadences
