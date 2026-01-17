"""Harmony analysis - Integrated harmonic analysis system.

Implements comprehensive harmony inference with:
- Integration of key detection and chord analysis
- Melodic context awareness
- Harmonic rhythm analysis
- Tension/resolution tracking
- Probabilistic inference with noise tolerance
- Musical plausibility scoring
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..core import Note, PITCH_NAMES
from .key import KeyDetector, KeyInfo
from .chords import ChordAnalyzer, Chord, ChordProgression


@dataclass
class HarmonicSegment:
    """A segment with consistent harmonic content."""
    
    onset: float
    offset: float
    chord: Optional[Chord]
    tension_level: float  # 0.0 (consonant) to 1.0 (dissonant)
    stability: float = 1.0  # How stable/resolved this segment feels
    melodic_notes: List[Note] = field(default_factory=list)  # Melody notes in this segment
    
    @property
    def duration(self) -> float:
        return self.offset - self.onset
    
    @property
    def is_resolved(self) -> bool:
        """Check if this segment feels resolved (low tension, high stability)."""
        return self.tension_level < 0.3 and self.stability > 0.7


@dataclass
class HarmonicContext:
    """Context for harmony inference at a point in time."""
    
    key_info: Optional[KeyInfo] = None
    previous_chords: List[Chord] = field(default_factory=list)
    local_melodic_pitches: List[int] = field(default_factory=list)
    beat_position: float = 0.0  # Position within measure (0-1)
    is_downbeat: bool = False


@dataclass
class HarmonyInfo:
    """Container for harmony analysis results."""
    
    key_info: Optional[KeyInfo] = None
    progression: Optional[ChordProgression] = None
    segments: List[HarmonicSegment] = field(default_factory=list)
    harmonic_rhythm: float = 0.0  # Average chord changes per beat
    average_tension: float = 0.0
    tension_curve: List[Tuple[float, float]] = field(default_factory=list)  # (time, tension)
    musical_coherence: float = 0.0  # Overall musical plausibility score
    
    @property
    def chord_symbols(self) -> List[str]:
        """Get list of chord symbols."""
        if self.progression:
            return self.progression.symbols
        return []
    
    @property
    def roman_numerals(self) -> List[str]:
        """Get roman numeral analysis."""
        if self.progression:
            return self.progression.roman_numerals
        return []


class HarmonyAnalyzer:
    """Integrated harmony analysis system.
    
    Combines key detection, chord analysis, and melodic context to produce
    musically coherent harmonic analysis with tolerance for noisy input.
    
    Features:
    - Automatic key detection with ambiguity handling
    - Context-aware chord detection
    - Tension/resolution analysis
    - Musical plausibility scoring
    - Harmonic rhythm calculation
    """

    # Consonance ratings for intervals (semitones)
    CONSONANCE = {
        0: 1.0,   # Unison - perfect
        3: 0.7,   # Minor third - consonant
        4: 0.7,   # Major third - consonant
        5: 0.8,   # Perfect fourth - consonant
        7: 0.9,   # Perfect fifth - very consonant
        8: 0.6,   # Minor sixth - consonant
        9: 0.6,   # Major sixth - consonant
        12: 1.0,  # Octave - perfect
        # Dissonant intervals
        1: 0.2,   # Minor second
        2: 0.3,   # Major second
        6: 0.3,   # Tritone
        10: 0.4,  # Minor seventh
        11: 0.4,  # Major seventh
    }
    
    # Stability ratings for chord functions
    CHORD_STABILITY = {
        "I": 1.0,    # Tonic - most stable
        "i": 1.0,
        "III": 0.6,  # Mediant
        "iii": 0.6,
        "IV": 0.7,   # Subdominant
        "iv": 0.7,
        "V": 0.4,    # Dominant - unstable, wants to resolve
        "V7": 0.3,
        "vi": 0.8,   # Relative minor - fairly stable
        "VI": 0.8,
        "ii": 0.5,   # Supertonic - pre-dominant
        "II": 0.5,
        "vii°": 0.2, # Leading tone - very unstable
        "VII": 0.4,
    }

    def __init__(
        self,
        segment_duration: float = 0.5,
        key_profile: str = "krumhansl",
        min_chord_duration: float = 0.25,
        tempo: float = 120.0,
        use_melodic_context: bool = True,
    ):
        """
        Initialize HarmonyAnalyzer.

        Args:
            segment_duration: Duration of analysis segments (seconds)
            key_profile: Key profile algorithm ("krumhansl" or "temperley")
            min_chord_duration: Minimum duration for chord detection
            tempo: Tempo in BPM (for beat-relative analysis)
            use_melodic_context: Whether to use melodic context in analysis
        """
        self.segment_duration = segment_duration
        self.tempo = tempo
        self.use_melodic_context = use_melodic_context
        
        # Initialize sub-analyzers
        self.key_detector = KeyDetector(
            profile_type=key_profile,
            detect_modes=False,
        )
        self.chord_analyzer = ChordAnalyzer(
            min_chord_duration=min_chord_duration,
            use_key_context=True,
        )

    def analyze(self, notes: List[Note], melody_notes: Optional[List[Note]] = None) -> HarmonyInfo:
        """
        Perform full harmonic analysis.

        Args:
            notes: List of all notes to analyze
            melody_notes: Optional separate melody line (if already extracted)

        Returns:
            HarmonyInfo with comprehensive analysis results
        """
        if not notes:
            return HarmonyInfo()
        
        # Step 1: Detect global key
        key_info = self.key_detector.analyze(notes)
        
        # Step 2: Detect chords with key context
        chords = self.chord_analyzer.detect_chords(
            notes,
            key_root=key_info.root if key_info.confidence > 0.3 else None,
            key_mode=key_info.mode if key_info.confidence > 0.3 else None,
        )
        
        # Step 3: Analyze chord progression
        progression = self.chord_analyzer.analyze_progression(
            chords,
            key_root=key_info.root,
            key_mode=key_info.mode,
        )
        
        # Step 4: Create harmonic segments with tension analysis
        segments = self._create_harmonic_segments(notes, chords, melody_notes)
        
        # Step 5: Calculate tension curve
        tension_curve = self._calculate_tension_curve(segments)
        
        # Step 6: Calculate harmonic rhythm
        harmonic_rhythm = self._calculate_harmonic_rhythm(chords)
        
        # Step 7: Calculate musical coherence score
        coherence = self._calculate_coherence(key_info, progression, segments)
        
        # Step 8: Calculate average tension
        avg_tension = np.mean([s.tension_level for s in segments]) if segments else 0.0
        
        return HarmonyInfo(
            key_info=key_info,
            progression=progression,
            segments=segments,
            harmonic_rhythm=harmonic_rhythm,
            average_tension=float(avg_tension),
            tension_curve=tension_curve,
            musical_coherence=coherence,
        )
    
    def analyze_with_context(
        self,
        notes: List[Note],
        key_root: Optional[str] = None,
        key_mode: Optional[str] = None,
        previous_chords: Optional[List[Chord]] = None,
    ) -> HarmonyInfo:
        """
        Analyze harmony with pre-specified context.
        
        Useful when key or previous chords are already known.
        
        Args:
            notes: Notes to analyze
            key_root: Known key root
            key_mode: Known key mode
            previous_chords: Chords from previous segment for continuity
            
        Returns:
            HarmonyInfo with analysis results
        """
        if not notes:
            return HarmonyInfo()
        
        # Use provided key or detect
        if key_root and key_mode:
            key_info = KeyInfo(
                root=key_root,
                mode=key_mode,
                confidence=1.0,
            )
        else:
            key_info = self.key_detector.analyze(notes)
        
        # Detect chords with context
        chords = self.chord_analyzer.detect_chords(
            notes,
            key_root=key_info.root,
            key_mode=key_info.mode,
        )
        
        progression = self.chord_analyzer.analyze_progression(
            chords, key_info.root, key_info.mode
        )
        
        segments = self._create_harmonic_segments(notes, chords, None)
        tension_curve = self._calculate_tension_curve(segments)
        harmonic_rhythm = self._calculate_harmonic_rhythm(chords)
        coherence = self._calculate_coherence(key_info, progression, segments)
        
        return HarmonyInfo(
            key_info=key_info,
            progression=progression,
            segments=segments,
            harmonic_rhythm=harmonic_rhythm,
            average_tension=float(np.mean([s.tension_level for s in segments])) if segments else 0.0,
            tension_curve=tension_curve,
            musical_coherence=coherence,
        )
    
    def _create_harmonic_segments(
        self,
        notes: List[Note],
        chords: List[Chord],
        melody_notes: Optional[List[Note]],
    ) -> List[HarmonicSegment]:
        """
        Create harmonic segments from notes and detected chords.
        """
        if not notes:
            return []
        
        segments = []
        
        if chords:
            # Create segments based on chord boundaries
            for chord in chords:
                # Get notes active during this chord
                active_notes = [
                    n for n in notes
                    if n.onset < chord.offset and n.offset > chord.onset
                ]
                
                # Get melody notes if provided
                if melody_notes:
                    segment_melody = [
                        n for n in melody_notes
                        if n.onset < chord.offset and n.offset > chord.onset
                    ]
                else:
                    segment_melody = []
                
                # Calculate tension and stability
                tension = self._calculate_tension(active_notes)
                stability = self._get_chord_stability(chord)
                
                segments.append(HarmonicSegment(
                    onset=chord.onset,
                    offset=chord.offset,
                    chord=chord,
                    tension_level=tension,
                    stability=stability,
                    melodic_notes=segment_melody,
                ))
        else:
            # No chords detected - create segments by time windows
            segments = self.segment_harmony(notes)
        
        return segments
    
    def _get_chord_stability(self, chord: Chord) -> float:
        """
        Get stability rating for a chord based on its function.
        """
        # Default stability for chords without clear function
        base_stability = 0.5
        
        # Major chords tend to be more stable than diminished
        quality_stability = {
            "major": 0.7,
            "minor": 0.6,
            "diminished": 0.3,
            "augmented": 0.3,
            "dominant7": 0.3,
            "major7": 0.65,
            "minor7": 0.55,
        }
        
        return quality_stability.get(chord.quality, base_stability)

    def segment_harmony(self, notes: List[Note]) -> List[HarmonicSegment]:
        """
        Segment notes into harmonic regions using time windows.
        
        Fallback when no chords are detected.

        Args:
            notes: List of notes

        Returns:
            List of HarmonicSegment objects
        """
        if not notes:
            return []
        
        segments = []
        
        min_time = min(n.onset for n in notes)
        max_time = max(n.offset for n in notes)
        
        current_time = min_time
        while current_time < max_time:
            end_time = min(current_time + self.segment_duration, max_time)
            
            # Get notes active in this segment
            active_notes = [
                n for n in notes
                if n.onset < end_time and n.offset > current_time
            ]
            
            # Calculate tension
            tension = self._calculate_tension(active_notes)
            
            segments.append(HarmonicSegment(
                onset=current_time,
                offset=end_time,
                chord=None,
                tension_level=tension,
                stability=1.0 - tension,  # Inverse of tension as rough stability
            ))
            
            current_time = end_time
        
        return segments
    
    def _calculate_tension_curve(
        self,
        segments: List[HarmonicSegment],
    ) -> List[Tuple[float, float]]:
        """
        Calculate tension over time.
        
        Returns:
            List of (time, tension) tuples
        """
        curve = []
        for segment in segments:
            # Add point at start of segment
            curve.append((segment.onset, segment.tension_level))
            # Add point at end of segment
            curve.append((segment.offset, segment.tension_level))
        return curve
    
    def _calculate_harmonic_rhythm(self, chords: List[Chord]) -> float:
        """
        Calculate harmonic rhythm (chord changes per beat).
        
        Args:
            chords: List of detected chords
            
        Returns:
            Chord changes per beat
        """
        if not chords or len(chords) < 2:
            return 0.0
        
        # Calculate total duration
        total_duration = chords[-1].offset - chords[0].onset
        if total_duration <= 0:
            return 0.0
        
        # Calculate beats in this duration
        beat_duration = 60.0 / self.tempo  # seconds per beat
        total_beats = total_duration / beat_duration
        
        # Chord changes (number of chords - 1)
        changes = len(chords) - 1
        
        return changes / total_beats if total_beats > 0 else 0.0
    
    def _calculate_coherence(
        self,
        key_info: KeyInfo,
        progression: ChordProgression,
        segments: List[HarmonicSegment],
    ) -> float:
        """
        Calculate musical coherence score.
        
        High coherence = clear key, diatonic chords, smooth progression.
        Low coherence = ambiguous key, many non-diatonic chords.
        
        Returns:
            Coherence score 0-1
        """
        scores = []
        
        # Key confidence contributes to coherence
        if key_info:
            key_score = key_info.confidence * (1 - key_info.ambiguity_score)
            scores.append(key_score)
        
        # Diatonic chord ratio
        if progression and progression.roman_numerals:
            diatonic_count = sum(
                1 for rn in progression.roman_numerals
                if not rn.startswith("(")
            )
            diatonic_ratio = diatonic_count / len(progression.roman_numerals)
            scores.append(diatonic_ratio)
        
        # Average stability
        if segments:
            avg_stability = np.mean([s.stability for s in segments])
            scores.append(float(avg_stability))
        
        # Resolution patterns (tension followed by release)
        if len(segments) > 1:
            resolution_count = 0
            for i in range(1, len(segments)):
                if segments[i-1].tension_level > 0.5 and segments[i].tension_level < 0.4:
                    resolution_count += 1
            resolution_ratio = resolution_count / (len(segments) - 1)
            scores.append(min(1.0, resolution_ratio + 0.5))  # Base + bonus
        
        return float(np.mean(scores)) if scores else 0.5

    def _calculate_tension(self, notes: List[Note]) -> float:
        """
        Calculate harmonic tension from simultaneous notes.
        
        Considers:
        - Interval consonance/dissonance
        - Note density
        - Chromatic density
        """
        if len(notes) < 2:
            return 0.0
        
        # Get all intervals between notes
        pitches = [n.pitch for n in notes]
        total_consonance = 0.0
        count = 0
        
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                interval = abs(pitches[i] - pitches[j]) % 12
                consonance = self.CONSONANCE.get(interval, 0.5)
                total_consonance += consonance
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_consonance = total_consonance / count
        
        # Additional tension from chromatic density
        pitch_classes = set(p % 12 for p in pitches)
        chromatic_pairs = 0
        pc_list = list(pitch_classes)
        for i in range(len(pc_list)):
            for j in range(i + 1, len(pc_list)):
                if abs(pc_list[i] - pc_list[j]) == 1 or abs(pc_list[i] - pc_list[j]) == 11:
                    chromatic_pairs += 1
        
        chromatic_tension = min(0.3, chromatic_pairs * 0.1)
        
        # Combine
        base_tension = 1.0 - avg_consonance
        return min(1.0, base_tension + chromatic_tension)
    
    def get_chord_at_time(self, time: float, chords: List[Chord]) -> Optional[Chord]:
        """
        Get the chord sounding at a specific time.
        
        Args:
            time: Time in seconds
            chords: List of chords
            
        Returns:
            Chord at that time, or None
        """
        for chord in chords:
            if chord.onset <= time < chord.offset:
                return chord
        return None
    
    def analyze_melodic_harmony_fit(
        self,
        melody_notes: List[Note],
        chords: List[Chord],
        key_root: str,
        key_mode: str,
    ) -> float:
        """
        Analyze how well melody fits with the harmony.
        
        Args:
            melody_notes: Melody notes
            chords: Detected chords
            key_root: Key root
            key_mode: Key mode
            
        Returns:
            Fit score 0-1 (1 = perfect fit)
        """
        if not melody_notes or not chords:
            return 0.5
        
        fit_scores = []
        scale_notes = set(self.key_detector.get_scale_notes(key_root, key_mode))
        
        for note in melody_notes:
            pc = note.pitch % 12
            
            # Check if note is in scale
            in_scale = pc in scale_notes
            
            # Check if note fits current chord
            chord = self.get_chord_at_time(note.onset, chords)
            if chord:
                chord_pcs = chord.pitch_classes
                in_chord = pc in chord_pcs
                
                # Chord tones are best fit
                if in_chord:
                    fit_scores.append(1.0)
                # Scale tones are acceptable
                elif in_scale:
                    fit_scores.append(0.7)
                # Chromatic passing tones if short
                elif note.duration < 0.1:
                    fit_scores.append(0.4)
                else:
                    fit_scores.append(0.2)
            else:
                # No chord context - just check scale
                fit_scores.append(0.8 if in_scale else 0.3)
        
        return float(np.mean(fit_scores)) if fit_scores else 0.5
    
    def suggest_chord_for_melody(
        self,
        melody_notes: List[Note],
        key_root: str,
        key_mode: str,
        previous_chord: Optional[Chord] = None,
    ) -> Optional[Chord]:
        """
        Suggest a chord that fits the given melody notes.
        
        Args:
            melody_notes: Melody notes to harmonize
            key_root: Key root
            key_mode: Key mode
            previous_chord: Previous chord for continuity
            
        Returns:
            Suggested Chord or None
        """
        if not melody_notes:
            return None
        
        # Get melody pitch classes weighted by duration
        weighted_pcs = defaultdict(float)
        for note in melody_notes:
            pc = note.pitch % 12
            weighted_pcs[pc] += note.duration
        
        # Sort by weight
        top_pcs = sorted(weighted_pcs.keys(), key=lambda pc: weighted_pcs[pc], reverse=True)
        
        if not top_pcs:
            return None
        
        # Get onset and offset
        onset = min(n.onset for n in melody_notes)
        offset = max(n.offset for n in melody_notes)
        
        # Try to find a diatonic chord that contains the top melody notes
        scale_notes = set(self.key_detector.get_scale_notes(key_root, key_mode))
        
        # Build candidate chords on each scale degree
        best_chord = None
        best_score = -1
        
        key_root_pc = PITCH_NAMES.index(key_root)
        
        diatonic_roots = [
            (key_root_pc + interval) % 12
            for interval in ([0, 2, 4, 5, 7, 9, 11] if key_mode == "major" else [0, 2, 3, 5, 7, 8, 10])
        ]
        
        for root_pc in diatonic_roots:
            root_name = PITCH_NAMES[root_pc]
            
            # Determine expected quality based on scale degree
            interval_from_key = (root_pc - key_root_pc) % 12
            if key_mode == "major":
                expected = self.chord_analyzer.DIATONIC_CHORDS_MAJOR.get(interval_from_key, "major")
            else:
                expected = self.chord_analyzer.DIATONIC_CHORDS_MINOR.get(interval_from_key, "minor")
            
            # Build chord pitch classes
            template = self.chord_analyzer.CHORD_TEMPLATES.get(expected, [0, 4, 7])
            chord_pcs = set((root_pc + i) % 12 for i in template)
            
            # Score: how many melody notes are in this chord
            score = sum(weighted_pcs.get(pc, 0) for pc in chord_pcs if pc in top_pcs[:3])
            
            # Bonus for smooth voice leading from previous chord
            if previous_chord:
                prev_root_pc = PITCH_NAMES.index(previous_chord.root)
                interval = (root_pc - prev_root_pc) % 12
                if interval in [5, 7]:  # Fourth/fifth motion
                    score += 0.2
            
            if score > best_score:
                best_score = score
                best_chord = Chord(
                    root=root_name,
                    quality=expected,
                    onset=onset,
                    offset=offset,
                    notes=[],
                    confidence=min(1.0, score),
                )
        
        return best_chord
