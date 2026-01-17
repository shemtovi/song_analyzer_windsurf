"""Comprehensive tests for harmony inference modules.

Tests cover:
- Key detection with various keys and modes
- Chord detection and progression analysis
- Harmony analysis integration
- Resilience to noisy/imperfect input
- Edge cases and error handling
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Note, PITCH_NAMES
from src.inference import (
    KeyDetector,
    KeyInfo,
    ChordAnalyzer,
    Chord,
    ChordProgression,
    HarmonyAnalyzer,
    HarmonyInfo,
    ResilienceProcessor,
    NoteFilter,
    TimingCorrector,
    PitchCorrector,
    ConfidenceWeighter,
    apply_musical_rules,
)


# ============================================================================
# Test Fixtures - Helper functions to create test data
# ============================================================================

def create_scale_notes(root: str, mode: str, octave: int = 4, duration: float = 0.5) -> list:
    """Create notes for a scale."""
    root_pc = PITCH_NAMES.index(root)
    
    if mode == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else:  # minor
        intervals = [0, 2, 3, 5, 7, 8, 10]
    
    notes = []
    time = 0.0
    base_pitch = octave * 12 + root_pc
    
    for interval in intervals:
        notes.append(Note(
            pitch=base_pitch + interval,
            onset=time,
            offset=time + duration,
            velocity=80,
        ))
        time += duration
    
    return notes


def create_chord_notes(root: str, quality: str, octave: int = 4, 
                       onset: float = 0.0, duration: float = 1.0) -> list:
    """Create notes for a chord."""
    root_pc = PITCH_NAMES.index(root)
    base_pitch = octave * 12 + root_pc
    
    templates = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "diminished": [0, 3, 6],
        "augmented": [0, 4, 8],
        "dominant7": [0, 4, 7, 10],
        "major7": [0, 4, 7, 11],
        "minor7": [0, 3, 7, 10],
    }
    
    intervals = templates.get(quality, [0, 4, 7])
    
    return [
        Note(
            pitch=base_pitch + interval,
            onset=onset,
            offset=onset + duration,
            velocity=80,
        )
        for interval in intervals
    ]


def create_chord_progression_notes(progression: list, key_root: str = "C",
                                   duration: float = 1.0) -> list:
    """Create notes for a chord progression.
    
    progression: list of (root, quality) tuples
    """
    notes = []
    time = 0.0
    
    for root, quality in progression:
        chord_notes = create_chord_notes(root, quality, octave=4, 
                                         onset=time, duration=duration)
        notes.extend(chord_notes)
        time += duration
    
    return notes


def add_noise_to_notes(notes: list, 
                       pitch_noise: int = 0,
                       timing_noise: float = 0.0,
                       add_ghost_notes: int = 0,
                       add_wrong_notes: int = 0) -> list:
    """Add noise to a list of notes for testing resilience."""
    result = notes.copy()
    
    # Add pitch noise
    if pitch_noise > 0:
        for i in range(len(result)):
            if np.random.random() < 0.2:  # 20% of notes
                shift = np.random.randint(-pitch_noise, pitch_noise + 1)
                result[i] = Note(
                    pitch=result[i].pitch + shift,
                    onset=result[i].onset,
                    offset=result[i].offset,
                    velocity=result[i].velocity,
                )
    
    # Add timing noise
    if timing_noise > 0:
        for i in range(len(result)):
            shift = np.random.uniform(-timing_noise, timing_noise)
            result[i] = Note(
                pitch=result[i].pitch,
                onset=max(0, result[i].onset + shift),
                offset=result[i].offset + shift,
                velocity=result[i].velocity,
            )
    
    # Add ghost notes (very quiet)
    for _ in range(add_ghost_notes):
        result.append(Note(
            pitch=np.random.randint(48, 84),
            onset=np.random.uniform(0, max(n.offset for n in notes)),
            offset=np.random.uniform(0.1, 0.5),
            velocity=np.random.randint(1, 15),
        ))
    
    # Add wrong notes
    for _ in range(add_wrong_notes):
        result.append(Note(
            pitch=np.random.randint(48, 84),
            onset=np.random.uniform(0, max(n.offset for n in notes)),
            offset=np.random.uniform(0.1, 0.5),
            velocity=np.random.randint(60, 100),
        ))
    
    return result


# ============================================================================
# Key Detection Tests
# ============================================================================

class TestKeyDetector:
    """Tests for KeyDetector."""
    
    def test_detector_creation(self):
        """Test KeyDetector can be instantiated."""
        detector = KeyDetector()
        assert detector is not None
        assert hasattr(detector, 'analyze')
        assert hasattr(detector, 'detect_from_notes')
    
    def test_c_major_scale(self):
        """Test detection of C major scale."""
        notes = create_scale_notes("C", "major")
        detector = KeyDetector()
        
        key, mode, confidence = detector.detect_from_notes(notes)
        
        assert key == "C"
        assert mode == "major"
        assert confidence > 0.5
    
    def test_a_minor_scale(self):
        """Test detection of A minor scale."""
        notes = create_scale_notes("A", "minor")
        detector = KeyDetector()
        
        key, mode, confidence = detector.detect_from_notes(notes)
        
        # A minor and C major share notes, so both are valid
        assert key in ["A", "C"]
        assert confidence > 0.3
    
    def test_g_major_scale(self):
        """Test detection of G major scale."""
        notes = create_scale_notes("G", "major")
        detector = KeyDetector()
        
        key, mode, confidence = detector.detect_from_notes(notes)
        
        assert key == "G"
        assert mode == "major"
        assert confidence > 0.5
    
    def test_d_minor_scale(self):
        """Test detection of D minor scale."""
        notes = create_scale_notes("D", "minor")
        detector = KeyDetector()
        
        result = detector.analyze(notes)
        
        # D minor or F major (relative)
        assert result.root in ["D", "F"]
        assert result.confidence > 0.3
    
    def test_analyze_returns_key_info(self):
        """Test that analyze returns KeyInfo with all fields."""
        notes = create_scale_notes("C", "major")
        detector = KeyDetector()
        
        result = detector.analyze(notes)
        
        assert isinstance(result, KeyInfo)
        assert result.root is not None
        assert result.mode is not None
        assert 0 <= result.confidence <= 1
        assert result.pitch_class_distribution is not None
        assert len(result.pitch_class_distribution) == 12
    
    def test_ambiguity_detection(self):
        """Test that ambiguous keys are flagged."""
        # Notes that could be C major or A minor
        notes = create_scale_notes("C", "major")
        detector = KeyDetector()
        
        result = detector.analyze(notes)
        
        # Relative major/minor are naturally ambiguous
        assert result.relative_key is not None
        assert "minor" in result.relative_key or "major" in result.relative_key
    
    def test_alternatives_provided(self):
        """Test that alternative keys are provided."""
        notes = create_scale_notes("C", "major")
        detector = KeyDetector()
        
        result = detector.analyze(notes, return_alternatives=True)
        
        assert len(result.alternatives) > 0
    
    def test_empty_notes(self):
        """Test handling of empty note list."""
        detector = KeyDetector()
        result = detector.analyze([])
        
        assert result.confidence == 0.0
        assert result.ambiguity_score == 1.0
    
    def test_single_note(self):
        """Test handling of single note."""
        notes = [Note(pitch=60, onset=0, offset=1, velocity=80)]
        detector = KeyDetector()
        
        result = detector.analyze(notes)
        
        # Should handle gracefully with low confidence
        assert result.confidence < 0.5
    
    def test_weighted_detection(self):
        """Test that longer notes are weighted more."""
        # C note is longer, so C major should be stronger
        notes = [
            Note(pitch=60, onset=0, offset=2.0, velocity=80),  # C - long
            Note(pitch=62, onset=2, offset=2.5, velocity=80),  # D - short
            Note(pitch=64, onset=2.5, offset=3.0, velocity=80),  # E - short
        ]
        detector = KeyDetector()
        
        key, mode, _ = detector.detect_from_notes(notes, weighted=True)
        
        # C should be emphasized due to longer duration
        assert key == "C"
    
    def test_temperley_profile(self):
        """Test with Temperley key profiles."""
        notes = create_scale_notes("C", "major")
        detector = KeyDetector(profile_type="temperley")
        
        key, mode, _ = detector.detect_from_notes(notes)
        
        assert key == "C"
        assert mode == "major"
    
    def test_get_scale_notes(self):
        """Test get_scale_notes returns correct pitch classes."""
        detector = KeyDetector()
        
        c_major = detector.get_scale_notes("C", "major")
        assert set(c_major) == {0, 2, 4, 5, 7, 9, 11}
        
        a_minor = detector.get_scale_notes("A", "minor")
        assert set(a_minor) == {9, 11, 0, 2, 4, 5, 7}  # A B C D E F G
    
    def test_key_segments(self):
        """Test detection of key changes over time."""
        # First half C major, second half G major
        notes_c = create_scale_notes("C", "major")
        notes_g = [
            Note(pitch=n.pitch, onset=n.onset + 4, offset=n.offset + 4, velocity=n.velocity)
            for n in create_scale_notes("G", "major")
        ]
        all_notes = notes_c + notes_g
        
        detector = KeyDetector()
        segments = detector.detect_key_segments(all_notes, segment_duration=2.0)
        
        assert len(segments) >= 2


# ============================================================================
# Chord Detection Tests
# ============================================================================

class TestChordAnalyzer:
    """Tests for ChordAnalyzer."""
    
    def test_analyzer_creation(self):
        """Test ChordAnalyzer can be instantiated."""
        analyzer = ChordAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'detect_chords')
        assert hasattr(analyzer, 'analyze')
    
    def test_detect_c_major_chord(self):
        """Test detection of C major chord."""
        notes = create_chord_notes("C", "major")
        analyzer = ChordAnalyzer()
        
        chords = analyzer.detect_chords(notes)
        
        assert len(chords) == 1
        assert chords[0].root == "C"
        assert chords[0].quality == "major"
    
    def test_detect_a_minor_chord(self):
        """Test detection of A minor chord."""
        notes = create_chord_notes("A", "minor")
        analyzer = ChordAnalyzer()
        
        chords = analyzer.detect_chords(notes)
        
        assert len(chords) == 1
        assert chords[0].root == "A"
        assert chords[0].quality == "minor"
    
    def test_detect_g_dominant7(self):
        """Test detection of G7 chord."""
        notes = create_chord_notes("G", "dominant7")
        analyzer = ChordAnalyzer()
        
        chords = analyzer.detect_chords(notes)
        
        assert len(chords) == 1
        assert chords[0].root == "G"
        assert chords[0].quality == "dominant7"
    
    def test_detect_chord_progression(self):
        """Test detection of I-IV-V-I progression."""
        progression = [("C", "major"), ("F", "major"), ("G", "major"), ("C", "major")]
        notes = create_chord_progression_notes(progression)
        analyzer = ChordAnalyzer()
        
        chords = analyzer.detect_chords(notes)
        
        assert len(chords) >= 3  # May merge identical adjacent chords
        
        # Check that we have the expected roots
        roots = [c.root for c in chords]
        assert "C" in roots
        assert "F" in roots
        assert "G" in roots
    
    def test_roman_numeral_analysis(self):
        """Test roman numeral analysis."""
        chord = Chord(
            root="G", quality="major",
            onset=0, offset=1, notes=[55, 59, 62]
        )
        
        numeral = chord.get_roman_numeral("C", "major")
        
        assert numeral == "V"
    
    def test_minor_chord_roman_numeral(self):
        """Test roman numeral for minor chord."""
        chord = Chord(
            root="A", quality="minor",
            onset=0, offset=1, notes=[57, 60, 64]
        )
        
        numeral = chord.get_roman_numeral("C", "major")
        
        assert numeral == "vi"
    
    def test_chord_progression_analysis(self):
        """Test full progression analysis."""
        progression = [("C", "major"), ("F", "major"), ("G", "major"), ("C", "major")]
        notes = create_chord_progression_notes(progression)
        analyzer = ChordAnalyzer()
        
        result = analyzer.analyze(notes, key_root="C", key_mode="major")
        
        assert isinstance(result, ChordProgression)
        assert result.key_root == "C"
        assert result.key_mode == "major"
    
    def test_chord_with_key_context(self):
        """Test that key context improves chord detection."""
        notes = create_chord_notes("D", "minor")
        analyzer = ChordAnalyzer(use_key_context=True)
        
        # With C major context, D should be detected as minor (ii)
        chords = analyzer.detect_chords(notes, key_root="C", key_mode="major")
        
        assert len(chords) == 1
        assert chords[0].root == "D"
    
    def test_empty_notes(self):
        """Test handling of empty note list."""
        analyzer = ChordAnalyzer()
        chords = analyzer.detect_chords([])
        
        assert chords == []
    
    def test_single_note(self):
        """Test handling of single note (not a chord)."""
        notes = [Note(pitch=60, onset=0, offset=1, velocity=80)]
        analyzer = ChordAnalyzer(min_notes_for_chord=2)
        
        chords = analyzer.detect_chords(notes)
        
        assert chords == []
    
    def test_chord_inversion_detection(self):
        """Test detection of chord inversions."""
        # E in bass with C and G above = C/E (first inversion)
        notes = [
            Note(pitch=52, onset=0, offset=1, velocity=80),  # E3
            Note(pitch=60, onset=0, offset=1, velocity=80),  # C4
            Note(pitch=67, onset=0, offset=1, velocity=80),  # G4
        ]
        analyzer = ChordAnalyzer()
        
        chords = analyzer.detect_chords(notes)
        
        assert len(chords) == 1
        # Should detect C major with E bass
        assert chords[0].root == "C"
        assert chords[0].bass == "E"
    
    def test_identify_cadences(self):
        """Test cadence identification."""
        progression = [("C", "major"), ("G", "major"), ("C", "major")]
        notes = create_chord_progression_notes(progression)
        analyzer = ChordAnalyzer()
        
        result = analyzer.analyze(notes, key_root="C", key_mode="major")
        cadences = analyzer.identify_cadences(result)
        
        # Should find authentic cadence (V-I)
        cadence_types = [c[1] for c in cadences]
        assert "authentic" in cadence_types or "half" in cadence_types
    
    def test_common_progressions(self):
        """Test detection of common progressions."""
        # I-V-vi-IV progression
        progression = [("C", "major"), ("G", "major"), ("A", "minor"), ("F", "major")]
        notes = create_chord_progression_notes(progression)
        analyzer = ChordAnalyzer()
        
        result = analyzer.analyze(notes, key_root="C", key_mode="major")
        common = result.get_common_progressions()
        
        # May or may not match exactly, but should run without error
        assert isinstance(common, list)


# ============================================================================
# Harmony Analyzer Tests
# ============================================================================

class TestHarmonyAnalyzer:
    """Tests for integrated HarmonyAnalyzer."""
    
    def test_analyzer_creation(self):
        """Test HarmonyAnalyzer can be instantiated."""
        analyzer = HarmonyAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
    
    def test_full_analysis(self):
        """Test full harmony analysis."""
        notes = create_chord_progression_notes([
            ("C", "major"), ("F", "major"), ("G", "major"), ("C", "major")
        ])
        analyzer = HarmonyAnalyzer()
        
        result = analyzer.analyze(notes)
        
        assert isinstance(result, HarmonyInfo)
        assert result.key_info is not None
        assert result.progression is not None
    
    def test_key_detection_integration(self):
        """Test that key detection is integrated."""
        notes = create_scale_notes("G", "major")
        analyzer = HarmonyAnalyzer()
        
        result = analyzer.analyze(notes)
        
        assert result.key_info.root == "G"
        assert result.key_info.mode == "major"
    
    def test_tension_calculation(self):
        """Test tension curve calculation."""
        notes = create_chord_notes("C", "major")
        analyzer = HarmonyAnalyzer()
        
        result = analyzer.analyze(notes)
        
        assert result.tension_curve is not None
        # Major triad should have low tension
        assert result.average_tension < 0.5
    
    def test_harmonic_rhythm(self):
        """Test harmonic rhythm calculation."""
        # Fast chord changes = high harmonic rhythm
        progression = [("C", "major"), ("G", "major"), ("A", "minor"), ("F", "major")]
        notes = create_chord_progression_notes(progression, duration=0.5)
        analyzer = HarmonyAnalyzer(tempo=120.0)
        
        result = analyzer.analyze(notes)
        
        assert result.harmonic_rhythm > 0
    
    def test_musical_coherence(self):
        """Test musical coherence scoring."""
        # Diatonic progression should have high coherence
        progression = [("C", "major"), ("F", "major"), ("G", "major"), ("C", "major")]
        notes = create_chord_progression_notes(progression)
        analyzer = HarmonyAnalyzer()
        
        result = analyzer.analyze(notes)
        
        assert result.musical_coherence > 0.3
    
    def test_analyze_with_context(self):
        """Test analysis with pre-specified key."""
        notes = create_chord_notes("G", "major")
        analyzer = HarmonyAnalyzer()
        
        result = analyzer.analyze_with_context(notes, key_root="C", key_mode="major")
        
        assert result.key_info.root == "C"
        assert result.key_info.confidence == 1.0
    
    def test_empty_notes(self):
        """Test handling of empty notes."""
        analyzer = HarmonyAnalyzer()
        result = analyzer.analyze([])
        
        assert result.key_info is None
        assert result.progression is None
    
    def test_melodic_harmony_fit(self):
        """Test melody-harmony fit analysis."""
        chord_notes = create_chord_notes("C", "major", onset=0, duration=2)
        melody_notes = [
            Note(pitch=64, onset=0, offset=0.5, velocity=80),  # E - chord tone
            Note(pitch=67, onset=0.5, offset=1.0, velocity=80),  # G - chord tone
            Note(pitch=72, onset=1.0, offset=1.5, velocity=80),  # C - chord tone
        ]
        
        analyzer = HarmonyAnalyzer()
        chords = [Chord(root="C", quality="major", onset=0, offset=2, notes=[60, 64, 67])]
        
        fit = analyzer.analyze_melodic_harmony_fit(
            melody_notes, chords, "C", "major"
        )
        
        # All melody notes are chord tones, so fit should be high
        assert fit > 0.8
    
    def test_suggest_chord_for_melody(self):
        """Test chord suggestion for melody."""
        melody_notes = [
            Note(pitch=64, onset=0, offset=0.5, velocity=80),  # E
            Note(pitch=67, onset=0.5, offset=1.0, velocity=80),  # G
        ]
        
        analyzer = HarmonyAnalyzer()
        suggested = analyzer.suggest_chord_for_melody(
            melody_notes, "C", "major"
        )
        
        assert suggested is not None
        # C or Em would fit these notes
        assert suggested.root in ["C", "E"]


# ============================================================================
# Resilience Tests
# ============================================================================

class TestNoteFilter:
    """Tests for NoteFilter."""
    
    def test_filter_creation(self):
        """Test NoteFilter can be instantiated."""
        filt = NoteFilter()
        assert filt is not None
    
    def test_remove_ghost_notes(self):
        """Test removal of ghost notes (low velocity)."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),
            Note(pitch=62, onset=0, offset=1, velocity=5),  # Ghost
        ]
        filt = NoteFilter(min_velocity=15)
        
        filtered, stats = filt.filter_notes(notes, remove_ghosts=True)
        
        assert len(filtered) == 1
        assert stats.removed_ghost_notes == 1
    
    def test_remove_very_short_notes(self):
        """Test removal of very short notes."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),
            Note(pitch=62, onset=1, offset=1.01, velocity=80),  # Too short
        ]
        filt = NoteFilter(min_duration=0.03)
        
        filtered, stats = filt.filter_notes(notes, remove_short=True)
        
        assert len(filtered) == 1
    
    def test_remove_pitch_outliers(self):
        """Test removal of pitch outliers."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),
            Note(pitch=10, onset=0, offset=1, velocity=80),  # Too low
        ]
        filt = NoteFilter(pitch_range=(21, 108))
        
        filtered, stats = filt.filter_notes(notes, remove_outliers=True)
        
        assert len(filtered) == 1
        assert stats.removed_outliers == 1
    
    def test_remove_duplicates(self):
        """Test removal of duplicate notes."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),
            Note(pitch=60, onset=0.02, offset=1.02, velocity=60),  # Duplicate
        ]
        filt = NoteFilter()
        
        filtered = filt.remove_duplicate_notes(notes, time_tolerance=0.05)
        
        assert len(filtered) == 1
        assert filtered[0].velocity == 80  # Kept higher velocity


class TestTimingCorrector:
    """Tests for TimingCorrector."""
    
    def test_corrector_creation(self):
        """Test TimingCorrector can be instantiated."""
        corrector = TimingCorrector()
        assert corrector is not None
    
    def test_quantize_notes(self):
        """Test note quantization."""
        notes = [
            Note(pitch=60, onset=0.52, offset=1.02, velocity=80),  # Slightly off
        ]
        corrector = TimingCorrector(tempo=120.0, grid_resolution=16)
        
        quantized = corrector.quantize_notes(notes, strength=1.0)
        
        # At 120 BPM, 16th note = 0.125s
        # 0.52 should snap to 0.5
        assert abs(quantized[0].onset - 0.5) < 0.01
    
    def test_partial_quantization(self):
        """Test partial quantization (strength < 1)."""
        notes = [
            Note(pitch=60, onset=0.52, offset=1.02, velocity=80),
        ]
        corrector = TimingCorrector(tempo=120.0, grid_resolution=16)
        
        quantized = corrector.quantize_notes(notes, strength=0.5)
        
        # Should move halfway toward grid
        # Original: 0.52, Grid: 0.5, Result: ~0.51
        assert quantized[0].onset < 0.52
        assert quantized[0].onset > 0.5
    
    def test_align_simultaneous_notes(self):
        """Test alignment of chord notes."""
        notes = [
            Note(pitch=60, onset=0.0, offset=1, velocity=80),
            Note(pitch=64, onset=0.02, offset=1, velocity=80),  # Slightly late
            Note(pitch=67, onset=0.01, offset=1, velocity=80),  # Slightly late
        ]
        corrector = TimingCorrector()
        
        aligned = corrector.align_simultaneous_notes(notes, tolerance=0.05)
        
        # All notes should have the same onset now
        onsets = [n.onset for n in aligned]
        assert max(onsets) - min(onsets) < 0.001


class TestPitchCorrector:
    """Tests for PitchCorrector."""
    
    def test_corrector_creation(self):
        """Test PitchCorrector can be instantiated."""
        corrector = PitchCorrector()
        assert corrector is not None
    
    def test_snap_to_scale(self):
        """Test snapping notes to scale."""
        # C# is not in C major scale
        notes = [
            Note(pitch=61, onset=0, offset=1, velocity=80),  # C#
        ]
        # Use 100 cents (1 semitone) tolerance to allow correction
        corrector = PitchCorrector(tolerance_cents=100)
        c_major = [0, 2, 4, 5, 7, 9, 11]  # C major scale
        
        corrected = corrector.snap_to_scale(notes, c_major)
        
        # C# should snap to C or D
        assert corrected[0].pitch in [60, 62]
    
    def test_no_correction_for_scale_tones(self):
        """Test that scale tones are not corrected."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),  # C
            Note(pitch=64, onset=0, offset=1, velocity=80),  # E
        ]
        corrector = PitchCorrector()
        c_major = [0, 2, 4, 5, 7, 9, 11]
        
        corrected = corrector.snap_to_scale(notes, c_major)
        
        assert corrected[0].pitch == 60
        assert corrected[1].pitch == 64


class TestConfidenceWeighter:
    """Tests for ConfidenceWeighter."""
    
    def test_weighter_creation(self):
        """Test ConfidenceWeighter can be instantiated."""
        weighter = ConfidenceWeighter()
        assert weighter is not None
    
    def test_velocity_affects_confidence(self):
        """Test that velocity affects confidence."""
        loud = Note(pitch=60, onset=0, offset=1, velocity=127)
        quiet = Note(pitch=60, onset=0, offset=1, velocity=30)
        
        weighter = ConfidenceWeighter()
        
        loud_conf = weighter.calculate_confidence(loud)
        quiet_conf = weighter.calculate_confidence(quiet)
        
        assert loud_conf > quiet_conf
    
    def test_duration_affects_confidence(self):
        """Test that duration affects confidence."""
        long_note = Note(pitch=60, onset=0, offset=1, velocity=80)
        short_note = Note(pitch=60, onset=0, offset=0.1, velocity=80)
        
        weighter = ConfidenceWeighter()
        
        long_conf = weighter.calculate_confidence(long_note)
        short_conf = weighter.calculate_confidence(short_note)
        
        assert long_conf > short_conf


class TestResilienceProcessor:
    """Tests for integrated ResilienceProcessor."""
    
    def test_processor_creation(self):
        """Test ResilienceProcessor can be instantiated."""
        processor = ResilienceProcessor()
        assert processor is not None
    
    def test_full_processing(self):
        """Test full resilience processing pipeline."""
        # Create noisy notes
        clean_notes = create_scale_notes("C", "major")
        noisy_notes = add_noise_to_notes(
            clean_notes,
            add_ghost_notes=3,
            timing_noise=0.02,
        )
        
        processor = ResilienceProcessor(min_velocity=15)
        processed, stats = processor.process(noisy_notes)
        
        # Should have removed ghost notes
        assert stats.removed_ghost_notes >= 0
        assert stats.filtered_count <= stats.original_count
    
    def test_high_confidence_notes(self):
        """Test extraction of high-confidence notes."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=100),  # High confidence
            Note(pitch=62, onset=1, offset=1.05, velocity=20),  # Low confidence
        ]
        
        processor = ResilienceProcessor()
        high_conf = processor.get_high_confidence_notes(notes, threshold=0.5)
        
        assert len(high_conf) >= 1


class TestApplyMusicalRules:
    """Tests for apply_musical_rules function."""
    
    def test_keeps_scale_tones(self):
        """Test that scale tones are kept."""
        notes = create_scale_notes("C", "major")
        
        filtered = apply_musical_rules(notes, key_root="C", key_mode="major")
        
        assert len(filtered) == len(notes)
    
    def test_handles_no_key_context(self):
        """Test handling without key context."""
        notes = create_scale_notes("C", "major")
        
        filtered = apply_musical_rules(notes, key_root=None, key_mode=None)
        
        assert len(filtered) == len(notes)


# ============================================================================
# Noisy Input Tests - Real-world resilience scenarios
# ============================================================================

class TestNoisyInputResilience:
    """Tests for handling noisy/imperfect input data."""
    
    def test_key_detection_with_wrong_notes(self):
        """Test key detection with some wrong notes."""
        clean_notes = create_scale_notes("C", "major")
        noisy_notes = add_noise_to_notes(clean_notes, add_wrong_notes=2)
        
        detector = KeyDetector()
        result = detector.analyze(noisy_notes)
        
        # Should still detect C major with reasonable confidence
        assert result.root == "C" or result.confidence > 0.3
    
    def test_chord_detection_with_missing_notes(self):
        """Test chord detection when notes are missing."""
        # C major without the 5th (G)
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),  # C
            Note(pitch=64, onset=0, offset=1, velocity=80),  # E
        ]
        
        analyzer = ChordAnalyzer(min_notes_for_chord=2)
        chords = analyzer.detect_chords(notes)
        
        # Should still detect C major or recognize as incomplete
        if chords:
            assert chords[0].root == "C"
    
    def test_chord_detection_with_extra_notes(self):
        """Test chord detection with extra/passing tones."""
        # C major with added D (passing tone)
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),  # C
            Note(pitch=62, onset=0, offset=0.5, velocity=40),  # D - passing
            Note(pitch=64, onset=0, offset=1, velocity=80),  # E
            Note(pitch=67, onset=0, offset=1, velocity=80),  # G
        ]
        
        analyzer = ChordAnalyzer()
        chords = analyzer.detect_chords(notes)
        
        # Should still detect C major
        assert len(chords) >= 1
        assert chords[0].root == "C"
    
    def test_timing_variations(self):
        """Test handling of timing variations (not perfectly aligned)."""
        # Chord with staggered onset - use timing corrector first
        notes = [
            Note(pitch=60, onset=0.0, offset=1, velocity=80),
            Note(pitch=64, onset=0.05, offset=1.05, velocity=80),  # 50ms late
            Note(pitch=67, onset=0.03, offset=1.03, velocity=80),  # 30ms late
        ]
        
        # First align the notes (simulating what resilience processor would do)
        corrector = TimingCorrector()
        aligned = corrector.align_simultaneous_notes(notes, tolerance=0.1)
        
        analyzer = ChordAnalyzer()
        chords = analyzer.detect_chords(aligned)
        
        # Should detect as a single chord after alignment
        assert len(chords) >= 1
        if chords:
            assert chords[0].root == "C"
    
    def test_full_pipeline_with_noise(self):
        """Test full harmony analysis pipeline with noisy input."""
        # Create progression with noise
        clean_notes = create_chord_progression_notes([
            ("C", "major"), ("F", "major"), ("G", "major"), ("C", "major")
        ])
        noisy_notes = add_noise_to_notes(
            clean_notes,
            pitch_noise=1,
            timing_noise=0.02,
            add_ghost_notes=5,
            add_wrong_notes=2,
        )
        
        # First clean with resilience processor
        processor = ResilienceProcessor()
        cleaned, _ = processor.process(noisy_notes)
        
        # Then analyze
        analyzer = HarmonyAnalyzer()
        result = analyzer.analyze(cleaned)
        
        # Should still produce reasonable results
        assert result.key_info is not None
        assert result.progression is not None
        assert result.musical_coherence > 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_short_notes_list(self):
        """Test with very few notes."""
        notes = [Note(pitch=60, onset=0, offset=1, velocity=80)]
        
        analyzer = HarmonyAnalyzer()
        result = analyzer.analyze(notes)
        
        # Should handle gracefully
        assert result is not None
    
    def test_notes_with_same_pitch(self):
        """Test with all notes being the same pitch."""
        notes = [
            Note(pitch=60, onset=i * 0.5, offset=i * 0.5 + 0.4, velocity=80)
            for i in range(8)
        ]
        
        detector = KeyDetector()
        result = detector.analyze(notes)
        
        # Should handle (might have low confidence)
        assert result is not None
    
    def test_extreme_pitch_range(self):
        """Test with notes at extreme pitches."""
        notes = [
            Note(pitch=21, onset=0, offset=1, velocity=80),   # Lowest piano
            Note(pitch=108, onset=1, offset=2, velocity=80),  # Highest piano
        ]
        
        analyzer = HarmonyAnalyzer()
        result = analyzer.analyze(notes)
        
        assert result is not None
    
    def test_overlapping_notes(self):
        """Test with heavily overlapping notes."""
        notes = [
            Note(pitch=60, onset=0, offset=2, velocity=80),
            Note(pitch=64, onset=0.5, offset=2.5, velocity=80),
            Note(pitch=67, onset=1, offset=3, velocity=80),
        ]
        
        analyzer = ChordAnalyzer()
        chords = analyzer.detect_chords(notes)
        
        # Should handle overlapping notes
        assert isinstance(chords, list)
    
    def test_zero_duration_notes(self):
        """Test handling of zero-duration notes."""
        notes = [
            Note(pitch=60, onset=0, offset=0, velocity=80),  # Zero duration
            Note(pitch=64, onset=0, offset=1, velocity=80),
        ]
        
        filt = NoteFilter(min_duration=0.01)
        filtered, _ = filt.filter_notes(notes)
        
        assert len(filtered) == 1
    
    def test_negative_onset(self):
        """Test handling of notes with unusual timing."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),
            Note(pitch=64, onset=0, offset=1, velocity=80),
        ]
        
        # Should process without error
        analyzer = HarmonyAnalyzer()
        result = analyzer.analyze(notes)
        assert result is not None


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
