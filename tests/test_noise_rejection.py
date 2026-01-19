"""Tests for noise rejection in transcription and chord detection.

These tests verify that the improved thresholds and filtering
correctly reject noise while accepting valid musical content.
"""

import numpy as np
import pytest
from pathlib import Path

from src.transcription import MonophonicTranscriber, PolyphonicTranscriber
from src.inference.chords import ChordAnalyzer
from src.core import Note


class TestNoiseRejection:
    """Test that various types of noise produce minimal false detections."""

    @pytest.fixture
    def sample_rate(self):
        return 22050

    @pytest.fixture
    def duration(self):
        return 3.0  # 3 seconds

    def generate_white_noise(self, duration: float, sr: int, amplitude: float = 0.1) -> np.ndarray:
        """Generate white noise."""
        n_samples = int(duration * sr)
        return np.random.randn(n_samples).astype(np.float32) * amplitude

    def generate_pink_noise(self, duration: float, sr: int, amplitude: float = 0.1) -> np.ndarray:
        """Generate pink noise (1/f noise)."""
        n_samples = int(duration * sr)
        # Simple pink noise approximation using filtered white noise
        white = np.random.randn(n_samples).astype(np.float32)

        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, 1/sr)
        freqs[0] = 1  # Avoid division by zero
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n_samples)

        # Normalize
        pink = pink / np.max(np.abs(pink)) * amplitude
        return pink.astype(np.float32)

    def generate_silence(self, duration: float, sr: int) -> np.ndarray:
        """Generate silence."""
        n_samples = int(duration * sr)
        return np.zeros(n_samples, dtype=np.float32)

    def generate_low_level_noise(self, duration: float, sr: int, amplitude: float = 0.005) -> np.ndarray:
        """Generate very low level noise (below threshold)."""
        return self.generate_white_noise(duration, sr, amplitude)

    # === Monophonic Transcriber Tests ===

    def test_monophonic_rejects_white_noise(self, sample_rate, duration):
        """Monophonic transcriber should detect very few notes from white noise."""
        audio = self.generate_white_noise(duration, sample_rate)
        transcriber = MonophonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect very few or no notes from pure noise
        assert len(notes) <= 2, f"Expected <=2 notes from white noise, got {len(notes)}"

    def test_monophonic_rejects_pink_noise(self, sample_rate, duration):
        """Monophonic transcriber should detect very few notes from pink noise."""
        audio = self.generate_pink_noise(duration, sample_rate)
        transcriber = MonophonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect very few or no notes
        assert len(notes) <= 3, f"Expected <=3 notes from pink noise, got {len(notes)}"

    def test_monophonic_rejects_silence(self, sample_rate, duration):
        """Monophonic transcriber should detect no notes from silence."""
        audio = self.generate_silence(duration, sample_rate)
        transcriber = MonophonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect no notes from silence
        assert len(notes) == 0, f"Expected 0 notes from silence, got {len(notes)}"

    def test_monophonic_rejects_low_level_noise(self, sample_rate, duration):
        """Monophonic transcriber should reject very low level noise."""
        audio = self.generate_low_level_noise(duration, sample_rate)
        transcriber = MonophonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect no notes from below-threshold noise
        assert len(notes) == 0, f"Expected 0 notes from low level noise, got {len(notes)}"

    # === Polyphonic Transcriber Tests ===

    def test_polyphonic_rejects_white_noise(self, sample_rate, duration):
        """Polyphonic transcriber should detect very few notes from white noise."""
        audio = self.generate_white_noise(duration, sample_rate)
        transcriber = PolyphonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect very few or no notes
        assert len(notes) <= 3, f"Expected <=3 notes from white noise, got {len(notes)}"

    def test_polyphonic_rejects_pink_noise(self, sample_rate, duration):
        """Polyphonic transcriber should detect very few notes from pink noise."""
        audio = self.generate_pink_noise(duration, sample_rate)
        transcriber = PolyphonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect very few or no notes
        assert len(notes) <= 5, f"Expected <=5 notes from pink noise, got {len(notes)}"

    def test_polyphonic_rejects_silence(self, sample_rate, duration):
        """Polyphonic transcriber should detect no notes from silence."""
        audio = self.generate_silence(duration, sample_rate)
        transcriber = PolyphonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect no notes
        assert len(notes) == 0, f"Expected 0 notes from silence, got {len(notes)}"

    # === Chord Analyzer Tests ===

    def test_chord_analyzer_rejects_low_velocity_notes(self):
        """Chord analyzer should reject notes with low velocity."""
        # Create notes with very low velocities (ghost notes)
        notes = [
            Note(pitch=60, onset=0.0, offset=1.0, velocity=10),  # Below threshold
            Note(pitch=64, onset=0.0, offset=1.0, velocity=15),  # Below threshold
            Note(pitch=67, onset=0.0, offset=1.0, velocity=20),  # Below threshold
        ]

        analyzer = ChordAnalyzer()
        chords = analyzer.detect_chords(notes)

        # Should not detect chord from low-velocity notes
        assert len(chords) == 0, f"Expected 0 chords from ghost notes, got {len(chords)}"

    def test_chord_analyzer_rejects_sparse_notes(self):
        """Chord analyzer should require minimum number of notes."""
        # Only 2 notes (below minimum of 3)
        notes = [
            Note(pitch=60, onset=0.0, offset=1.0, velocity=80),
            Note(pitch=64, onset=0.0, offset=1.0, velocity=80),
        ]

        analyzer = ChordAnalyzer()
        chords = analyzer.detect_chords(notes)

        # Should not detect chord with only 2 notes
        assert len(chords) == 0, f"Expected 0 chords from 2 notes, got {len(chords)}"

    def test_chord_analyzer_accepts_valid_triad(self):
        """Chord analyzer should accept a valid triad with good velocity."""
        # C major triad with good velocities
        notes = [
            Note(pitch=60, onset=0.0, offset=1.0, velocity=80),  # C
            Note(pitch=64, onset=0.0, offset=1.0, velocity=80),  # E
            Note(pitch=67, onset=0.0, offset=1.0, velocity=80),  # G
        ]

        analyzer = ChordAnalyzer()
        chords = analyzer.detect_chords(notes)

        # Should detect the C major chord
        assert len(chords) == 1, f"Expected 1 chord, got {len(chords)}"
        assert chords[0].root == "C", f"Expected C root, got {chords[0].root}"
        assert chords[0].quality == "major", f"Expected major quality, got {chords[0].quality}"


class TestThresholdEffectiveness:
    """Test that thresholds are effective at filtering noise."""

    @pytest.fixture
    def sample_rate(self):
        return 22050

    def generate_pure_tone(self, freq: float, duration: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
        """Generate a pure sine tone."""
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        return np.sin(2 * np.pi * freq * t) * amplitude

    def test_monophonic_detects_pure_tone(self, sample_rate):
        """Monophonic transcriber should detect a clear pure tone."""
        # A4 = 440 Hz
        audio = self.generate_pure_tone(440.0, 1.0, sample_rate)
        transcriber = MonophonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect at least one note
        assert len(notes) >= 1, f"Expected at least 1 note from pure tone, got {len(notes)}"

        # The note should be close to A4 (MIDI 69)
        if notes:
            pitches = [n.pitch for n in notes]
            closest_to_a4 = min(pitches, key=lambda p: abs(p - 69))
            assert abs(closest_to_a4 - 69) <= 2, f"Expected pitch near 69 (A4), got {closest_to_a4}"

    def test_polyphonic_detects_chord(self, sample_rate):
        """Polyphonic transcriber should detect a clear chord."""
        # C major chord: C4 (262 Hz), E4 (330 Hz), G4 (392 Hz)
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        audio = (
            np.sin(2 * np.pi * 261.63 * t) * 0.3 +  # C4
            np.sin(2 * np.pi * 329.63 * t) * 0.3 +  # E4
            np.sin(2 * np.pi * 392.00 * t) * 0.3    # G4
        ).astype(np.float32)

        transcriber = PolyphonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should detect at least 2 notes
        assert len(notes) >= 2, f"Expected at least 2 notes from chord, got {len(notes)}"

    def test_rms_threshold_filters_quiet_segments(self, sample_rate):
        """RMS threshold should filter out quiet segments."""
        # Create audio with loud note followed by very quiet section
        loud = self.generate_pure_tone(440.0, 0.5, sample_rate, amplitude=0.5)
        quiet = self.generate_pure_tone(440.0, 0.5, sample_rate, amplitude=0.005)  # Below RMS threshold
        audio = np.concatenate([loud, quiet])

        transcriber = MonophonicTranscriber()
        notes = transcriber.transcribe(audio, sample_rate)

        # Should only detect note from loud section, not quiet
        assert len(notes) <= 2, f"Expected <=2 notes, got {len(notes)}"


class TestConfigurableThresholds:
    """Test that thresholds can be configured to be stricter or more lenient."""

    @pytest.fixture
    def sample_rate(self):
        return 22050

    def generate_moderate_noise(self, duration: float, sr: int) -> np.ndarray:
        """Generate moderate level noise."""
        n_samples = int(duration * sr)
        return np.random.randn(n_samples).astype(np.float32) * 0.15

    def test_stricter_confidence_threshold(self, sample_rate):
        """Higher confidence threshold should detect fewer notes."""
        audio = self.generate_moderate_noise(2.0, sample_rate)

        lenient = MonophonicTranscriber(pitch_confidence_threshold=0.5)
        strict = MonophonicTranscriber(pitch_confidence_threshold=0.8)

        lenient_notes = lenient.transcribe(audio, sample_rate)
        strict_notes = strict.transcribe(audio, sample_rate)

        # Strict should detect same or fewer notes
        assert len(strict_notes) <= len(lenient_notes), \
            f"Strict ({len(strict_notes)}) should detect <= lenient ({len(lenient_notes)})"

    def test_stricter_rms_threshold(self, sample_rate):
        """Higher RMS threshold should detect fewer notes."""
        audio = self.generate_moderate_noise(2.0, sample_rate)

        lenient = MonophonicTranscriber(min_rms_threshold=0.005)
        strict = MonophonicTranscriber(min_rms_threshold=0.05)

        lenient_notes = lenient.transcribe(audio, sample_rate)
        strict_notes = strict.transcribe(audio, sample_rate)

        # Strict should detect same or fewer notes
        assert len(strict_notes) <= len(lenient_notes), \
            f"Strict ({len(strict_notes)}) should detect <= lenient ({len(lenient_notes)})"

    def test_chord_confidence_threshold(self):
        """Chord analyzer confidence threshold should filter weak matches."""
        # Create notes that form an ambiguous chord
        notes = [
            Note(pitch=60, onset=0.0, offset=1.0, velocity=50),
            Note(pitch=63, onset=0.0, offset=1.0, velocity=50),  # Minor third
            Note(pitch=68, onset=0.0, offset=1.0, velocity=50),  # Augmented fifth
        ]

        lenient = ChordAnalyzer(min_chord_confidence=0.2)
        strict = ChordAnalyzer(min_chord_confidence=0.6)

        lenient_chords = lenient.detect_chords(notes)
        strict_chords = strict.detect_chords(notes)

        # Strict should detect same or fewer chords
        assert len(strict_chords) <= len(lenient_chords), \
            f"Strict ({len(strict_chords)}) should detect <= lenient ({len(lenient_chords)})"
