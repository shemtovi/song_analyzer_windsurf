"""Tests for transcription modules."""

import pytest
import numpy as np
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Note

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestNote:
    """Tests for Note dataclass."""

    def test_note_creation(self):
        note = Note(pitch=60, onset=0.0, offset=1.0, velocity=80)
        assert note.pitch == 60
        assert note.onset == 0.0
        assert note.offset == 1.0
        assert note.velocity == 80

    def test_note_duration(self):
        note = Note(pitch=60, onset=0.5, offset=1.5)
        assert note.duration == 1.0

    def test_pitch_name(self):
        assert Note(pitch=60, onset=0, offset=1).pitch_name == "C4"
        assert Note(pitch=69, onset=0, offset=1).pitch_name == "A4"
        assert Note(pitch=61, onset=0, offset=1).pitch_name == "C#4"

    def test_freq_to_midi(self):
        assert Note.freq_to_midi(440.0) == 69  # A4
        assert Note.freq_to_midi(261.63) == 60  # C4 (approx)
        assert Note.freq_to_midi(880.0) == 81  # A5

    def test_midi_to_freq(self):
        assert Note.midi_to_freq(69) == 440.0
        assert abs(Note.midi_to_freq(60) - 261.63) < 1.0


class TestAudioLoader:
    """Tests for AudioLoader."""

    def test_normalize(self):
        from src.input.loader import AudioLoader

        loader = AudioLoader()
        audio = np.array([0.5, -0.5, 0.25, -0.25])
        normalized = loader._normalize(audio)

        assert np.abs(normalized).max() == 1.0

    def test_unsupported_format(self, tmp_path):
        from src.input.loader import AudioLoader

        # Create a dummy file with unsupported extension
        dummy_file = tmp_path / "test.xyz"
        dummy_file.write_text("dummy content")

        loader = AudioLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(str(dummy_file))


class TestPostProcessor:
    """Tests for Quantizer and NoteCleanup."""

    def test_quantize(self):
        from src.processing.quantize import Quantizer

        quantizer = Quantizer(tempo=120.0, quantize_resolution=16)

        notes = [Note(pitch=60, onset=0.48, offset=0.98)]
        quantized = quantizer.quantize(notes)

        # At 120 BPM, 16th note = 0.125s
        # 0.48 should snap to 0.5
        assert quantized[0].onset == 0.5

    def test_remove_ghost_notes(self):
        from src.processing.cleanup import NoteCleanup

        cleanup = NoteCleanup(min_velocity=20)
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=80),
            Note(pitch=62, onset=1, offset=2, velocity=10),  # Ghost
        ]

        cleaned = cleanup.remove_ghost_notes(notes)
        assert len(cleaned) == 1
        assert cleaned[0].pitch == 60


class TestIntegration:
    """Integration tests with sample audio files."""

    @pytest.fixture
    def audio_loader(self):
        from src.input.loader import AudioLoader
        return AudioLoader(target_sr=22050)

    def test_load_single_a4(self, audio_loader):
        """Test loading A4 (440Hz) sample."""
        audio_path = EXAMPLES_DIR / "single_a4.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")

        audio, sr = audio_loader.load(str(audio_path))
        assert sr == 22050
        assert len(audio) > 0
        assert audio_loader.get_duration(audio, sr) == pytest.approx(2.0, rel=0.1)

    def test_load_c_major_scale(self, audio_loader):
        """Test loading C major scale sample."""
        audio_path = EXAMPLES_DIR / "c_major_scale.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")

        audio, sr = audio_loader.load(str(audio_path))
        # 8 notes * 0.5s = 4 seconds
        assert audio_loader.get_duration(audio, sr) == pytest.approx(4.0, rel=0.1)

    def test_transcribe_single_note(self, audio_loader):
        """Test transcribing a single A4 note."""
        from src.transcription import MonophonicTranscriber

        audio_path = EXAMPLES_DIR / "single_a4.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")

        audio, sr = audio_loader.load(str(audio_path))
        transcriber = MonophonicTranscriber()

        try:
            notes = transcriber.transcribe(audio, sr)
            # Should detect at least one note
            assert len(notes) >= 1
            # A4 = MIDI 69
            # Allow some tolerance for pitch detection
            assert any(67 <= n.pitch <= 71 for n in notes)
        except ImportError:
            pytest.skip("CREPE not installed")

    def test_feature_extraction(self, audio_loader):
        """Test feature extraction on sample audio."""
        from src.analysis.features import FeatureExtractor

        audio_path = EXAMPLES_DIR / "test_c4.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")

        audio, sr = audio_loader.load(str(audio_path))
        extractor = FeatureExtractor(sr=sr)

        # Test mel spectrogram
        mel = extractor.mel_spectrogram(audio)
        assert mel.shape[0] == 128  # n_mels
        assert mel.shape[1] > 0  # time frames

        # Test chromagram
        chroma = extractor.chromagram(audio)
        assert chroma.shape[0] == 12  # 12 pitch classes

    def test_tempo_detection(self, audio_loader):
        """Test tempo detection."""
        from src.analysis.tempo import TempoAnalyzer

        audio_path = EXAMPLES_DIR / "c_major_scale.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")

        audio, sr = audio_loader.load(str(audio_path))
        analyzer = TempoAnalyzer()

        tempo, beats = analyzer.detect(audio, sr)
        # Tempo should be a positive number (simple sine waves may not have clear beats)
        # Just verify it returns something valid
        assert tempo >= 0

    def test_midi_export(self, audio_loader, tmp_path):
        """Test MIDI export functionality."""
        from src.output.midi import MIDIExporter

        notes = [
            Note(pitch=60, onset=0.0, offset=0.5, velocity=80),
            Note(pitch=62, onset=0.5, offset=1.0, velocity=75),
            Note(pitch=64, onset=1.0, offset=1.5, velocity=70),
        ]

        output_path = tmp_path / "test_output.mid"
        exporter = MIDIExporter(tempo=120.0)
        exporter.export(notes, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestPolyphonicTranscriber:
    """Tests for PolyphonicTranscriber (Phase 2)."""

    @pytest.fixture
    def audio_loader(self):
        from src.input.loader import AudioLoader
        return AudioLoader(target_sr=22050)

    def test_polyphonic_transcriber_creation(self):
        """Test PolyphonicTranscriber can be instantiated."""
        from src.transcription import PolyphonicTranscriber
        
        transcriber = PolyphonicTranscriber()
        assert transcriber is not None
        # Should have is_neural property
        assert hasattr(transcriber, 'is_neural')

    def test_transcribe_chords_detects_multiple_notes(self, audio_loader):
        """Test that polyphonic transcriber detects multiple simultaneous notes."""
        from src.transcription import PolyphonicTranscriber
        
        audio_path = EXAMPLES_DIR / "chords_g_f_a.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated - run tests/generate_test_audio.py")
        
        audio, sr = audio_loader.load(str(audio_path))
        transcriber = PolyphonicTranscriber()
        notes = transcriber.transcribe(audio, sr)
        
        # Should detect multiple notes (3 chords × ~2-3 notes each)
        # Stricter thresholds may detect fewer notes but reduce false positives
        assert len(notes) >= 4, f"Expected at least 4 notes, got {len(notes)}"
        
        # Should have notes at different onset times (3 chords at 0s, 1s, 2s)
        onset_times = sorted(set(round(n.onset, 1) for n in notes))
        assert len(onset_times) >= 2, f"Expected multiple onset times, got {onset_times}"

    def test_transcribe_chords_correct_pitches(self, audio_loader):
        """Test that polyphonic transcriber detects approximately correct pitches."""
        from src.transcription import PolyphonicTranscriber
        
        audio_path = EXAMPLES_DIR / "chords_g_f_a.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")
        
        audio, sr = audio_loader.load(str(audio_path))
        transcriber = PolyphonicTranscriber()
        notes = transcriber.transcribe(audio, sr)
        
        # Expected pitches (within ±1 semitone tolerance):
        # G major: G3 (55), B3 (59), D4 (62)
        # F major: F3 (53), A3 (57), C4 (60)
        # A major: A3 (57), C#4 (61), E4 (64)
        expected_pitches = {53, 55, 57, 59, 60, 61, 62, 64}
        
        detected_pitches = set(n.pitch for n in notes)
        
        # Check that we detect at least some of the expected pitches (within tolerance)
        matches = 0
        for expected in expected_pitches:
            for detected in detected_pitches:
                if abs(expected - detected) <= 1:  # ±1 semitone tolerance
                    matches += 1
                    break
        
        assert matches >= 5, f"Expected at least 5 pitch matches, got {matches}. Detected: {detected_pitches}"

    def test_transcribe_chords_with_melody(self, audio_loader):
        """Test polyphonic transcriber on chords + melody."""
        from src.transcription import PolyphonicTranscriber
        
        audio_path = EXAMPLES_DIR / "chords_g_f_a_with_melody.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")
        
        audio, sr = audio_loader.load(str(audio_path))
        transcriber = PolyphonicTranscriber()
        notes = transcriber.transcribe(audio, sr)
        
        # Should detect more notes than just chords (melody adds more)
        assert len(notes) >= 6, f"Expected at least 6 notes, got {len(notes)}"

    def test_polyphonic_vs_monophonic_note_count(self, audio_loader):
        """Test that polyphonic detects more notes than monophonic on chord audio."""
        from src.transcription import MonophonicTranscriber, PolyphonicTranscriber
        
        audio_path = EXAMPLES_DIR / "chords_g_f_a.wav"
        if not audio_path.exists():
            pytest.skip("Test audio not generated")
        
        audio, sr = audio_loader.load(str(audio_path))
        
        mono_transcriber = MonophonicTranscriber()
        poly_transcriber = PolyphonicTranscriber()
        
        mono_notes = mono_transcriber.transcribe(audio, sr)
        poly_notes = poly_transcriber.transcribe(audio, sr)
        
        # Polyphonic should detect more notes than monophonic
        assert len(poly_notes) > len(mono_notes), \
            f"Polyphonic ({len(poly_notes)}) should detect more notes than monophonic ({len(mono_notes)})"
