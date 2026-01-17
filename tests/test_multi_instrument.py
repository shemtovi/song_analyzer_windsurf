"""Tests for multi-instrument transcription pipeline.

Tests the Phase 3 multi-instrument capabilities:
- Source separation (Demucs)
- Instrument classification
- Per-stem transcription
- Multi-instrument transcriber integration
"""

import pytest
import numpy as np
from pathlib import Path

from src.separation import (
    SourceSeparator,
    SeparatedStems,
    StemType,
    StemAudio,
    InstrumentClassifier,
    InstrumentCategory,
    InstrumentInfo,
)
from src.transcription import (
    MultiInstrumentTranscriber,
    MultiInstrumentTranscription,
    StemTranscription,
)
from src.core import Note


@pytest.fixture
def sample_rate():
    return 44100


@pytest.fixture
def sine_wave(sample_rate):
    """Generate a simple sine wave."""
    duration = 2.0
    freq = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t) * 0.5


@pytest.fixture
def bass_tone(sample_rate):
    """Generate a low frequency bass tone."""
    duration = 2.0
    freq = 82.41
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t) * 0.5


@pytest.fixture
def noise_burst(sample_rate):
    """Generate noise burst."""
    duration = 0.5
    samples = int(sample_rate * duration)
    noise = np.random.randn(samples).astype(np.float32) * 0.3
    envelope = np.exp(-np.linspace(0, 10, samples))
    return noise * envelope


class TestStemType:
    def test_all_stems(self):
        stems = StemType.all_stems()
        assert len(stems) == 4
        assert StemType.DRUMS in stems
        assert StemType.BASS in stems

    def test_melodic_stems(self):
        melodic = StemType.melodic_stems()
        assert StemType.DRUMS not in melodic
        assert StemType.BASS in melodic


class TestStemAudio:
    def test_creation(self, sine_wave, sample_rate):
        stem = StemAudio(StemType.VOCALS, sine_wave, sample_rate)
        assert stem.stem_type == StemType.VOCALS
        assert stem.sample_rate == sample_rate

    def test_duration(self, sine_wave, sample_rate):
        stem = StemAudio(StemType.VOCALS, sine_wave, sample_rate)
        expected = len(sine_wave) / sample_rate
        assert abs(stem.duration - expected) < 0.001

    def test_to_mono(self, sine_wave, sample_rate):
        stem = StemAudio(StemType.VOCALS, sine_wave, sample_rate)
        mono = stem.to_mono()
        assert len(mono.shape) == 1

    def test_is_silent(self, sample_rate):
        silent = np.zeros(sample_rate, dtype=np.float32)
        stem = StemAudio(StemType.VOCALS, silent, sample_rate)
        assert stem.is_silent()


class TestSeparatedStems:
    def test_get_stem(self, sine_wave, sample_rate):
        vocals = StemAudio(StemType.VOCALS, sine_wave, sample_rate)
        separated = SeparatedStems(stems={StemType.VOCALS: vocals})
        assert separated.get_stem(StemType.VOCALS) == vocals
        assert separated.get_stem("vocals") == vocals

    def test_get_active_stems(self, sine_wave, sample_rate):
        active = StemAudio(StemType.VOCALS, sine_wave, sample_rate)
        silent = StemAudio(StemType.DRUMS, np.zeros_like(sine_wave), sample_rate)
        separated = SeparatedStems(stems={StemType.VOCALS: active, StemType.DRUMS: silent})
        active_stems = separated.get_active_stems()
        assert len(active_stems) == 1

    def test_remix(self, sine_wave, sample_rate):
        vocals = StemAudio(StemType.VOCALS, sine_wave, sample_rate)
        bass = StemAudio(StemType.BASS, sine_wave * 0.5, sample_rate)
        separated = SeparatedStems(stems={StemType.VOCALS: vocals, StemType.BASS: bass})
        mixed = separated.remix()
        expected = sine_wave + sine_wave * 0.5
        np.testing.assert_array_almost_equal(mixed, expected)


class TestInstrumentClassifier:
    def test_classify_with_hint(self, sine_wave, sample_rate):
        classifier = InstrumentClassifier()
        info = classifier.classify(sine_wave, sample_rate, stem_hint="vocals")
        assert info.category == InstrumentCategory.VOCALS
        assert info.is_melodic
        assert info.is_monophonic

    def test_classify_bass(self, bass_tone, sample_rate):
        classifier = InstrumentClassifier()
        info = classifier.classify(bass_tone, sample_rate, stem_hint="bass")
        assert info.category == InstrumentCategory.BASS

    def test_transcription_mode(self, sine_wave, sample_rate):
        classifier = InstrumentClassifier()
        info = classifier.classify(sine_wave, sample_rate, stem_hint="vocals")
        assert info.transcription_mode == "monophonic"

    def test_drums_not_melodic(self, noise_burst, sample_rate):
        classifier = InstrumentClassifier()
        info = classifier.classify(noise_burst, sample_rate, stem_hint="drums")
        assert info.category == InstrumentCategory.DRUMS
        assert not info.is_melodic


class TestSourceSeparator:
    def test_init(self):
        separator = SourceSeparator()
        assert separator.model_name == "htdemucs"

    def test_available_models(self):
        separator = SourceSeparator()
        models = separator.available_models
        assert "htdemucs" in models

    def test_fallback_separation(self, sine_wave, sample_rate):
        """Test fallback separation when Demucs is not available."""
        separator = SourceSeparator()
        # Force fallback
        separator._demucs_available = False
        result = separator.separate(sine_wave, sample_rate)
        assert isinstance(result, SeparatedStems)
        assert len(result.stems) > 0


class TestMultiInstrumentTranscriber:
    def test_init(self):
        transcriber = MultiInstrumentTranscriber()
        assert transcriber.skip_drums == False

    def test_transcribe_returns_notes(self, sine_wave, sample_rate):
        transcriber = MultiInstrumentTranscriber()
        transcriber.separator._demucs_available = False
        notes = transcriber.transcribe(sine_wave, sample_rate)
        assert isinstance(notes, list)

    def test_transcribe_multi(self, sine_wave, sample_rate):
        transcriber = MultiInstrumentTranscriber()
        transcriber.separator._demucs_available = False
        result = transcriber.transcribe_multi(sine_wave, sample_rate)
        assert isinstance(result, MultiInstrumentTranscription)
        assert result.total_time > 0


class TestStemTranscription:
    def test_creation(self):
        info = InstrumentInfo(
            category=InstrumentCategory.VOCALS,
            confidence=0.9,
            is_melodic=True,
            is_monophonic=True,
        )
        notes = [Note(pitch=60, onset=0.0, offset=1.0)]
        stem_trans = StemTranscription(
            stem_type=StemType.VOCALS,
            instrument_info=info,
            notes=notes,
        )
        assert stem_trans.note_count == 1
        assert stem_trans.instrument_name == "Vocals"


class TestMultiInstrumentTranscription:
    def test_get_all_notes(self):
        info = InstrumentInfo(InstrumentCategory.VOCALS, 0.9)
        notes1 = [Note(pitch=60, onset=0.0, offset=1.0)]
        notes2 = [Note(pitch=64, onset=0.5, offset=1.5)]
        
        stem1 = StemTranscription(StemType.VOCALS, info, notes1)
        stem2 = StemTranscription(StemType.OTHER, info, notes2)
        
        result = MultiInstrumentTranscription(
            stems={StemType.VOCALS: stem1, StemType.OTHER: stem2}
        )
        
        all_notes = result.get_all_notes()
        assert len(all_notes) == 2
        assert all_notes[0].onset <= all_notes[1].onset

    def test_total_notes(self):
        info = InstrumentInfo(InstrumentCategory.VOCALS, 0.9)
        notes = [Note(pitch=60, onset=0.0, offset=1.0)] * 5
        stem = StemTranscription(StemType.VOCALS, info, notes)
        result = MultiInstrumentTranscription(stems={StemType.VOCALS: stem})
        assert result.total_notes == 5

    def test_summary(self):
        info = InstrumentInfo(InstrumentCategory.VOCALS, 0.9)
        notes = [Note(pitch=60, onset=0.0, offset=1.0)]
        stem = StemTranscription(StemType.VOCALS, info, notes)
        result = MultiInstrumentTranscription(stems={StemType.VOCALS: stem})
        
        summary = result.summary()
        assert "total_notes" in summary
        assert "stems" in summary
