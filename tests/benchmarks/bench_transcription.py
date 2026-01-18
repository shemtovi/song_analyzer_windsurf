"""Benchmark tests for transcription pipeline."""

import pytest
import numpy as np
from pathlib import Path

from .bench_framework import BenchmarkRunner, BenchmarkReport


def generate_test_audio(duration: float = 30.0, sr: int = 22050) -> np.ndarray:
    """Generate synthetic test audio with multiple notes.

    Creates a series of sine waves at different frequencies to simulate
    a simple melody for benchmarking purposes.
    """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Generate a simple melody with multiple notes
    audio = np.zeros_like(t)
    note_duration = 0.5  # seconds per note
    notes_per_second = 1 / note_duration

    # A simple ascending/descending pattern
    midi_notes = [60, 62, 64, 65, 67, 65, 64, 62]  # C4 to G4 and back

    for i, time_point in enumerate(np.arange(0, duration, note_duration)):
        note_idx = i % len(midi_notes)
        midi = midi_notes[note_idx]
        freq = 440.0 * (2 ** ((midi - 69) / 12.0))

        # Create envelope
        start_sample = int(time_point * sr)
        end_sample = min(int((time_point + note_duration) * sr), len(t))

        if start_sample >= len(t):
            break

        note_t = t[start_sample:end_sample] - time_point

        # Simple ADSR envelope
        envelope = np.ones(len(note_t))
        attack = int(0.01 * sr)
        release = int(0.05 * sr)
        if len(envelope) > attack:
            envelope[:attack] = np.linspace(0, 1, attack)
        if len(envelope) > release:
            envelope[-release:] = np.linspace(1, 0, release)

        # Add note with harmonics
        note = np.sin(2 * np.pi * freq * note_t) * 0.6
        note += np.sin(2 * np.pi * freq * 2 * note_t) * 0.3  # 2nd harmonic
        note += np.sin(2 * np.pi * freq * 3 * note_t) * 0.1  # 3rd harmonic

        audio[start_sample:end_sample] += note * envelope

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8

    return audio


class TestTranscriptionBenchmarks:
    """Benchmark tests for the transcription pipeline."""

    @pytest.fixture
    def test_audio_10s(self):
        """10-second test audio."""
        return generate_test_audio(duration=10.0)

    @pytest.fixture
    def test_audio_30s(self):
        """30-second test audio."""
        return generate_test_audio(duration=30.0)

    @pytest.fixture
    def benchmark_runner(self):
        """Create a fresh benchmark runner."""
        return BenchmarkRunner(name="transcription_benchmark")

    def test_monophonic_transcription_benchmark(self, test_audio_10s, benchmark_runner):
        """Benchmark monophonic transcription on 10s audio."""
        from src.transcription import MonophonicTranscriber

        sr = 22050
        duration = len(test_audio_10s) / sr

        transcriber = MonophonicTranscriber()
        notes = benchmark_runner.time_stage(
            "monophonic_transcription",
            transcriber.transcribe,
            test_audio_10s,
            sr,
            input_duration=duration,
        )

        # Verify we got results
        assert isinstance(notes, list)
        assert len(notes) > 0

        # Print summary
        benchmark_runner.print_summary()

        # Check no major regression
        regressions = benchmark_runner.check_regression(threshold_percent=50.0)
        if regressions:
            print(f"Warning: Potential regressions: {regressions}")

    def test_polyphonic_transcription_benchmark(self, test_audio_10s, benchmark_runner):
        """Benchmark polyphonic transcription on 10s audio."""
        from src.transcription import PolyphonicTranscriber

        sr = 22050
        duration = len(test_audio_10s) / sr

        transcriber = PolyphonicTranscriber(device="cpu")
        notes = benchmark_runner.time_stage(
            "polyphonic_transcription",
            transcriber.transcribe,
            test_audio_10s,
            sr,
            input_duration=duration,
        )

        assert isinstance(notes, list)

        benchmark_runner.print_summary()

    def test_full_pipeline_benchmark(self, test_audio_30s, benchmark_runner):
        """Benchmark the full transcription pipeline on 30s audio."""
        from src.input import AudioLoader
        from src.transcription import PolyphonicTranscriber
        from src.processing import Quantizer, NoteCleanup
        from src.output import MIDIExporter
        from src.inference import KeyDetector

        sr = 22050
        duration = len(test_audio_30s) / sr

        # Transcribe
        transcriber = PolyphonicTranscriber(device="cpu")
        notes = benchmark_runner.time_stage(
            "transcribe",
            transcriber.transcribe,
            test_audio_30s,
            sr,
            input_duration=duration,
        )

        if not notes:
            pytest.skip("No notes detected in synthetic audio")

        # Key detection
        key_detector = KeyDetector()
        key_info = benchmark_runner.time_stage(
            "key_detection",
            key_detector.analyze,
            notes,
            input_duration=duration,
        )

        # Quantize
        quantizer = Quantizer(tempo=120)
        notes = benchmark_runner.time_stage(
            "quantize",
            quantizer.quantize,
            notes,
            input_duration=duration,
        )

        # Cleanup
        cleaner = NoteCleanup()
        notes = benchmark_runner.time_stage(
            "cleanup",
            cleaner.cleanup,
            notes,
            input_duration=duration,
        )

        # Export (to temp file)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            temp_path = f.name

        try:
            exporter = MIDIExporter(tempo=120.0)
            benchmark_runner.time_stage(
                "midi_export",
                exporter.export,
                notes,
                temp_path,
                input_duration=duration,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        # Generate report
        benchmark_runner.print_summary()
        report = benchmark_runner.generate_report()

        # Verify report structure
        assert report.total_time > 0
        assert len(report.stages) >= 4

    def test_feature_extraction_benchmark(self, test_audio_10s, benchmark_runner):
        """Benchmark feature extraction methods."""
        from src.analysis import FeatureExtractor

        sr = 22050
        duration = len(test_audio_10s) / sr

        extractor = FeatureExtractor(sr=sr)

        # Mel spectrogram
        mel = benchmark_runner.time_stage(
            "mel_spectrogram",
            extractor.mel_spectrogram,
            test_audio_10s,
            input_duration=duration,
        )
        assert mel.shape[0] > 0

        # CQT
        cqt = benchmark_runner.time_stage(
            "cqt",
            extractor.cqt,
            test_audio_10s,
            input_duration=duration,
        )
        assert cqt.shape[0] > 0

        # Chromagram
        chroma = benchmark_runner.time_stage(
            "chromagram",
            extractor.chromagram,
            test_audio_10s,
            input_duration=duration,
        )
        assert chroma.shape[0] == 12

        benchmark_runner.print_summary()


def run_benchmarks_and_save():
    """Run all benchmarks and save report to file."""
    runner = BenchmarkRunner(name="full_benchmark_suite")
    audio = generate_test_audio(duration=30.0)
    sr = 22050
    duration = len(audio) / sr

    from src.transcription import MonophonicTranscriber, PolyphonicTranscriber
    from src.processing import Quantizer, NoteCleanup
    from src.inference import KeyDetector

    # Run benchmarks
    mono = MonophonicTranscriber()
    notes = runner.time_stage("monophonic", mono.transcribe, audio, sr, input_duration=duration)

    poly = PolyphonicTranscriber(device="cpu")
    notes = runner.time_stage("polyphonic", poly.transcribe, audio, sr, input_duration=duration)

    if notes:
        detector = KeyDetector()
        runner.time_stage("key_detection", detector.analyze, notes, input_duration=duration)

        quantizer = Quantizer(tempo=120)
        notes = runner.time_stage("quantize", quantizer.quantize, notes, input_duration=duration)

        cleaner = NoteCleanup()
        runner.time_stage("cleanup", cleaner.cleanup, notes, input_duration=duration)

    # Save report
    report_path = Path("tests/benchmarks/benchmark_report.json")
    runner.save_report(report_path)
    runner.print_summary()

    return runner.generate_report()


if __name__ == "__main__":
    run_benchmarks_and_save()
