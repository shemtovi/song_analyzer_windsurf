"""Tests for enhanced cleanup pipeline (Phase C improvements).

Tests for:
- C.1: Auto-aggressive mode for noisy input
- C.2: Improved harmonic detection
- C.3: Adaptive velocity threshold
- C.4: Note density filter
"""

import pytest
import numpy as np
from src.core import Note
from src.processing.cleanup import NoteCleanup, CleanupConfig, CleanupStats


class TestAutoAggressiveMode:
    """Test C.1: Auto-aggressive mode for noisy input."""

    def test_auto_aggressive_triggers_on_high_density(self):
        """Auto-aggressive mode should trigger when note density is high."""
        # Create many notes in a short time (simulating noise)
        notes = [
            Note(pitch=60 + i % 12, onset=i * 0.05, offset=i * 0.05 + 0.1, velocity=50)
            for i in range(50)
        ]
        # 50 notes in 2.5 seconds = 20 notes/sec (above threshold of 15)

        config = CleanupConfig(auto_aggressive=True, noise_threshold_notes_per_sec=15)
        cleanup = NoteCleanup(config=config)

        result, stats = cleanup.cleanup(notes, return_stats=True)

        assert stats.auto_aggressive_enabled, "Auto-aggressive should be enabled"
        assert stats.detected_notes_per_sec > 15, f"Should detect high density: {stats.detected_notes_per_sec}"

    def test_auto_aggressive_does_not_trigger_on_normal_density(self):
        """Auto-aggressive mode should not trigger for normal note density."""
        # Create reasonable number of notes
        notes = [
            Note(pitch=60, onset=i * 0.5, offset=i * 0.5 + 0.4, velocity=80)
            for i in range(10)
        ]
        # 10 notes in 5 seconds = 2 notes/sec (below threshold)

        config = CleanupConfig(auto_aggressive=True, noise_threshold_notes_per_sec=15)
        cleanup = NoteCleanup(config=config)

        result, stats = cleanup.cleanup(notes, return_stats=True)

        assert not stats.auto_aggressive_enabled, "Auto-aggressive should not be enabled"

    def test_auto_aggressive_can_be_disabled(self):
        """Auto-aggressive mode can be disabled."""
        notes = [
            Note(pitch=60 + i % 12, onset=i * 0.05, offset=i * 0.05 + 0.1, velocity=50)
            for i in range(50)
        ]

        config = CleanupConfig(auto_aggressive=False)
        cleanup = NoteCleanup(config=config)

        result, stats = cleanup.cleanup(notes, return_stats=True)

        assert not stats.auto_aggressive_enabled, "Auto-aggressive should be disabled"


class TestImprovedHarmonicDetection:
    """Test C.2: Improved harmonic detection."""

    def test_harmonic_removal_with_extended_ratios(self):
        """Should detect harmonics with extended ratios (2-8x)."""
        # Create fundamental and harmonics
        fundamental = Note(pitch=36, onset=0.0, offset=1.0, velocity=100)  # C2
        harmonic_2x = Note(pitch=48, onset=0.0, offset=1.0, velocity=60)  # C3 (2x)
        harmonic_3x = Note(pitch=55, onset=0.0, offset=1.0, velocity=40)  # G3 (3x)
        harmonic_4x = Note(pitch=60, onset=0.0, offset=1.0, velocity=30)  # C4 (4x)

        notes = [fundamental, harmonic_2x, harmonic_3x, harmonic_4x]

        config = CleanupConfig(
            remove_harmonics=True,
            harmonic_ratios=(2, 3, 4, 5, 6, 7, 8),
        )
        cleanup = NoteCleanup(config=config)

        result = cleanup.remove_harmonics(notes)

        # Should keep fundamental and remove harmonics
        assert len(result) < len(notes), f"Should remove some harmonics, got {len(result)}"
        assert any(n.pitch == 36 for n in result), "Should keep fundamental"

    def test_harmonic_ratios_configurable(self):
        """Harmonic ratios should be configurable."""
        config = CleanupConfig(harmonic_ratios=(2, 3))
        assert config.harmonic_ratios == (2, 3)

        config2 = CleanupConfig(harmonic_ratios=(2, 3, 4, 5, 6, 7, 8))
        assert config2.harmonic_ratios == (2, 3, 4, 5, 6, 7, 8)


class TestAdaptiveVelocityThreshold:
    """Test C.3: Adaptive velocity threshold."""

    def test_adaptive_velocity_filters_low_percentile(self):
        """Adaptive velocity should filter notes below percentile."""
        # Create notes with varying velocities
        notes = [
            Note(pitch=60, onset=i * 0.1, offset=i * 0.1 + 0.1, velocity=10 + i * 10)
            for i in range(10)
        ]
        # Velocities: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

        config = CleanupConfig(
            adaptive_velocity=True,
            adaptive_velocity_percentile=20,  # Filter bottom 20%
            min_velocity=1,  # Don't use fixed threshold
        )
        cleanup = NoteCleanup(config=config)

        result, stats = cleanup.cleanup(notes, return_stats=True)

        assert stats.removed_adaptive_velocity > 0, "Should remove some low-velocity notes"
        # Bottom 20% = velocities 10, 20 should be removed
        assert len(result) < len(notes), f"Should filter some notes: {len(result)} vs {len(notes)}"

    def test_adaptive_velocity_uses_higher_of_fixed_and_adaptive(self):
        """Should use the higher of fixed min_velocity and adaptive threshold."""
        # Notes with low velocities
        notes = [
            Note(pitch=60, onset=i * 0.1, offset=i * 0.1 + 0.1, velocity=15 + i)
            for i in range(10)
        ]
        # Velocities: 15-24, percentile 20 would be ~17

        config = CleanupConfig(
            adaptive_velocity=True,
            adaptive_velocity_percentile=20,
            min_velocity=25,  # Fixed threshold higher than adaptive
        )
        cleanup = NoteCleanup(config=config)

        result, stats = cleanup.cleanup(notes, return_stats=True)

        # All notes should be removed because fixed threshold (25) is higher than all velocities
        remaining_velocities = [n.velocity for n in result]
        for v in remaining_velocities:
            assert v >= 25, f"All remaining notes should have velocity >= 25, got {v}"

    def test_adaptive_velocity_can_be_disabled(self):
        """Adaptive velocity can be disabled."""
        notes = [
            Note(pitch=60, onset=0, offset=1, velocity=15),
            Note(pitch=62, onset=0, offset=1, velocity=80),
        ]

        config = CleanupConfig(adaptive_velocity=False, min_velocity=10)
        cleanup = NoteCleanup(config=config)

        result, stats = cleanup.cleanup(notes, return_stats=True)

        assert stats.removed_adaptive_velocity == 0, "Should not use adaptive velocity"


class TestNoteDensityFilter:
    """Test C.4: Note density filter."""

    def test_density_filter_limits_notes_per_second(self):
        """Should limit the number of notes per second."""
        # Create many notes in one second
        notes = [
            Note(pitch=60 + i, onset=0.0, offset=0.5, velocity=50 + i)
            for i in range(30)
        ]

        config = CleanupConfig(max_notes_per_second=10)
        cleanup = NoteCleanup(config=config)

        result = cleanup.filter_by_density(notes)

        # Should keep at most 10 notes per second
        assert len(result) <= 10, f"Should limit to 10 notes, got {len(result)}"

    def test_density_filter_keeps_loudest_notes(self):
        """Should keep the loudest notes when filtering by density."""
        # Create notes with different velocities
        notes = [
            Note(pitch=60 + i, onset=0.0, offset=0.5, velocity=100 - i * 10)
            for i in range(10)
        ]
        # Velocities: 100, 90, 80, 70, 60, 50, 40, 30, 20, 10

        config = CleanupConfig(max_notes_per_second=5)
        cleanup = NoteCleanup(config=config)

        result = cleanup.filter_by_density(notes)

        # Should keep the 5 loudest notes
        assert len(result) <= 5, f"Should limit to 5 notes, got {len(result)}"
        velocities = [n.velocity for n in result]
        assert min(velocities) >= 60, f"Should keep loudest notes, min velocity: {min(velocities)}"

    def test_density_filter_can_be_disabled(self):
        """Density filter can be disabled with max_notes_per_second=0."""
        notes = [
            Note(pitch=60 + i, onset=0.0, offset=0.5, velocity=50)
            for i in range(30)
        ]

        config = CleanupConfig(max_notes_per_second=0)  # Disabled
        cleanup = NoteCleanup(config=config)

        result = cleanup.filter_by_density(notes)

        assert len(result) == len(notes), "Should not filter any notes"


class TestCleanupStats:
    """Test cleanup statistics tracking."""

    def test_stats_track_all_removals(self):
        """Stats should track all types of note removals."""
        # Create notes that will trigger various filters
        notes = [
            Note(pitch=60, onset=0.0, offset=1.0, velocity=100),  # Keep
            Note(pitch=61, onset=0.0, offset=1.0, velocity=5),   # Ghost note
            Note(pitch=72, onset=0.0, offset=1.0, velocity=50),  # Harmonic of 60
        ]

        config = CleanupConfig(
            min_velocity=20,
            remove_harmonics=True,
            adaptive_velocity=False,
        )
        cleanup = NoteCleanup(config=config)

        result, stats = cleanup.cleanup(notes, return_stats=True)

        assert stats.original_count == 3
        assert stats.removed_ghost_notes >= 1, "Should count removed ghost notes"

    def test_stats_track_notes_per_second(self):
        """Stats should track detected notes per second."""
        notes = [
            Note(pitch=60, onset=i * 0.1, offset=i * 0.1 + 0.1, velocity=80)
            for i in range(20)
        ]
        # 20 notes in 2 seconds = 10 notes/sec

        cleanup = NoteCleanup()
        result, stats = cleanup.cleanup(notes, return_stats=True)

        assert stats.detected_notes_per_sec > 0, "Should calculate notes per second"
        assert 8 < stats.detected_notes_per_sec < 12, f"Expected ~10 nps, got {stats.detected_notes_per_sec}"


class TestCleanupConfig:
    """Test CleanupConfig defaults and customization."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = CleanupConfig()

        assert config.min_velocity == 20
        assert config.auto_aggressive == True
        assert config.noise_threshold_notes_per_sec == 15.0
        assert config.adaptive_velocity == True
        assert config.adaptive_velocity_percentile == 15.0
        assert config.max_notes_per_second == 20.0
        assert config.harmonic_ratios == (2, 3, 4, 5, 6, 7, 8)

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = CleanupConfig(
            min_velocity=30,
            auto_aggressive=False,
            noise_threshold_notes_per_sec=20.0,
            adaptive_velocity=False,
            max_notes_per_second=15.0,
            harmonic_ratios=(2, 3, 4),
        )

        assert config.min_velocity == 30
        assert config.auto_aggressive == False
        assert config.noise_threshold_notes_per_sec == 20.0
        assert config.adaptive_velocity == False
        assert config.max_notes_per_second == 15.0
        assert config.harmonic_ratios == (2, 3, 4)
