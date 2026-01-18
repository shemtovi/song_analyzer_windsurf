"""Note cleanup - Filter, deduplicate, and refine transcribed notes.

This module provides comprehensive note cleanup capabilities:
- Ghost note removal (low velocity)
- Short note merging
- Gap filling
- Overlap resolution
- Harmonic filtering (remove overtones)
- Frame deduplication (remove duplicate detections)
- Outlier filtering (statistical pitch filtering)
- Temporal smoothing (remove isolated blips)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import numpy as np

from ..core import Note


@dataclass
class CleanupConfig:
    """Configuration for note cleanup operations.

    Attributes:
        min_velocity: Minimum velocity to keep (default: 20)
        min_duration: Minimum note duration in seconds (default: 0.05)
        max_gap_to_fill: Maximum gap to fill between same-pitch notes (default: 0.05)
        remove_harmonics: Whether to filter harmonic overtones (default: True)
        harmonic_tolerance_cents: Tolerance for harmonic detection in cents (default: 50)
        deduplicate_time_tolerance: Time window for deduplication in seconds (default: 0.03)
        outlier_pitch_std_factor: Standard deviations for outlier detection (default: 2.5)
        smooth_min_gap: Minimum gap for temporal smoothing (default: 0.1)
        enable_all: Enable all cleanup operations (default: False)
    """

    min_velocity: int = 20
    min_duration: float = 0.05
    max_gap_to_fill: float = 0.05
    remove_harmonics: bool = True
    harmonic_tolerance_cents: float = 50.0
    deduplicate_time_tolerance: float = 0.03
    outlier_pitch_std_factor: float = 2.5
    smooth_min_gap: float = 0.1
    enable_all: bool = False


@dataclass
class CleanupStats:
    """Statistics from cleanup operations."""

    original_count: int = 0
    final_count: int = 0
    removed_ghost_notes: int = 0
    removed_short_notes: int = 0
    removed_harmonics: int = 0
    removed_duplicates: int = 0
    removed_outliers: int = 0
    merged_notes: int = 0
    filled_gaps: int = 0

    @property
    def total_removed(self) -> int:
        """Total notes removed."""
        return self.original_count - self.final_count


class NoteCleanup:
    """Clean up and refine transcribed notes.

    Provides multiple filtering and refinement strategies to reduce
    spurious note detections while preserving true musical notes.
    """

    def __init__(
        self,
        min_velocity: int = 20,
        min_duration: float = 0.05,
        max_gap_to_fill: float = 0.05,
        config: Optional[CleanupConfig] = None,
    ):
        """Initialize NoteCleanup.

        Args:
            min_velocity: Minimum velocity to keep (filter ghost notes)
            min_duration: Minimum note duration in seconds
            max_gap_to_fill: Maximum gap to fill between same-pitch notes
            config: Optional CleanupConfig for advanced settings
        """
        if config is not None:
            self.config = config
        else:
            self.config = CleanupConfig(
                min_velocity=min_velocity,
                min_duration=min_duration,
                max_gap_to_fill=max_gap_to_fill,
            )

        # Keep backward-compatible properties
        self.min_velocity = self.config.min_velocity
        self.min_duration = self.config.min_duration
        self.max_gap_to_fill = self.config.max_gap_to_fill

    def cleanup(
        self,
        notes: List[Note],
        return_stats: bool = False,
    ) -> List[Note] | Tuple[List[Note], CleanupStats]:
        """Apply all cleanup operations.

        Args:
            notes: List of notes to clean
            return_stats: Whether to return cleanup statistics

        Returns:
            Cleaned notes, optionally with statistics
        """
        stats = CleanupStats(original_count=len(notes))

        # Core cleanup
        notes = self.remove_ghost_notes(notes)
        stats.removed_ghost_notes = stats.original_count - len(notes)

        notes = self.merge_short_notes(notes)
        notes = self.fill_gaps(notes)

        # Advanced cleanup if enabled
        if self.config.enable_all or self.config.remove_harmonics:
            count_before = len(notes)
            notes = self.remove_harmonics(notes)
            stats.removed_harmonics = count_before - len(notes)

        if self.config.enable_all:
            count_before = len(notes)
            notes = self.deduplicate_frames(notes)
            stats.removed_duplicates = count_before - len(notes)

            count_before = len(notes)
            notes = self.filter_outliers(notes)
            stats.removed_outliers = count_before - len(notes)

        stats.final_count = len(notes)

        if return_stats:
            return notes, stats
        return notes

    def cleanup_aggressive(
        self,
        notes: List[Note],
        return_stats: bool = False,
    ) -> List[Note] | Tuple[List[Note], CleanupStats]:
        """Apply aggressive cleanup with all filters enabled.

        This is useful for noisy transcriptions with many spurious notes.
        """
        # Temporarily enable all filters
        old_enable_all = self.config.enable_all
        self.config.enable_all = True

        result = self.cleanup(notes, return_stats=return_stats)

        self.config.enable_all = old_enable_all
        return result

    def merge_short_notes(self, notes: List[Note]) -> List[Note]:
        """Merge very short notes into adjacent notes of the same pitch.

        Args:
            notes: List of notes

        Returns:
            List with short notes merged
        """
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.onset)
        merged = []

        for note in sorted_notes:
            if note.duration < self.min_duration:
                # Try to extend previous note of same pitch
                if merged and merged[-1].pitch == note.pitch:
                    merged[-1] = Note(
                        pitch=merged[-1].pitch,
                        onset=merged[-1].onset,
                        offset=note.offset,
                        velocity=merged[-1].velocity,
                        instrument=merged[-1].instrument,
                    )
                    continue

            merged.append(note)

        return merged

    def remove_ghost_notes(self, notes: List[Note]) -> List[Note]:
        """Remove notes with very low velocity (ghost notes)."""
        return [n for n in notes if n.velocity >= self.min_velocity]

    def fill_gaps(self, notes: List[Note]) -> List[Note]:
        """Fill small gaps between consecutive notes of same pitch.

        Args:
            notes: List of notes

        Returns:
            List with gaps filled
        """
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.onset)
        filled = [sorted_notes[0]]

        for note in sorted_notes[1:]:
            prev = filled[-1]

            # Check if same pitch and small gap
            gap = note.onset - prev.offset
            if prev.pitch == note.pitch and 0 < gap <= self.max_gap_to_fill:
                # Extend previous note
                filled[-1] = Note(
                    pitch=prev.pitch,
                    onset=prev.onset,
                    offset=note.offset,
                    velocity=max(prev.velocity, note.velocity),
                    instrument=prev.instrument,
                )
            else:
                filled.append(note)

        return filled

    def remove_overlaps(self, notes: List[Note]) -> List[Note]:
        """Remove overlapping notes of the same pitch.

        Args:
            notes: List of notes

        Returns:
            List without overlaps
        """
        if not notes:
            return []

        # Group by pitch
        pitch_groups: dict = {}
        for note in notes:
            if note.pitch not in pitch_groups:
                pitch_groups[note.pitch] = []
            pitch_groups[note.pitch].append(note)

        result = []
        for pitch, group in pitch_groups.items():
            group.sort(key=lambda n: n.onset)

            for i, note in enumerate(group):
                # Truncate if overlaps with next note
                if i < len(group) - 1:
                    next_note = group[i + 1]
                    if note.offset > next_note.onset:
                        note = Note(
                            pitch=note.pitch,
                            onset=note.onset,
                            offset=next_note.onset,
                            velocity=note.velocity,
                            instrument=note.instrument,
                        )
                result.append(note)

        return sorted(result, key=lambda n: (n.onset, n.pitch))

    def remove_harmonics(
        self,
        notes: List[Note],
        tolerance_cents: Optional[float] = None,
    ) -> List[Note]:
        """Remove likely harmonic overtones.

        Harmonics are detected as notes that:
        1. Have frequency ratios of 2x, 3x, 4x, etc. to a louder note
        2. Occur at the same time as the fundamental
        3. Have lower velocity than the fundamental

        Args:
            notes: List of notes
            tolerance_cents: Pitch tolerance in cents (default: from config)

        Returns:
            List with harmonics removed
        """
        if not notes or len(notes) < 2:
            return notes

        tolerance_cents = tolerance_cents or self.config.harmonic_tolerance_cents
        tolerance_semitones = tolerance_cents / 100.0

        # Group notes by overlapping time
        keep_notes: Set[int] = set(range(len(notes)))
        notes_list = list(notes)

        for i, note in enumerate(notes_list):
            if i not in keep_notes:
                continue

            note_freq = 440.0 * (2 ** ((note.pitch - 69) / 12.0))

            for j, other in enumerate(notes_list):
                if i == j or j not in keep_notes:
                    continue

                # Check for time overlap
                if not (other.onset < note.offset and other.offset > note.onset):
                    continue

                other_freq = 440.0 * (2 ** ((other.pitch - 69) / 12.0))

                # Check if other is a harmonic of note (2x, 3x, 4x, 5x frequency)
                for harmonic in [2, 3, 4, 5]:
                    expected_freq = note_freq * harmonic
                    # Convert frequency difference to cents
                    if expected_freq > 0:
                        cents_diff = 1200 * np.log2(other_freq / expected_freq)
                        if abs(cents_diff) < tolerance_cents:
                            # other is likely a harmonic - remove if quieter
                            if other.velocity <= note.velocity:
                                keep_notes.discard(j)
                            break

        return [notes_list[i] for i in sorted(keep_notes)]

    def deduplicate_frames(
        self,
        notes: List[Note],
        time_tolerance: Optional[float] = None,
    ) -> List[Note]:
        """Remove duplicate notes within the same time window.

        When multiple notes of the same pitch occur within a short time
        window, keep only the one with highest velocity.

        Args:
            notes: List of notes
            time_tolerance: Time window for deduplication (default: from config)

        Returns:
            List with duplicates removed
        """
        if not notes:
            return []

        time_tolerance = time_tolerance or self.config.deduplicate_time_tolerance

        # Sort by pitch, then onset
        sorted_notes = sorted(notes, key=lambda n: (n.pitch, n.onset))
        result = []

        i = 0
        while i < len(sorted_notes):
            current = sorted_notes[i]
            best = current

            # Find all notes of same pitch within time tolerance
            j = i + 1
            while j < len(sorted_notes):
                next_note = sorted_notes[j]
                if next_note.pitch != current.pitch:
                    break
                if next_note.onset - current.onset > time_tolerance:
                    break

                # Keep note with highest velocity
                if next_note.velocity > best.velocity:
                    best = next_note
                j += 1

            result.append(best)
            i = j

        return sorted(result, key=lambda n: (n.onset, n.pitch))

    def filter_outliers(
        self,
        notes: List[Note],
        pitch_std_factor: Optional[float] = None,
        min_notes_for_stats: int = 10,
    ) -> List[Note]:
        """Remove notes with pitch far outside the expected range.

        Uses statistical analysis to identify outliers based on
        standard deviation from the mean pitch.

        Args:
            notes: List of notes
            pitch_std_factor: Number of std devs for outlier threshold
            min_notes_for_stats: Minimum notes needed for statistical filtering

        Returns:
            List with outliers removed
        """
        if not notes or len(notes) < min_notes_for_stats:
            return notes

        pitch_std_factor = pitch_std_factor or self.config.outlier_pitch_std_factor

        # Calculate pitch statistics
        pitches = np.array([n.pitch for n in notes])
        mean_pitch = np.mean(pitches)
        std_pitch = np.std(pitches)

        if std_pitch < 1:  # Very uniform, don't filter
            return notes

        # Filter outliers
        lower_bound = mean_pitch - pitch_std_factor * std_pitch
        upper_bound = mean_pitch + pitch_std_factor * std_pitch

        return [n for n in notes if lower_bound <= n.pitch <= upper_bound]

    def smooth_temporal(
        self,
        notes: List[Note],
        min_gap: Optional[float] = None,
        min_isolated_duration: float = 0.1,
    ) -> List[Note]:
        """Apply temporal smoothing to remove isolated short notes.

        Removes "blips" - short isolated notes that are likely noise.
        A note is considered isolated if it has gaps on both sides.

        Args:
            notes: List of notes
            min_gap: Minimum gap threshold (default: from config)
            min_isolated_duration: Minimum duration for isolated notes

        Returns:
            List with isolated short notes removed
        """
        if not notes or len(notes) < 3:
            return notes

        min_gap = min_gap or self.config.smooth_min_gap
        sorted_notes = sorted(notes, key=lambda n: n.onset)
        result = []

        for i, note in enumerate(sorted_notes):
            # Check if note is isolated
            gap_before = note.onset if i == 0 else note.onset - sorted_notes[i - 1].offset
            gap_after = float("inf") if i == len(sorted_notes) - 1 else sorted_notes[i + 1].onset - note.offset

            is_isolated = gap_before > min_gap and gap_after > min_gap
            is_short = note.duration < min_isolated_duration

            if is_isolated and is_short:
                continue  # Skip isolated short notes

            result.append(note)

        return result

    def filter_by_velocity_percentile(
        self,
        notes: List[Note],
        percentile: float = 10.0,
    ) -> List[Note]:
        """Filter notes below a velocity percentile.

        Args:
            notes: List of notes
            percentile: Percentile threshold (0-100)

        Returns:
            List with low-velocity notes removed
        """
        if not notes:
            return []

        velocities = [n.velocity for n in notes]
        threshold = np.percentile(velocities, percentile)

        return [n for n in notes if n.velocity >= threshold]

    def filter_by_duration_percentile(
        self,
        notes: List[Note],
        percentile: float = 5.0,
    ) -> List[Note]:
        """Filter notes below a duration percentile.

        Args:
            notes: List of notes
            percentile: Percentile threshold (0-100)

        Returns:
            List with very short notes removed
        """
        if not notes:
            return []

        durations = [n.duration for n in notes]
        threshold = np.percentile(durations, percentile)

        return [n for n in notes if n.duration >= threshold]

    def limit_notes_per_frame(
        self,
        notes: List[Note],
        max_notes: int = 8,
        frame_duration: float = 0.05,
    ) -> List[Note]:
        """Limit the number of simultaneous notes per time frame.

        Keeps only the N loudest notes in each time frame.

        Args:
            notes: List of notes
            max_notes: Maximum notes per frame
            frame_duration: Frame duration in seconds

        Returns:
            List with note count limited per frame
        """
        if not notes:
            return []

        # Find time range
        min_time = min(n.onset for n in notes)
        max_time = max(n.offset for n in notes)

        result_notes: Set[int] = set()
        notes_list = list(notes)

        # Process each frame
        current_time = min_time
        while current_time < max_time:
            frame_end = current_time + frame_duration

            # Find notes active in this frame
            frame_notes = [
                (i, n)
                for i, n in enumerate(notes_list)
                if n.onset < frame_end and n.offset > current_time
            ]

            # Keep top N by velocity
            frame_notes.sort(key=lambda x: x[1].velocity, reverse=True)
            for i, _ in frame_notes[:max_notes]:
                result_notes.add(i)

            current_time = frame_end

        return [notes_list[i] for i in sorted(result_notes)]
