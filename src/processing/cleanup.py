"""Note cleanup - Filter and merge notes."""

from typing import List

from ..core import Note


class NoteCleanup:
    """Clean up and refine transcribed notes."""

    def __init__(
        self,
        min_velocity: int = 20,
        min_duration: float = 0.05,
        max_gap_to_fill: float = 0.05,
    ):
        """
        Initialize NoteCleanup.

        Args:
            min_velocity: Minimum velocity to keep (filter ghost notes)
            min_duration: Minimum note duration in seconds
            max_gap_to_fill: Maximum gap to fill between same-pitch notes
        """
        self.min_velocity = min_velocity
        self.min_duration = min_duration
        self.max_gap_to_fill = max_gap_to_fill

    def cleanup(self, notes: List[Note]) -> List[Note]:
        """Apply all cleanup operations."""
        notes = self.remove_ghost_notes(notes)
        notes = self.merge_short_notes(notes)
        notes = self.fill_gaps(notes)
        return notes

    def merge_short_notes(self, notes: List[Note]) -> List[Note]:
        """
        Merge very short notes into adjacent notes.

        Args:
            notes: List of notes

        Returns:
            List with short notes merged
        """
        if not notes:
            return []

        # Sort by onset time
        sorted_notes = sorted(notes, key=lambda n: n.onset)
        merged = []

        for note in sorted_notes:
            if note.duration < self.min_duration:
                # Try to extend previous note
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
        """Remove notes with very low velocity."""
        return [n for n in notes if n.velocity >= self.min_velocity]

    def fill_gaps(self, notes: List[Note]) -> List[Note]:
        """
        Fill small gaps between consecutive notes of same pitch.

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
        """
        Remove overlapping notes of the same pitch.

        Args:
            notes: List of notes

        Returns:
            List without overlaps
        """
        if not notes:
            return []

        # Group by pitch
        pitch_groups = {}
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
