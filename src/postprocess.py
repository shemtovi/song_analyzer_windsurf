"""Post-processing utilities for transcribed notes."""

import numpy as np
import librosa
from typing import List, Tuple, Optional
from .transcriber.base import Note


class PostProcessor:
    """Post-process transcribed notes (quantization, cleanup, etc.)."""

    def __init__(
        self,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4),
        quantize_resolution: int = 16,
    ):
        """
        Initialize PostProcessor.

        Args:
            tempo: Tempo in BPM
            time_signature: Time signature as (numerator, denominator)
            quantize_resolution: Quantization grid (e.g., 16 for 16th notes)
        """
        self.tempo = tempo
        self.time_signature = time_signature
        self.quantize_resolution = quantize_resolution

    @property
    def beat_duration(self) -> float:
        """Duration of one beat in seconds."""
        return 60.0 / self.tempo

    @property
    def grid_duration(self) -> float:
        """Duration of one grid unit in seconds."""
        return self.beat_duration * (4 / self.quantize_resolution)

    def quantize(self, notes: List[Note]) -> List[Note]:
        """
        Quantize note onsets and offsets to grid.

        Args:
            notes: List of notes to quantize

        Returns:
            List of quantized notes
        """
        quantized = []

        for note in notes:
            q_onset = self._snap_to_grid(note.onset)
            q_offset = self._snap_to_grid(note.offset)

            # Ensure minimum duration
            if q_offset <= q_onset:
                q_offset = q_onset + self.grid_duration

            quantized.append(
                Note(
                    pitch=note.pitch,
                    onset=q_onset,
                    offset=q_offset,
                    velocity=note.velocity,
                    instrument=note.instrument,
                )
            )

        return quantized

    def _snap_to_grid(self, time: float) -> float:
        """Snap time to nearest grid position."""
        grid_units = round(time / self.grid_duration)
        return grid_units * self.grid_duration

    def merge_short_notes(
        self,
        notes: List[Note],
        min_duration: float = 0.05,
    ) -> List[Note]:
        """
        Merge very short notes into adjacent notes.

        Args:
            notes: List of notes
            min_duration: Minimum note duration in seconds

        Returns:
            List with short notes merged
        """
        if not notes:
            return []

        # Sort by onset time
        sorted_notes = sorted(notes, key=lambda n: n.onset)
        merged = []

        for note in sorted_notes:
            if note.duration < min_duration:
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

    def remove_ghost_notes(
        self,
        notes: List[Note],
        min_velocity: int = 20,
    ) -> List[Note]:
        """Remove notes with very low velocity."""
        return [n for n in notes if n.velocity >= min_velocity]

    def fill_gaps(
        self,
        notes: List[Note],
        max_gap: float = 0.05,
    ) -> List[Note]:
        """
        Fill small gaps between consecutive notes of same pitch.

        Args:
            notes: List of notes
            max_gap: Maximum gap to fill in seconds

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
            if prev.pitch == note.pitch and 0 < gap <= max_gap:
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

    def cleanup(self, notes: List[Note]) -> List[Note]:
        """Apply all cleanup operations."""
        notes = self.remove_ghost_notes(notes)
        notes = self.merge_short_notes(notes)
        notes = self.fill_gaps(notes)
        return notes


class TempoDetector:
    """Detect tempo from audio."""

    def __init__(self, hop_length: int = 512):
        self.hop_length = hop_length

    def detect(self, audio: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """
        Detect tempo and beat positions.

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            Tuple of (tempo in BPM, beat times in seconds)
        """
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
        )

        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)

        # Handle tempo as array (newer librosa versions)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0

        return tempo, beat_times


class KeyDetector:
    """Detect musical key from audio or notes."""

    # Krumhansl-Schmuckler key profiles
    MAJOR_PROFILE = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    MINOR_PROFILE = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def detect_from_audio(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[str, str, float]:
        """
        Detect key from audio using chromagram.

        Returns:
            Tuple of (key name, mode, correlation score)
        """
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        return self._find_best_key(chroma_mean)

    def detect_from_notes(self, notes: List[Note]) -> Tuple[str, str, float]:
        """
        Detect key from note list.

        Returns:
            Tuple of (key name, mode, correlation score)
        """
        # Build pitch class histogram
        pitch_classes = np.zeros(12)
        for note in notes:
            pc = note.pitch % 12
            pitch_classes[pc] += note.duration

        # Normalize
        if pitch_classes.sum() > 0:
            pitch_classes /= pitch_classes.sum()

        return self._find_best_key(pitch_classes)

    def _find_best_key(
        self, pitch_classes: np.ndarray
    ) -> Tuple[str, str, float]:
        """Find best matching key using correlation."""
        best_corr = -1
        best_key = "C"
        best_mode = "major"

        for shift in range(12):
            # Rotate pitch classes
            rotated = np.roll(pitch_classes, -shift)

            # Correlate with major profile
            major_corr = np.corrcoef(rotated, self.MAJOR_PROFILE)[0, 1]
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = self.KEY_NAMES[shift]
                best_mode = "major"

            # Correlate with minor profile
            minor_corr = np.corrcoef(rotated, self.MINOR_PROFILE)[0, 1]
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = self.KEY_NAMES[shift]
                best_mode = "minor"

        return best_key, best_mode, best_corr
