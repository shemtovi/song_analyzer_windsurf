"""Note quantization - Snap notes to rhythmic grid."""

from typing import List, Tuple

from ..core import Note


class Quantizer:
    """Quantize note timings to a rhythmic grid."""

    def __init__(
        self,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4),
        quantize_resolution: int = 16,
    ):
        """
        Initialize Quantizer.

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

    def quantize_swing(
        self, 
        notes: List[Note], 
        swing_ratio: float = 0.67
    ) -> List[Note]:
        """
        Quantize with swing feel.

        Args:
            notes: List of notes
            swing_ratio: Swing ratio (0.5 = straight, 0.67 = triplet swing)

        Returns:
            List of quantized notes with swing
        
        TODO: Implement swing quantization
        """
        # For now, just regular quantize
        return self.quantize(notes)
