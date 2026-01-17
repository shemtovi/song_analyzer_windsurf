"""MIDI export functionality."""

import pretty_midi
from typing import List
from pathlib import Path

from ..core import Note


class MIDIExporter:
    """Export notes to MIDI format."""

    def __init__(
        self,
        tempo: float = 120.0,
        instrument_name: str = "Acoustic Grand Piano",
        instrument_program: int = 0,
    ):
        """
        Initialize MIDIExporter.

        Args:
            tempo: Tempo in BPM
            instrument_name: MIDI instrument name
            instrument_program: MIDI program number (0-127)
        """
        self.tempo = tempo
        self.instrument_name = instrument_name
        self.instrument_program = instrument_program

    def export(self, notes: List[Note], output_path: str) -> None:
        """
        Export notes to MIDI file.

        Args:
            notes: List of Note objects
            output_path: Path to output MIDI file
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        instrument = pretty_midi.Instrument(
            program=self.instrument_program,
            name=self.instrument_name,
        )

        for note in notes:
            midi_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.onset,
                end=note.offset,
            )
            instrument.notes.append(midi_note)

        midi.instruments.append(instrument)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        midi.write(output_path)

    def notes_to_pretty_midi(self, notes: List[Note]) -> pretty_midi.PrettyMIDI:
        """Convert notes to PrettyMIDI object without saving."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        instrument = pretty_midi.Instrument(
            program=self.instrument_program,
            name=self.instrument_name,
        )

        for note in notes:
            midi_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.onset,
                end=note.offset,
            )
            instrument.notes.append(midi_note)

        midi.instruments.append(instrument)
        return midi
