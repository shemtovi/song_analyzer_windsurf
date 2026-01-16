"""Export transcribed notes to various formats."""

import pretty_midi
from typing import List, Optional
from pathlib import Path
from .transcriber.base import Note


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


class MusicXMLExporter:
    """Export notes to MusicXML format via music21."""

    def __init__(self, tempo: float = 120.0, time_signature: str = "4/4"):
        """
        Initialize MusicXMLExporter.

        Args:
            tempo: Tempo in BPM
            time_signature: Time signature (e.g., "4/4", "3/4")
        """
        self.tempo = tempo
        self.time_signature = time_signature

    def export(self, notes: List[Note], output_path: str) -> None:
        """
        Export notes to MusicXML file.

        Args:
            notes: List of Note objects
            output_path: Path to output MusicXML file
        """
        try:
            from music21 import stream, note as m21_note, tempo as m21_tempo
            from music21 import meter, metadata
        except ImportError:
            raise ImportError("music21 is required for MusicXML export")

        # Create score
        score = stream.Score()
        score.metadata = metadata.Metadata()
        score.metadata.title = "Transcribed Music"

        # Create part
        part = stream.Part()

        # Add tempo
        mm = m21_tempo.MetronomeMark(number=self.tempo)
        part.append(mm)

        # Add time signature
        ts = meter.TimeSignature(self.time_signature)
        part.append(ts)

        # Convert notes
        for n in notes:
            m21_n = m21_note.Note()
            m21_n.pitch.midi = n.pitch
            m21_n.duration.quarterLength = self._seconds_to_quarters(n.duration)
            m21_n.volume.velocity = n.velocity
            part.append(m21_n)

        score.append(part)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        score.write("musicxml", fp=output_path)

    def _seconds_to_quarters(self, seconds: float) -> float:
        """Convert duration in seconds to quarter notes."""
        beats_per_second = self.tempo / 60.0
        return seconds * beats_per_second
