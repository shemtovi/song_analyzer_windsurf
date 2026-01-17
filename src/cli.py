"""Command-line interface for Song Analyzer."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="song-analyzer",
    help="Audio to Sheet Music Transcription System",
)
console = Console()


@app.command()
def transcribe(
    input_file: Path = typer.Argument(..., help="Input audio file (WAV, MP3, etc.)"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output MIDI file path"
    ),
    tempo: float = typer.Option(
        0.0, "-t", "--tempo", help="Override tempo (BPM). 0 = auto-detect"
    ),
    quantize: bool = typer.Option(
        True, "-q", "--quantize/--no-quantize", help="Quantize notes to grid"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Verbose output"
    ),
    polyphonic: bool = typer.Option(
        False, "-p", "--polyphonic", help="Use polyphonic transcription (for chords/multiple notes)"
    ),
):
    """Transcribe audio file to MIDI."""
    from .input import AudioLoader
    from .transcription import MonophonicTranscriber, PolyphonicTranscriber
    from .analysis import TempoAnalyzer
    from .inference import KeyDetector
    from .processing import Quantizer, NoteCleanup
    from .output import MIDIExporter

    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Default output path
    if output is None:
        output = input_file.with_suffix(".mid")

    console.print(f"[blue]Loading audio:[/blue] {input_file}")

    # Load audio
    loader = AudioLoader(target_sr=22050)
    audio, sr = loader.load(str(input_file))
    duration = loader.get_duration(audio, sr)

    if verbose:
        console.print(f"  Duration: {duration:.2f}s, Sample rate: {sr}Hz")

    # Detect tempo if not provided
    if tempo <= 0:
        console.print("[blue]Detecting tempo...[/blue]")
        tempo_analyzer = TempoAnalyzer()
        detected_tempo, beats = tempo_analyzer.detect(audio, sr)

        # Some signals (e.g., a single sustained sine tone) may not
        # produce reliable beat tracking and can return tempo == 0.
        # In that case, fall back to a sensible default (120 BPM)
        # so that quantization still works.
        if detected_tempo is None or (isinstance(detected_tempo, (int, float)) and detected_tempo <= 0):
            tempo = 120.0
            console.print(
                "  [yellow]Tempo detection failed; using default 120.0 BPM[/yellow]"
            )
        else:
            tempo = float(detected_tempo)
            console.print(f"  Detected tempo: {tempo:.1f} BPM")

    # Transcribe
    if polyphonic:
        console.print("[blue]Transcribing (polyphonic mode)...[/blue]")
        transcriber = PolyphonicTranscriber(device="cpu")
        if transcriber.is_neural:
            console.print("  Using neural network (Onsets and Frames)")
        else:
            console.print("  Using CQT-based multi-pitch detection")
    else:
        console.print("[blue]Transcribing (monophonic mode)...[/blue]")
        transcriber = MonophonicTranscriber()

    notes = transcriber.transcribe(audio, sr)
    console.print(f"  Detected {len(notes)} notes")

    # Post-process
    if quantize:
        console.print("[blue]Quantizing...[/blue]")
        quantizer = Quantizer(tempo=tempo)
        cleaner = NoteCleanup()
        notes = quantizer.quantize(notes)
        notes = cleaner.cleanup(notes)
        console.print(f"  After cleanup: {len(notes)} notes")

    # Detect key
    if verbose:
        key_detector = KeyDetector()
        key, mode, confidence = key_detector.detect_from_notes(notes)
        console.print(f"  Detected key: {key} {mode} (confidence: {confidence:.2f})")

    # Export
    console.print(f"[blue]Exporting to:[/blue] {output}")
    exporter = MIDIExporter(tempo=tempo)
    exporter.export(notes, str(output))

    console.print(f"[green]✓ Transcription complete![/green]")

    # Show note summary
    if verbose and notes:
        _show_notes_table(notes)


@app.command()
def info(
    input_file: Path = typer.Argument(..., help="Input audio file"),
):
    """Show information about an audio file."""
    from .input import AudioLoader
    from .analysis import TempoAnalyzer
    from .inference import KeyDetector

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    loader = AudioLoader()
    audio, sr = loader.load(str(input_file))

    console.print(f"\n[bold]Audio Info:[/bold] {input_file.name}")
    console.print(f"  Duration: {loader.get_duration(audio, sr):.2f} seconds")
    console.print(f"  Sample rate: {sr} Hz")
    console.print(f"  Samples: {len(audio):,}")

    # Detect tempo
    tempo_analyzer = TempoAnalyzer()
    tempo, _ = tempo_analyzer.detect(audio, sr)
    console.print(f"  Estimated tempo: {tempo:.1f} BPM")

    # Detect key
    key_detector = KeyDetector()
    key, mode, conf = key_detector.detect_from_audio(audio, sr)
    console.print(f"  Estimated key: {key} {mode} (confidence: {conf:.2f})")


def _show_notes_table(notes):
    """Display notes in a table."""
    table = Table(title="Detected Notes")
    table.add_column("Pitch", style="cyan")
    table.add_column("Onset (s)", style="green")
    table.add_column("Duration (s)", style="yellow")
    table.add_column("Velocity", style="magenta")

    for note in notes:
        table.add_row(
            note.pitch_name,
            f"{note.onset:.3f}",
            f"{note.duration:.3f}",
            str(note.velocity),
        )

    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
