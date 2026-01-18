"""Command-line interface for Song Analyzer.

Provides commands for:
- transcribe: Convert audio to MIDI
- analyze: Full harmony analysis (key, chords, harmony)
- separate: Multi-instrument transcription with source separation
- info: Show audio file information
"""

import typer
import tempfile
import shutil
import time
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

app = typer.Typer(
    name="song-analyzer",
    help="Audio to Sheet Music Transcription System",
    rich_markup_mode="markdown",
)
console = Console()


@dataclass
class StageTimings:
    """Track timing of processing stages."""

    stages: Dict[str, float] = field(default_factory=dict)
    _current_stage: Optional[str] = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._current_stage = stage
        self._start_time = time.time()

    def stop(self) -> float:
        """Stop timing the current stage, return duration."""
        if self._current_stage is None:
            return 0.0
        duration = time.time() - self._start_time
        self.stages[self._current_stage] = duration
        self._current_stage = None
        return duration

    @property
    def total_time(self) -> float:
        """Get total time across all stages."""
        return sum(self.stages.values())

    def print_summary(self) -> None:
        """Print timing summary to console."""
        console.print("\n[bold]Timing Summary:[/bold]")
        for stage, duration in self.stages.items():
            console.print(f"  {stage}: {duration:.2f}s")
        console.print(f"  [bold]Total: {self.total_time:.2f}s[/bold]")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "stages": self.stages,
            "total_time": self.total_time,
        }


def download_audio_from_url(url: str, output_dir: Optional[Path] = None) -> Path:
    """
    Download audio from a URL (YouTube, etc.) using yt-dlp.
    
    Args:
        url: URL to download from (YouTube, SoundCloud, etc.)
        output_dir: Directory to save the file (default: temp directory)
    
    Returns:
        Path to the downloaded audio file
    """
    try:
        import yt_dlp
    except ImportError:
        console.print("[red]yt-dlp not installed. Run: pip install yt-dlp[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="song_analyzer_"))
    
    output_template = str(output_dir / "%(title)s.%(ext)s")
    
    # Try with ffmpeg conversion first, fall back to direct download
    ydl_opts_with_ffmpeg = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
    }
    
    # Fallback: download best audio without conversion
    ydl_opts_no_ffmpeg = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
    }
    
    console.print(f"[cyan]Downloading audio from URL...[/cyan]")
    
    # Try with ffmpeg first
    try:
        with yt_dlp.YoutubeDL(ydl_opts_with_ffmpeg) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio')
            downloaded_file = output_dir / f"{title}.wav"
            
            if downloaded_file.exists():
                console.print(f"   Downloaded: {downloaded_file.name}")
                return downloaded_file
    except Exception as e:
        if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
            console.print("   [yellow]FFmpeg not found, downloading without conversion...[/yellow]")
        else:
            pass  # Try fallback
    
    # Fallback: download without conversion
    try:
        with yt_dlp.YoutubeDL(ydl_opts_no_ffmpeg) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio')
            
            # Find any audio file in the output dir
            audio_files = list(output_dir.glob("*.*"))
            audio_files = [f for f in audio_files if f.suffix.lower() in 
                          ['.wav', '.mp3', '.m4a', '.webm', '.ogg', '.opus', '.aac', '.flac']]
            
            if not audio_files:
                console.print("[red]Failed to find downloaded audio file[/red]")
                raise typer.Exit(1)
            
            downloaded_file = audio_files[0]
            console.print(f"   Downloaded: {downloaded_file.name}")
            
            # Convert to WAV if needed (librosa needs wav/mp3)
            if downloaded_file.suffix.lower() not in ['.wav', '.mp3']:
                console.print("   [yellow]Converting to WAV format...[/yellow]")
                wav_file = downloaded_file.with_suffix('.wav')
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(str(downloaded_file))
                    audio.export(str(wav_file), format='wav')
                    downloaded_file = wav_file
                    console.print(f"   Converted to: {wav_file.name}")
                except Exception as conv_e:
                    console.print(f"[red]Conversion failed: {conv_e}[/red]")
                    console.print("[yellow]Please install ffmpeg for audio conversion.[/yellow]")
                    raise typer.Exit(1)
            
            return downloaded_file
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


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
    aggressive_cleanup: bool = typer.Option(
        False, "--aggressive", "-a", help="Use aggressive cleanup (remove harmonics, outliers, duplicates)"
    ),
    min_velocity: int = typer.Option(
        20, "--min-velocity", help="Minimum note velocity (0-127)"
    ),
    min_duration: float = typer.Option(
        0.05, "--min-duration", help="Minimum note duration in seconds"
    ),
    max_notes_per_frame: int = typer.Option(
        0, "--max-notes", help="Maximum simultaneous notes (0 = unlimited)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON (for scripting)"
    ),
):
    """Transcribe audio file to MIDI.

    **Examples:**

        song-analyzer transcribe song.wav

        song-analyzer transcribe song.mp3 -o output.mid -p

        song-analyzer transcribe song.wav --aggressive --max-notes 8
    """
    from .input import AudioLoader
    from .transcription import MonophonicTranscriber, PolyphonicTranscriber
    from .analysis import TempoAnalyzer
    from .inference import KeyDetector
    from .processing import Quantizer, NoteCleanup, CleanupConfig
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
        notes = quantizer.quantize(notes)

    # Cleanup
    console.print("[blue]Cleaning up notes...[/blue]")
    cleanup_config = CleanupConfig(
        min_velocity=min_velocity,
        min_duration=min_duration,
        enable_all=aggressive_cleanup,
    )
    cleaner = NoteCleanup(config=cleanup_config)

    if aggressive_cleanup:
        notes, stats = cleaner.cleanup_aggressive(notes, return_stats=True)
        console.print(f"  Removed: {stats.removed_ghost_notes} ghost, {stats.removed_harmonics} harmonics, {stats.removed_duplicates} duplicates, {stats.removed_outliers} outliers")
    else:
        notes = cleaner.cleanup(notes)

    # Limit notes per frame if requested
    if max_notes_per_frame > 0:
        before_count = len(notes)
        notes = cleaner.limit_notes_per_frame(notes, max_notes=max_notes_per_frame)
        console.print(f"  Limited to {max_notes_per_frame} notes/frame: {before_count} -> {len(notes)}")

    console.print(f"  After cleanup: {len(notes)} notes")

    # Detect key
    key_info_dict = None
    if verbose or json_output:
        key_detector = KeyDetector()
        key, mode, confidence = key_detector.detect_from_notes(notes)
        key_info_dict = {"key": key, "mode": mode, "confidence": confidence}
        if not json_output:
            console.print(f"  Detected key: {key} {mode} (confidence: {confidence:.2f})")

    # Export
    if not json_output:
        console.print(f"[blue]Exporting to:[/blue] {output}")
    exporter = MIDIExporter(tempo=tempo)
    exporter.export(notes, str(output))

    if not json_output:
        console.print(f"[green]Transcription complete![/green]")

    # JSON output mode
    if json_output:
        result = {
            "input": str(input_file),
            "output": str(output),
            "notes_count": len(notes),
            "tempo": tempo,
            "duration": duration,
            "polyphonic": polyphonic,
        }
        if key_info_dict:
            result["key"] = key_info_dict
        if aggressive_cleanup:
            result["cleanup_stats"] = {
                "ghost_notes_removed": stats.removed_ghost_notes,
                "harmonics_removed": stats.removed_harmonics,
                "duplicates_removed": stats.removed_duplicates,
                "outliers_removed": stats.removed_outliers,
            }
        console.print_json(data=result)
    elif verbose and notes:
        # Show note summary
        _show_notes_table(notes)


@app.command()
def analyze(
    input_file: Optional[Path] = typer.Argument(None, help="Input audio file"),
    url: Optional[str] = typer.Option(
        None, "--url", "-u", help="URL to download audio from (YouTube, etc.)"
    ),
    polyphonic: bool = typer.Option(
        True, "-p", "--polyphonic/--monophonic", help="Use polyphonic transcription"
    ),
    keep_download: bool = typer.Option(
        False, "--keep", "-k", help="Keep downloaded audio file after analysis"
    ),
):
    """Full harmony analysis: notes, key, chords, and harmony.
    
    Supports both local files and URLs (YouTube, SoundCloud, etc.)
    
    Examples:
        song-analyzer analyze audio.wav
        song-analyzer analyze --url "https://youtube.com/watch?v=..."
    """
    from .input import AudioLoader
    from .transcription import MonophonicTranscriber, PolyphonicTranscriber
    from .inference import (
        KeyDetector,
        ChordAnalyzer,
        HarmonyAnalyzer,
        ResilienceProcessor,
    )

    # Determine input source
    temp_dir = None
    if url:
        # Download from URL
        temp_dir = Path(tempfile.mkdtemp(prefix="song_analyzer_"))
        input_file = download_audio_from_url(url, temp_dir)
    elif input_file is None:
        console.print("[red]Error: Provide either an input file or --url[/red]")
        raise typer.Exit(1)
    elif not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"\n[bold blue]Full Analysis: {input_file.name}[/bold blue]\n")

        # Load audio
        console.print("[cyan]1. Loading audio...[/cyan]")
        loader = AudioLoader(target_sr=22050)
        audio, sr = loader.load(str(input_file))
        duration = loader.get_duration(audio, sr)
        console.print(f"   Duration: {duration:.2f}s")

        # Transcribe
        console.print("\n[cyan]2. Transcribing notes...[/cyan]")
        if polyphonic:
            transcriber = PolyphonicTranscriber(device="cpu")
        else:
            transcriber = MonophonicTranscriber()
        
        notes = transcriber.transcribe(audio, sr)
        console.print(f"   Detected {len(notes)} raw notes")

        # Clean notes
        processor = ResilienceProcessor(min_velocity=10, min_duration=0.02)
        notes, stats = processor.process(notes)
        console.print(f"   After cleanup: {len(notes)} notes")

        if not notes:
            console.print("[yellow]No notes detected![/yellow]")
            return

        # Show notes
        _show_notes_table(notes[:15])
        if len(notes) > 15:
            console.print(f"   [dim]... and {len(notes) - 15} more notes[/dim]")

        # Key detection
        console.print("\n[cyan]3. Key Detection...[/cyan]")
        key_detector = KeyDetector()
        key_info = key_detector.analyze(notes)
        console.print(f"   [green]Key: {key_info.root} {key_info.mode}[/green]")
        console.print(f"   Confidence: {key_info.confidence:.2f}")
        console.print(f"   Ambiguity: {key_info.ambiguity_score:.2f}")
        if key_info.relative_key:
            console.print(f"   Relative: {key_info.relative_key}")

        # Chord detection
        console.print("\n[cyan]4. Chord Detection...[/cyan]")
        chord_analyzer = ChordAnalyzer(min_chord_duration=0.2, use_key_context=True)
        chords = chord_analyzer.detect_chords(notes, key_info.root, key_info.mode)
        console.print(f"   Detected {len(chords)} chords")

        if chords:
            _show_chords_table(chords, key_info.root, key_info.mode)
            
            # Progression analysis
            progression = chord_analyzer.analyze_progression(chords, key_info.root, key_info.mode)
            if progression.roman_numerals:
                console.print(f"\n   [green]Progression: {' - '.join(progression.roman_numerals)}[/green]")

        # Harmony analysis
        console.print("\n[cyan]5. Harmony Analysis...[/cyan]")
        harmony_analyzer = HarmonyAnalyzer()
        harmony = harmony_analyzer.analyze(notes)
        
        console.print(f"   Harmonic rhythm: {harmony.harmonic_rhythm:.2f} changes/beat")
        console.print(f"   Average tension: {harmony.average_tension:.2f}")
        console.print(f"   Musical coherence: {harmony.musical_coherence:.2f}")

        console.print("\n[green]✓ Analysis complete![/green]")
        
        if keep_download and temp_dir:
            console.print(f"   [dim]Downloaded file saved at: {input_file}[/dim]")
    
    finally:
        # Cleanup temp directory if not keeping
        if temp_dir and not keep_download:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.command()
def separate(
    input_file: Optional[Path] = typer.Argument(None, help="Input audio file"),
    url: Optional[str] = typer.Option(
        None, "--url", "-u", help="URL to download audio from (YouTube, etc.)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory for stems and MIDI"
    ),
    stems: Optional[str] = typer.Option(
        None, "--stems", "-s",
        help="Comma-separated stems to analyze (e.g., 'bass,guitar,vocals'). Default: all"
    ),
    model: str = typer.Option(
        "htdemucs", "--model", "-m",
        help="Demucs model: htdemucs (4-stem), htdemucs_6s (6-stem with guitar/piano)"
    ),
    transcribe_stems: bool = typer.Option(
        True, "--transcribe/--no-transcribe", help="Transcribe each stem to MIDI"
    ),
    save_stems: bool = typer.Option(
        True, "--save-stems/--no-save-stems", help="Save separated audio stems"
    ),
    skip_drums: bool = typer.Option(
        False, "--skip-drums", help="Skip drum transcription"
    ),
    keep_download: bool = typer.Option(
        False, "--keep", "-k", help="Keep downloaded audio file"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable caching of separated stems"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Verbose output"
    ),
):
    """Multi-instrument transcription using source separation.

    Separates audio into stems using Demucs:
    - htdemucs: 4 stems (drums, bass, vocals, other)
    - htdemucs_6s: 6 stems (drums, bass, vocals, guitar, piano, other)
    
    Examples:
        song-analyzer separate audio.wav
        song-analyzer separate --stems bass,guitar audio.wav  # Only bass and guitar
        song-analyzer separate --model htdemucs_6s --stems guitar audio.wav
        song-analyzer separate --url "https://youtube.com/watch?v=..." -o output/
    """
    from .input import AudioLoader
    from .transcription import MultiInstrumentTranscriber
    from .separation import StemType
    from .output import MIDIExporter
    from .analysis import TempoAnalyzer

    # Determine input source
    temp_dir = None
    if url:
        temp_dir = Path(tempfile.mkdtemp(prefix="song_analyzer_"))
        input_file = download_audio_from_url(url, temp_dir)
    elif input_file is None:
        console.print("[red]Error: Provide either an input file or --url[/red]")
        raise typer.Exit(1)
    elif not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Set up output directory
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_separated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse stems option
    selected_stems = None
    if stems:
        stem_names = [s.strip().lower() for s in stems.split(",")]
        selected_stems = []
        valid_stems = ["drums", "bass", "vocals", "guitar", "piano", "other"]
        for name in stem_names:
            if name not in valid_stems:
                console.print(f"[yellow]Warning: Unknown stem '{name}'. Valid: {', '.join(valid_stems)}[/yellow]")
            else:
                try:
                    selected_stems.append(StemType(name))
                except ValueError:
                    pass
        
        # Auto-select 6-stem model if guitar/piano requested
        if any(s in stem_names for s in ["guitar", "piano"]) and model == "htdemucs":
            model = "htdemucs_6s"
            console.print("[cyan]Auto-selecting htdemucs_6s model for guitar/piano separation[/cyan]")

    try:
        console.print(f"\n[bold blue]Multi-Instrument Transcription: {input_file.name}[/bold blue]\n")
        
        if selected_stems:
            console.print(f"[cyan]Analyzing stems: {', '.join(s.value for s in selected_stems)}[/cyan]\n")

        # Load audio (stereo for Demucs separation)
        console.print("[cyan]1. Loading audio...[/cyan]")
        loader = AudioLoader(target_sr=44100, mono=False)  # Stereo for Demucs
        audio, sr = loader.load(str(input_file))
        duration = loader.get_duration(audio, sr)
        console.print(f"   Duration: {duration:.2f}s")

        # Initialize transcriber
        console.print("\n[cyan]2. Initializing source separator...[/cyan]")
        transcriber = MultiInstrumentTranscriber(
            demucs_model=model,
            device="auto",
            skip_drums=skip_drums,
            enable_cache=not no_cache,
        )

        if verbose and not no_cache:
            cache_stats = transcriber.separator.get_cache_stats()
            if cache_stats["cache_entries"] > 0:
                console.print(f"   Cache: {cache_stats['cache_entries']} entries, {cache_stats['total_size_mb']:.1f}MB")
        
        if not transcriber.is_available:
            console.print("   [yellow]Demucs not available, using fallback separation[/yellow]")
            console.print("   [dim]Install demucs for better results: pip install demucs torch torchaudio[/dim]")
        else:
            model_desc = "6-stem (guitar/piano)" if model == "htdemucs_6s" else "4-stem"
            console.print(f"   Using Demucs ({model} - {model_desc})")

        # Perform separation and transcription
        console.print("\n[cyan]3. Separating into stems...[/cyan]")
        result = transcriber.transcribe_multi(audio, sr, stems_to_process=selected_stems)
        
        console.print(f"   Separation time: {result.separation_time:.1f}s")
        console.print(f"   Active stems: {len(result.active_stems)}")

        # Display results
        console.print("\n[cyan]4. Transcription Results:[/cyan]")
        _show_stems_table(result)

        # Save stems if requested
        if save_stems:
            console.print("\n[cyan]5. Saving stems...[/cyan]")
            separated = transcriber.separator.separate(audio, sr)
            
            import soundfile as sf
            for stem_type, stem_audio in separated.stems.items():
                stem_path = output_dir / f"{input_file.stem}_{stem_type.value}.wav"
                sf.write(str(stem_path), stem_audio.audio, stem_audio.sample_rate)
                console.print(f"   Saved: {stem_path.name}")

        # Export MIDI for each stem
        if transcribe_stems:
            console.print("\n[cyan]6. Exporting MIDI...[/cyan]")
            
            # Detect tempo for quantization (needs mono audio)
            tempo_analyzer = TempoAnalyzer()
            audio_mono = audio.mean(axis=0) if len(audio.shape) > 1 else audio
            detected_tempo, _ = tempo_analyzer.detect(audio_mono, sr)
            tempo = float(detected_tempo) if detected_tempo and detected_tempo > 0 else 120.0
            
            exporter = MIDIExporter(tempo=tempo)
            
            for stem_type, stem_trans in result.stems.items():
                if stem_trans.note_count == 0:
                    continue
                    
                midi_path = output_dir / f"{input_file.stem}_{stem_type.value}.mid"
                exporter.export(stem_trans.notes, str(midi_path))
                console.print(f"   Saved: {midi_path.name} ({stem_trans.note_count} notes)")
            
            # Also export combined MIDI
            all_notes = result.get_all_notes()
            if all_notes:
                combined_path = output_dir / f"{input_file.stem}_combined.mid"
                exporter.export(all_notes, str(combined_path))
                console.print(f"   Saved: {combined_path.name} ({len(all_notes)} total notes)")

        # Summary
        console.print(f"\n[green]✓ Multi-instrument transcription complete![/green]")
        console.print(f"   Output directory: {output_dir}")
        console.print(f"   Total notes: {result.total_notes}")
        console.print(f"   Processing time: {result.total_time:.1f}s")
        
        if keep_download and temp_dir:
            console.print(f"   [dim]Downloaded file: {input_file}[/dim]")

    finally:
        if temp_dir and not keep_download:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _show_stems_table(result):
    """Display stem transcription results in a table."""
    table = Table(title="Stem Transcription Results")
    table.add_column("Stem", style="cyan")
    table.add_column("Instrument", style="green")
    table.add_column("Notes", style="yellow")
    table.add_column("Mode", style="blue")
    table.add_column("Confidence", style="magenta")

    for stem_type, stem_trans in result.stems.items():
        table.add_row(
            stem_type.value.title(),
            stem_trans.instrument_name,
            str(stem_trans.note_count),
            stem_trans.instrument_info.transcription_mode,
            f"{stem_trans.confidence:.2f}",
        )

    console.print(table)


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


def _show_chords_table(chords, key_root, key_mode):
    """Display chords in a table."""
    table = Table(title="Detected Chords")
    table.add_column("Chord", style="cyan")
    table.add_column("Roman", style="green")
    table.add_column("Time", style="yellow")
    table.add_column("Confidence", style="magenta")

    for chord in chords:
        roman = chord.get_roman_numeral(key_root, key_mode)
        table.add_row(
            chord.symbol,
            roman,
            f"{chord.onset:.2f}-{chord.offset:.2f}s",
            f"{chord.confidence:.2f}",
        )

    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
