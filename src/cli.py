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

    # Get ffmpeg path from imageio-ffmpeg if available
    ffmpeg_path = None
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        # Check if ffmpeg is available in system PATH
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            ffmpeg_path = 'ffmpeg'
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[yellow]Warning: ffmpeg not found. Installing imageio-ffmpeg is recommended for better audio quality.[/yellow]")
            console.print("[yellow]Run: pip install imageio-ffmpeg[/yellow]\n")
    
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="song_analyzer_"))
    
    output_template = str(output_dir / "%(title)s.%(ext)s")

    # Add ffmpeg to PATH if we found it (helps librosa)
    if ffmpeg_path:
        import os
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

    # Download best audio without conversion (we'll convert ourselves if needed)
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
    }
    
    console.print(f"[cyan]Downloading audio from URL...[/cyan]")

    # Download without conversion (simpler, doesn't need ffprobe)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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

            # Convert to WAV using ffmpeg for best compatibility
            if downloaded_file.suffix.lower() != '.wav' and ffmpeg_path:
                console.print("   [yellow]Converting to WAV format...[/yellow]")
                wav_file = downloaded_file.with_suffix('.wav')
                try:
                    # Use ffmpeg directly via subprocess
                    import subprocess
                    result = subprocess.run(
                        [
                            ffmpeg_path,
                            '-i', str(downloaded_file),
                            '-acodec', 'pcm_s16le',
                            '-ar', '44100',
                            '-y',  # Overwrite output file
                            str(wav_file)
                        ],
                        capture_output=True,
                        check=True
                    )
                    downloaded_file = wav_file
                    console.print(f"   Converted to: {wav_file.name}")
                except Exception as conv_e:
                    console.print(f"[yellow]Warning: Conversion to WAV failed: {conv_e}[/yellow]")
                    console.print(f"[yellow]Will try to load {downloaded_file.suffix} directly[/yellow]")
                    # Return the original file and hope librosa can handle it

            return downloaded_file
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def transcribe(
    input_file: Path = typer.Argument(..., help="Input audio file, WAV, MP3, or URL"),
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
    sensitivity: str = typer.Option(
        "medium", "--sensitivity", "-s", help="Transcription sensitivity: low/medium/high/ultra"
    ),
    normalize: bool = typer.Option(
        False, "--normalize", help="Normalize audio before transcription"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON (for scripting)"
    ),
):
    """Transcribe audio file to MIDI. Supports local files and YouTube URLs.

    **Examples:**

        song-analyzer transcribe song.wav

        song-analyzer transcribe song.mp3 -o output.mid -p

        song-analyzer transcribe "https://youtube.com/watch?v=..." -o output.mid
    """
    from .input import AudioLoader
    from .transcription import MonophonicTranscriber, PolyphonicTranscriber
    from .analysis import TempoAnalyzer
    from .inference import KeyDetector
    from .processing import Quantizer, NoteCleanup, CleanupConfig
    from .output import MIDIExporter

    # Validate input and handle URLs
    temp_dir = None
    input_str = str(input_file)
    # Handle both forward slashes and Windows backslashes in URLs
    if input_str.startswith(("http://", "https://", "http:\\", "https:\\", "www.")):
        # Normalize Windows path separators back to URL format
        # Replace the protocol part first, then the rest
        if input_str.startswith(("http:\\", "https:\\")):
            input_str = input_str.replace(":\\", "://", 1).replace("\\", "/")
        else:
            input_str = input_str.replace("\\", "/")
        temp_dir = Path(tempfile.mkdtemp(prefix="song_analyzer_"))
        input_file = download_audio_from_url(input_str, temp_dir)
    elif not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Default output path
    if output is None:
        output = input_file.with_suffix(".mid")

    try:
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

            # Configure sensitivity
            sensitivity_configs = {
                "low": {
                    "min_peak_energy": 0.05,
                    "min_rms_threshold": 0.002,
                    "spectral_flatness_threshold": 0.90,
                },
                "medium": {
                    "min_peak_energy": 0.03,
                    "min_rms_threshold": 0.001,
                    "spectral_flatness_threshold": 0.95,
                },
                "high": {
                    "min_peak_energy": 0.02,
                    "min_rms_threshold": 0.0005,
                    "spectral_flatness_threshold": 0.97,
                },
                "ultra": {
                    "min_peak_energy": 0.01,
                    "min_rms_threshold": 0.0001,
                    "spectral_flatness_threshold": 0.99,
                },
            }

            sensitivity_lower = sensitivity.lower()
            if sensitivity_lower not in sensitivity_configs:
                console.print(f"[yellow]Unknown sensitivity '{sensitivity}', using 'medium'[/yellow]")
                sensitivity_lower = "medium"

            config = sensitivity_configs[sensitivity_lower]
            console.print(f"  Sensitivity: {sensitivity_lower}")

            transcriber = PolyphonicTranscriber(
                device="cpu",
                normalize_audio=normalize,
                use_adaptive_thresholds=True,
                **config,
            )

            if transcriber.is_neural:
                console.print("  Using neural network (Onsets and Frames)")
            else:
                console.print("  Using CQT-based multi-pitch detection")

            if normalize:
                console.print("  Audio normalization: enabled")
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

    finally:
        # Cleanup temp directory if URL was used
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


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
    else:
        # Check if input_file is actually a URL (auto-detect)
        input_str = str(input_file)
        # Handle both forward slashes and Windows backslashes in URLs
        if input_str.startswith(("http://", "https://", "http:\\", "https:\\", "www.")):
            # Normalize Windows path separators back to URL format
            # Replace the protocol part first, then the rest
            if input_str.startswith(("http:\\", "https:\\")):
                input_str = input_str.replace(":\\", "://", 1).replace("\\", "/")
            else:
                input_str = input_str.replace("\\", "/")
            temp_dir = Path(tempfile.mkdtemp(prefix="song_analyzer_"))
            input_file = download_audio_from_url(input_str, temp_dir)
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

        console.print("\n[green][OK] Analysis complete![/green]")
        
        if keep_download and temp_dir:
            console.print(f"   [dim]Downloaded file saved at: {input_file}[/dim]")
    
    finally:
        # Cleanup temp directory if not keeping
        if temp_dir and not keep_download:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.command()
def separate(
    input_file: Optional[Path] = typer.Argument(None, help="Input audio file or YouTube URL"),
    output_dir: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory for stem audio files"
    ),
    stems: Optional[str] = typer.Option(
        None, "--stems", "-s",
        help="Comma-separated stems to extract (e.g., 'bass,guitar,vocals'). Default: all"
    ),
    model: str = typer.Option(
        "htdemucs", "--model", "-m",
        help="Demucs model: htdemucs (4-stem), htdemucs_6s (6-stem with guitar/piano)"
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
    """Separate audio into individual instrument stems.

    Uses Demucs neural network to separate audio into stems:
    - htdemucs: 4 stems (drums, bass, vocals, other)
    - htdemucs_6s: 6 stems (drums, bass, vocals, guitar, piano, other)

    This command ONLY does separation - no transcription or MIDI export.
    Use 'transcribe' command on the output stems to generate MIDI files.

    Examples:
        song-analyzer separate audio.wav
        song-analyzer separate audio.wav -o output/
        song-analyzer separate --stems bass,vocals audio.wav
        song-analyzer separate --model htdemucs_6s audio.wav
        song-analyzer separate "https://youtube.com/watch?v=..." -o output/
    """
    from .input import AudioLoader
    from .separation import SourceSeparator, StemType

    # Determine input source
    temp_dir = None
    if input_file is None:
        console.print("[red]Error: Provide an input audio file or YouTube URL[/red]")
        raise typer.Exit(1)

    # Check if input_file is actually a URL (auto-detect)
    input_str = str(input_file)
    # Handle both forward slashes and Windows backslashes in URLs
    if input_str.startswith(("http://", "https://", "http:\\", "https:\\", "www.")):
        # Normalize Windows path separators back to URL format
        if input_str.startswith(("http:\\", "https:\\")):
            input_str = input_str.replace(":\\", "://", 1).replace("\\", "/")
        else:
            input_str = input_str.replace("\\", "/")
        temp_dir = Path(tempfile.mkdtemp(prefix="song_analyzer_"))
        input_file = download_audio_from_url(input_str, temp_dir)
    elif not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Set up output directory
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_stems"
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
        console.print(f"\n[bold blue]Audio Stem Separation: {input_file.name}[/bold blue]\n")

        if selected_stems:
            console.print(f"[cyan]Extracting stems: {', '.join(s.value for s in selected_stems)}[/cyan]\n")

        # Load audio (stereo for Demucs separation)
        console.print("[cyan]1. Loading audio...[/cyan]")
        loader = AudioLoader(target_sr=44100, mono=False)  # Stereo for Demucs
        audio, sr = loader.load(str(input_file))
        duration = loader.get_duration(audio, sr)
        console.print(f"   Duration: {duration:.2f}s")
        console.print(f"   Sample rate: {sr} Hz")
        console.print(f"   Channels: {'Stereo' if len(audio.shape) > 1 else 'Mono'}")

        # Initialize separator
        console.print("\n[cyan]2. Initializing source separator...[/cyan]")
        separator = SourceSeparator(
            model_name=model,
            device="auto",
            enable_cache=not no_cache,
        )

        if verbose and not no_cache:
            cache_stats = separator.get_cache_stats()
            if cache_stats["cache_entries"] > 0:
                console.print(f"   Cache: {cache_stats['cache_entries']} entries, {cache_stats['total_size_mb']:.1f}MB")

        if not separator.is_available:
            console.print("   [yellow]Demucs not available, using fallback separation[/yellow]")
            console.print("   [dim]Install demucs for better results: pip install demucs torch torchaudio[/dim]")
        else:
            model_desc = "6-stem (guitar/piano)" if model == "htdemucs_6s" else "4-stem"
            console.print(f"   Using Demucs ({model} - {model_desc})")

        # Perform separation
        console.print("\n[cyan]3. Separating into stems...[/cyan]")

        import time
        start_time = time.time()
        separated = separator.separate(audio, sr)
        separation_time = time.time() - start_time

        console.print(f"   Separation time: {separation_time:.1f}s")
        console.print(f"   Extracted stems: {len(separated.stems)}")

        # Save stems
        console.print("\n[cyan]4. Saving stems...[/cyan]")
        import soundfile as sf

        saved_count = 0
        for stem_type, stem_audio in separated.stems.items():
            # Skip if user requested specific stems and this isn't one of them
            if selected_stems and stem_type not in selected_stems:
                continue

            stem_path = output_dir / f"{input_file.stem}_{stem_type.value}.wav"
            sf.write(str(stem_path), stem_audio.audio, stem_audio.sample_rate)

            # Get audio stats
            max_amplitude = abs(stem_audio.audio).max()
            console.print(f"   Saved: {stem_path.name} (peak: {max_amplitude:.3f})")
            saved_count += 1

        # Summary
        console.print(f"\n[green][OK] Stem separation complete![/green]")
        console.print(f"   Output directory: {output_dir}")
        console.print(f"   Saved {saved_count} stem(s)")
        console.print(f"   Processing time: {separation_time:.1f}s")

        # Show next steps
        console.print("\n[dim]Next steps:[/dim]")
        console.print(f"[dim]  Transcribe stems: song-analyzer transcribe {output_dir}/<stem>.wav -o output.mid[/dim]")
        console.print(f"[dim]  Analyze music: song-analyzer analyze {input_file}[/dim]")

        if keep_download and temp_dir:
            console.print(f"\n   [dim]Downloaded file: {input_file}[/dim]")

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
