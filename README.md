# Song Analyzer - Audio to Sheet Music

Automatic Music Transcription (AMT) system that converts audio recordings into musical notation.

## Features

- **Audio Input:** WAV, MP3, MP4, FLAC, OGG, M4A
- **URL Download:** YouTube, SoundCloud, etc. via yt-dlp
- **Pitch Detection:** pYIN (monophonic), CQT-based (polyphonic), neural (when available)
- **Source Separation:** Demucs 4-stem (drums, bass, vocals, other) and 6-stem (+ guitar, piano)
- **Key & Chord Detection:** Krumhansl-Temperley key profiles, chord progression analysis
- **Output Formats:** MIDI, MusicXML

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic transcription (monophonic)
python -m src.cli transcribe input.wav -o output.mid

# Polyphonic transcription (chords)
python -m src.cli transcribe input.wav -p -o output.mid

# Full analysis (key, chords, harmony)
python -m src.cli analyze audio.wav

# Audio file info
python -m src.cli info audio.wav

# YouTube support - works with all commands
python -m src.cli transcribe "https://youtube.com/watch?v=..." -o output.mid
python -m src.cli analyze "https://youtube.com/watch?v=..."
```

## Multi-Instrument Workflow

Separate audio into individual instrument stems, then transcribe each:

```bash
# Step 1: Separate audio into stems (drums, bass, vocals, other)
python -m src.cli separate audio.wav -o stems/

# Step 2: Transcribe each stem to MIDI
python -m src.cli transcribe stems/audio_drums.wav -o drums.mid -p
python -m src.cli transcribe stems/audio_bass.wav -o bass.mid
python -m src.cli transcribe stems/audio_vocals.wav -o vocals.mid
python -m src.cli transcribe stems/audio_other.wav -o guitar.mid -p

# Or use 6-stem model for guitar/piano separation
python -m src.cli separate audio.wav --model htdemucs_6s -o stems/
```

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Complete | Monophonic transcription |
| Phase 2 | Complete | Polyphonic transcription (piano, guitar) |
| Phase 3 | Complete | Multi-instrument separation (Demucs) |
| Phase 4 | Planned | PDF sheet music rendering |

## Architecture

Seven-layer pipeline:

```
                                    [Separation]
                                         |
                                         v
Audio --> Input --> Analysis --> Transcription --> Inference --> Processing --> Output
           |           |              |                |              |            |
       (loader)    (features)    (mono/poly)      (key/chord)    (quantize)    (MIDI)
                   (pitch)       (multi-inst)     (harmony)      (cleanup)   (MusicXML)
                   (tempo)
```

| Layer | Purpose | Key Components |
|-------|---------|----------------|
| **Input** | Audio loading | WAV/MP3/FLAC/OGG/M4A/MP4 support via librosa |
| **Analysis** | Feature extraction | Mel spectrogram, CQT, chromagram, pYIN, beat tracking |
| **Transcription** | Note detection | Monophonic, polyphonic, multi-instrument transcribers |
| **Separation** | Source separation | Demucs 4/6-stem, fallback bandpass filtering |
| **Inference** | Musical analysis | Key detection, chord analysis, harmony, cadences |
| **Processing** | Note refinement | Quantization, ghost note removal, overlap cleanup |
| **Output** | Export | MIDI (pretty_midi), MusicXML (music21) |

## CLI Commands

### `transcribe`
Convert audio to MIDI. Supports local files and YouTube URLs.

```bash
python -m src.cli transcribe input.wav [options]
python -m src.cli transcribe "https://youtube.com/watch?v=..." -o output.mid

Options:
  -o, --output PATH      Output MIDI file (default: input.mid)
  -t, --tempo FLOAT      Override tempo (BPM), 0 = auto-detect
  -p, --polyphonic       Use polyphonic mode for chords/multiple notes
  -q, --quantize         Quantize notes to tempo grid (default: on)
  -a, --aggressive       Aggressive cleanup (remove harmonics, outliers)
  --sensitivity TEXT     Transcription sensitivity: low/medium/high/ultra
  -v, --verbose          Show detailed output
```

### `analyze`
Full harmony analysis with key, chord, and structure detection.

```bash
python -m src.cli analyze input.wav [options]
python -m src.cli analyze "https://youtube.com/watch?v=..." [options]

Options:
  -p, --polyphonic       Use polyphonic mode (default: on)
  -k, --keep             Keep downloaded audio file (for URLs)
  -v, --verbose          Show detailed output
```

### `separate`
Separate audio into individual instrument stems (audio files only, no transcription).

```bash
python -m src.cli separate input.wav [options]
python -m src.cli separate "https://youtube.com/watch?v=..." -o stems/

Options:
  -o, --output PATH      Output directory for stem WAV files
  -s, --stems TEXT       Comma-separated stems to extract (drums,bass,vocals,guitar,piano,other)
  -m, --model TEXT       htdemucs (4-stem) or htdemucs_6s (6-stem, includes guitar/piano)
  -k, --keep             Keep downloaded audio file (for URLs)
  --no-cache             Disable stem caching
  -v, --verbose          Show detailed output

Note: This command ONLY separates audio. Use 'transcribe' on output stems to generate MIDI.
```

### `info`
Show audio file information.

```bash
python -m src.cli info input.wav
```

## Requirements

- Python 3.9+
- FFmpeg (included via `imageio-ffmpeg` package for YouTube downloads and audio conversion)
- GPU recommended for neural transcription and Demucs separation

### Core Dependencies
```
librosa>=0.10.0
soundfile>=0.12.1
pretty_midi>=0.2.10
typer>=0.9.0
rich>=13.0.0
numpy>=1.24.0
```

### Optional Dependencies
```
demucs>=4.0.0          # Source separation
torch>=2.0.0           # Neural models
yt-dlp>=2023.0.0       # URL download
music21>=9.0.0         # MusicXML export
pydub>=0.25.1          # Audio format conversion
```

## Documentation

- [Technical Design Document](TECHNICAL_DESIGN_DOCUMENT.md) - Full system design
- [Extended Details](TECHNICAL_DESIGN_EXTENDED.md) - Diagrams and algorithms

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## License

MIT
