# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with code in this repository.

## Project Overview

**Song Analyzer** is an Automatic Music Transcription (AMT) system that converts audio recordings into musical notation. It processes audio files (WAV, MP3, MP4, FLAC, OGG) and outputs transcribed notes in MIDI format, with support for harmonic analysis, key detection, and multi-instrument separation.

## Tech Stack

- **Python 3.9+** with type hints throughout
- **Audio Processing:** librosa, soundfile, pydub
- **MIDI/Music:** pretty_midi, mido
- **CLI:** typer, rich
- **Source Separation:** demucs (optional)
- **Deep Learning:** torch, torchaudio (optional)
- **Testing:** pytest
- **Formatting:** black, ruff

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"

# Run CLI commands
python -m src.cli transcribe input.wav -o output.mid    # Monophonic transcription
python -m src.cli transcribe input.wav -p --quantize    # Polyphonic transcription
python -m src.cli analyze audio.wav                      # Full harmonic analysis
python -m src.cli separate audio.wav -o output_dir/      # Multi-instrument separation
python -m src.cli info audio.wav                         # Audio file info

# Run tests
pytest tests/
pytest -v tests/                    # Verbose
pytest --cov=src tests/             # With coverage

# Generate test audio files
python tests/generate_test_audio.py
```

## Architecture

Seven-layer architecture:

1. **Input** (`src/input/`) - Audio loading, format handling, preprocessing
2. **Analysis** (`src/analysis/`) - Feature extraction (mel spec, CQT, chromagram), pitch/tempo detection
3. **Transcription** (`src/transcription/`) - Monophonic, polyphonic, multi-instrument note detection
4. **Separation** (`src/separation/`) - Source separation using Demucs
5. **Inference** (`src/inference/`) - Key, chords, melody, harmony, structure analysis
6. **Processing** (`src/processing/`) - Quantization, note cleanup
7. **Output** (`src/output/`) - MIDI, MusicXML export

## Key Files

- `src/cli.py` - Main CLI interface
- `src/core/note.py` - Note dataclass (pitch, onset, offset, velocity)
- `src/core/constants.py` - Constants (sample rates, pitch names)
- `src/transcription/monophonic.py` - Single melody transcription (pYIN)
- `src/transcription/polyphonic.py` - Chord/multi-note transcription (CQT-based)
- `src/inference/key.py` - Key detection (Krumhansl/Temperley profiles)
- `src/inference/chords.py` - Chord analysis
- `src/output/midi.py` - MIDI export (pretty_midi-based)

## Code Conventions

- Type hints on all functions
- Dataclasses for data structures (Note, KeyInfo, ChordInfo)
- Google/NumPy style docstrings
- Default sample rate: 22050 Hz
- Hop length: 512 samples
