# 🎵 Song Analyzer - Audio to Sheet Music

Automatic Music Transcription (AMT) system that converts audio recordings into musical notation.

## Features

- **Audio Input:** WAV, MP3, MP4, FLAC, OGG
- **Pitch Detection:** CREPE (monophonic), Onsets & Frames (polyphonic)
- **Output Formats:** MIDI, MusicXML, PDF sheet music

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Transcribe audio (MVP - monophonic)
python -m src.cli transcribe input.wav -o output.mid
```

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 🚧 In Progress | Monophonic transcription (MVP) |
| Phase 2 | ⏳ Planned | Polyphonic (piano, guitar) |
| Phase 3 | ⏳ Planned | Multi-instrument |
| Phase 4 | ⏳ Planned | Sheet music rendering |

## Architecture

```
Audio → Preprocessing → Feature Extraction → Transcription → Post-processing → Output
         (resample)      (Mel spectrogram)    (CREPE/O&F)     (quantize)       (MIDI)
```

## Documentation

- [Technical Design Document](TECHNICAL_DESIGN_DOCUMENT.md) - Full system design
- [Extended Details](TECHNICAL_DESIGN_EXTENDED.md) - Diagrams and algorithms

## Requirements

- Python 3.10+
- FFmpeg (for MP3/MP4 support)
- GPU recommended for deep learning models

## License

MIT
