# Audio-to-Sheet-Music Transcription System
## Technical Design Document

**Version:** 1.0  
**Date:** January 2026  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Existing Approaches & Technologies](#2-existing-approaches--technologies)
3. [Classical DSP vs ML/AI Comparison](#3-classical-dsp-vs-mlai-comparison)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Phased Implementation Plan](#5-phased-implementation-plan)
6. [Recommended Libraries & Tools](#6-recommended-libraries--tools)
7. [Technical Challenges & Mitigations](#7-technical-challenges--mitigations)
8. [MVP Scope](#8-mvp-scope)
9. [Next Steps](#9-next-steps)

---

## 1. Executive Summary

This document outlines an **Automatic Music Transcription (AMT)** system converting audio recordings into sheet music. The system processes WAV/MP3/MP4 files and outputs MIDI, MusicXML, or PDF scores.

**Key Design Decisions:**
- Hybrid approach: classical DSP for preprocessing + deep learning for transcription
- Phased rollout: monophonic → polyphonic → multi-instrument
- Primary output: MIDI (intermediate) → MusicXML/PDF (final)

---

## 2. Existing Approaches & Technologies

### 2.1 Signal Processing Fundamentals

#### FFT / STFT
- **FFT:** Converts time-domain to frequency-domain
- **STFT:** Applies FFT to overlapping windows → spectrogram
- **Parameters:** n_fft=2048, hop_length=512, window=hann

#### Mel Spectrograms
- Perceptually-motivated frequency scale (logarithmic)
- Typical: 80-128 Mel bands, fmin=30Hz, fmax=8000Hz
- Better matches human pitch perception

### 2.2 Pitch Detection

| Algorithm | Type | Strengths | Weaknesses |
|-----------|------|-----------|------------|
| **YIN** | Classical | Fast, no training | Octave errors, mono only |
| **pYIN** | Classical+HMM | Smoother tracking | Still mono only |
| **CREPE** | Deep Learning | State-of-art mono | GPU needed, mono only |

### 2.3 Onset Detection
- **Spectral Flux:** Measures spectral change between frames
- **Complex Domain:** Combines magnitude + phase deviation
- **Peak Picking:** Adaptive threshold + minimum inter-onset interval

### 2.4 Rhythm & Tempo
- **Beat Tracking:** ODF → autocorrelation → dynamic programming
- **Quantization:** Snap note times to musical grid (8th/16th notes)

### 2.5 Symbolic Representations

| Format | Use Case | Library |
|--------|----------|---------|
| **MIDI** | Universal interchange | pretty_midi, mido |
| **MusicXML** | Sheet music notation | music21 |
| **PDF** | Final rendered score | MuseScore CLI |

### 2.6 AI/Deep Learning Approaches

#### Onsets and Frames (Google Magenta, 2018)
- CNN + BiLSTM architecture
- Outputs: onset probs, frame probs, velocities (88 piano keys)
- Note-level F1: ~96% on MAESTRO dataset

#### MT3 (Multi-Task Transcription, 2022)
- T5-style Transformer encoder-decoder
- Outputs token sequence (MIDI-like events)
- Handles multiple instruments

---

## 3. Classical DSP vs ML/AI Comparison

| Aspect | Classical DSP | ML/AI |
|--------|---------------|-------|
| **Accuracy (Mono)** | 85-95% | 95-99% |
| **Accuracy (Poly)** | 40-60% | 80-96% |
| **Compute** | CPU, real-time | GPU recommended |
| **Training Data** | None | Large datasets |
| **Interpretability** | High | Low |
| **Multi-instrument** | Poor | Good |

**Recommendation:** Hybrid approach - DSP for preprocessing, ML for transcription.

---

## 4. Proposed Architecture

### 4.1 System Overview

```
┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐
│  INPUT   │──▶│ FEATURE  │──▶│TRANSCRIBE │──▶│  POST-   │──▶│ OUTPUT │
│ HANDLER  │   │EXTRACTION│   │  (ML)     │   │ PROCESS  │   │ RENDER │
└──────────┘   └──────────┘   └───────────┘   └──────────┘   └────────┘
     │              │              │               │              │
   WAV/MP3      Mel Spec       Onsets &        Quantize        MIDI
   MP4/FLAC     CQT            Frames          Key detect      MusicXML
   Normalize    Chromagram     CREPE           Tempo           PDF
```

### 4.2 Component Details

**Input Handler:**
- Formats: WAV, FLAC, MP3, MP4, OGG
- Resample to 22.05kHz, mono, normalized

**Feature Extraction:**
- Mel spectrogram: 229 bins, n_fft=2048, hop=512
- Optional: CQT, chromagram

**Transcription:**
- Monophonic: CREPE + onset detection
- Polyphonic: Onsets and Frames
- Multi-instrument: Source separation + per-stem

**Post-Processing:**
- Tempo/beat tracking
- Quantization to grid
- Key signature detection
- Note cleanup (merge short notes, remove ghosts)

**Output:**
- MIDI → MusicXML (music21) → PDF (MuseScore)

---

## 5. Phased Implementation Plan

### Phase 1: Monophonic (2-3 weeks) - MVP
- Single melody (voice, flute, violin)
- CREPE pitch + librosa onsets
- Basic MIDI output
- **Success:** 90%+ pitch accuracy on clean recordings

### Phase 2: Polyphonic (3-4 weeks)
- Piano, guitar chords
- Onsets and Frames model
- Velocity, MusicXML export
- **Success:** Note F1 > 85% on piano

### Phase 3: Multi-Instrument (4-6 weeks)
- Source separation (Demucs)
- Per-stem transcription
- Instrument classification
- **Success:** 4+ stems, per-instrument F1 > 75%

### Phase 4: Sheet Music (2-3 weeks)
- Voice separation (treble/bass)
- PDF rendering via MuseScore
- Layout optimization
- **Success:** Readable, importable notation

---

## 6. Recommended Libraries & Tools

### Core Stack
```
Python 3.10+
├── librosa 0.10+      # Audio analysis
├── torch 2.0+         # Deep learning
├── pretty_midi        # MIDI I/O
├── music21            # MusicXML
└── typer              # CLI
```

### Pretrained Models
| Model | Task | Source |
|-------|------|--------|
| CREPE | Mono pitch | `pip install crepe` |
| Onsets & Frames | Piano | Magenta |
| Demucs | Source sep | Facebook |
| Basic Pitch | Poly pitch | Spotify |

### External Tools
- **MuseScore:** PDF rendering (`mscore -o out.pdf in.musicxml`)
- **FFmpeg:** Audio format conversion

---

## 7. Technical Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| **Polyphony accuracy** | Use Onsets & Frames, onset-gated frames |
| **Octave errors** | Pitch continuity constraints, instrument range validation |
| **Tempo ambiguity** | Multiple algorithms + voting, user override |
| **Noise sensitivity** | Preprocessing filters, train on augmented data |
| **Instrument bleeding** | Source separation before transcription |
| **Quantization artifacts** | Adaptive grid resolution, swing detection |

---

## 8. MVP Scope

### MVP Definition (Phase 1)
**Input:** Clean WAV file, single monophonic instrument/voice
**Output:** MIDI file with detected notes

### MVP Features
1. Load WAV/MP3 audio
2. Extract pitch contour (CREPE)
3. Detect note onsets (spectral flux)
4. Segment into discrete notes
5. Map to MIDI pitches
6. Export MIDI file
7. CLI interface

### MVP Non-Goals
- Polyphonic transcription
- Multiple instruments
- Sheet music rendering
- Real-time processing
- Web UI

### MVP Success Criteria
- Pitch accuracy > 90% on clean recordings
- Note count within ±10% of ground truth
- MIDI playback recognizable as original

---

## 9. Next Steps

### Immediate (Week 1)
1. Set up project structure
2. Install dependencies (librosa, crepe, pretty_midi)
3. Implement AudioLoader class
4. Test CREPE on sample files

### Short-term (Weeks 2-3)
5. Implement onset detection
6. Build note segmentation logic
7. Create MIDI exporter
8. Add tempo detection
9. Build CLI with typer
10. Test on diverse monophonic samples

### Medium-term (Month 2)
11. Integrate Onsets and Frames
12. Add GPU inference pipeline
13. Implement MusicXML export
14. Evaluation framework with mir_eval

---

## Appendix: Project Structure

```
song_analyzer/
├── src/
│   ├── __init__.py
│   ├── audio_loader.py      # Audio I/O, preprocessing
│   ├── features.py          # Mel spec, CQT, chromagram
│   ├── transcriber/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract transcriber
│   │   ├── monophonic.py    # CREPE-based
│   │   └── polyphonic.py    # Onsets & Frames
│   ├── postprocess.py       # Quantization, cleanup
│   ├── exporter.py          # MIDI, MusicXML, PDF
│   └── cli.py               # Command-line interface
├── tests/
├── models/                  # Pretrained model weights
├── examples/                # Sample audio files
├── requirements.txt
└── README.md
```

---

*Document generated for audio-to-sheet-music transcription project.*
