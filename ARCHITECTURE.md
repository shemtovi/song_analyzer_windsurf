# Song Analyzer Architecture

## Current Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLI Commands                               │
├─────────────┬─────────────────┬─────────────────┬───────────────────┤
│ transcribe  │    analyze      │    separate     │      info         │
└──────┬──────┴────────┬────────┴────────┬────────┴────────┬──────────┘
       │               │                 │                 │
       │               │                 │                 │
       v               v                 v                 v
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ AudioLoader  │  │   YouTube    │  │  Validation  │              │
│  │  (loader.py) │──│ (youtube.py) │──│              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Features   │  │    Tempo     │  │   Spectral   │              │
│  │ (features.py)│  │(tempo.py)    │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                v               v               v
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ TRANSCRIBE   │  │   ANALYZE    │  │  SEPARATE    │
     │              │  │              │  │              │
     │ 1. Load      │  │ 1. Load      │  │ 1. Load      │
     │ 2. Transcribe│  │ 2. Transcribe│  │ 2. Separate  │
     │ 3. Quantize  │  │ 3. Detect Key│  │ 3. Transcribe│
     │ 4. Export    │  │ 4. Chords    │  │ 4. Export    │
     │              │  │ 5. Harmony   │  │              │
     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
            │                 │                 │
            v                 v                 v
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ TRANSCRIPTION│  │  INFERENCE   │  │ SEPARATION   │
     │              │  │              │  │              │
     │ Monophonic   │  │ Key Detector │  │   Demucs     │
     │ Polyphonic   │  │ Chord Analyzer│ │ Multi-Inst   │
     └──────┬───────┘  └──────────────┘  └──────┬───────┘
            │                                    │
            v                                    v
     ┌──────────────┐                    ┌──────────────┐
     │  PROCESSING  │                    │ Per-Stem     │
     │              │                    │ Transcription│
     │  Quantizer   │                    │              │
     │  NoteCleanup │                    └──────────────┘
     └──────┬───────┘
            │
            v
     ┌──────────────┐
     │   OUTPUT     │
     │              │
     │ MIDIExporter │
     └──────────────┘
```

## Current Issues

1. **`separate` does TOO MUCH**: It separates audio AND transcribes each stem
2. **Redundant transcription**: Both `transcribe` and `analyze` do transcription
3. **Mixed concerns**: Commands mix separation, transcription, and analysis

---

## Proposed Improved Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLI Commands                               │
├─────────────┬─────────────────┬─────────────────┬───────────────────┤
│ transcribe  │    analyze      │    separate     │      info         │
│             │                 │                 │                   │
│ Purpose:    │ Purpose:        │ Purpose:        │ Purpose:          │
│ Audio→MIDI  │ Musical Analysis│ Audio→Stems     │ File Info         │
└──────┬──────┴────────┬────────┴────────┬────────┴────────┬──────────┘
       │               │                 │                 │
       │               │                 │                 │
       v               v                 v                 v
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ AudioLoader  │──│   YouTube    │──│  Validation  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                v
                ┌───────────────┴───────────────┐
                │                               │
                v                               v
    ┌─────────────────────┐       ┌─────────────────────┐
    │   SEPARATE          │       │   TRANSCRIBE        │
    │                     │       │                     │
    │  1. Load Audio      │       │  1. Load Audio      │
    │  2. Run Demucs      │       │     (or stems)      │
    │  3. Save Stems      │       │  2. Transcribe      │
    │  4. DONE            │       │  3. Quantize        │
    │                     │       │  4. Cleanup         │
    └──────────┬──────────┘       │  5. Export MIDI     │
               │                  └──────────┬──────────┘
               │                             │
               │                             │
               │  ┌──────────────────────────┘
               │  │
               │  │          ┌─────────────────────┐
               │  │          │   ANALYZE           │
               │  │          │                     │
               │  │          │  1. Load Audio      │
               │  └──────────│     (or MIDI)       │
               └─────────────│  2. Transcribe      │
                             │     (if needed)     │
                             │  3. Key Detection   │
                             │  4. Chord Analysis  │
                             │  5. Harmony Analysis│
                             │  6. Display Results │
                             └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        CORE COMPONENTS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ SEPARATION   │  │TRANSCRIPTION │  │  INFERENCE   │             │
│  │              │  │              │  │              │             │
│  │  Demucs      │  │ Monophonic   │  │ Key Detector │             │
│  │  Stem        │  │ Polyphonic   │  │ Chord        │             │
│  │  Splitter    │  │ Multi-Inst   │  │ Harmony      │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ PROCESSING   │  │   ANALYSIS   │  │   OUTPUT     │             │
│  │              │  │              │  │              │             │
│  │ Quantizer    │  │ Tempo        │  │ MIDI Export  │             │
│  │ NoteCleanup  │  │ Features     │  │ MusicXML     │             │
│  │              │  │ Spectral     │  │              │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Communication Flow

### 1. SEPARATE Command (Audio → Stems)
```
User Input (Audio/URL)
    │
    v
┌─────────────┐
│ AudioLoader │ ← Downloads from YouTube if URL
└──────┬──────┘
       │
       v
┌─────────────┐
│   Demucs    │ ← Separates into stems (drums, bass, vocals, other)
│ Separator   │
└──────┬──────┘
       │
       v
┌─────────────┐
│ Save Stems  │ ← Saves WAV files for each stem
│   to Disk   │   (output/song_drums.wav, etc.)
└─────────────┘
```

### 2. TRANSCRIBE Command (Audio → MIDI)
```
User Input (Audio/URL or Stem WAV)
    │
    v
┌─────────────┐
│ AudioLoader │ ← Loads audio or stem
└──────┬──────┘
       │
       v
┌─────────────┐
│ Transcriber │ ← Monophonic or Polyphonic
│ (Mono/Poly) │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Quantizer  │ ← Aligns to tempo grid
└──────┬──────┘
       │
       v
┌─────────────┐
│ NoteCleanup │ ← Removes artifacts
└──────┬──────┘
       │
       v
┌─────────────┐
│ MIDI Export │ ← Saves MIDI file
└─────────────┘
```

### 3. ANALYZE Command (Audio → Analysis Report)
```
User Input (Audio/URL)
    │
    v
┌─────────────┐
│ AudioLoader │
└──────┬──────┘
       │
       v
┌─────────────┐
│ Transcriber │ ← Gets notes for analysis
└──────┬──────┘
       │
       ├─────────────┐
       │             │
       v             v
┌─────────────┐  ┌─────────────┐
│ Key Detector│  │ Tempo       │
└──────┬──────┘  └──────┬──────┘
       │                │
       v                │
┌─────────────┐         │
│Chord Analyzer│        │
└──────┬──────┘         │
       │                │
       v                │
┌─────────────┐         │
│  Harmony    │         │
│  Analyzer   │         │
└──────┬──────┘         │
       │                │
       └────────┬───────┘
                v
       ┌─────────────┐
       │Display Report│
       │ (Console)    │
       └─────────────┘
```

---

## Workflow Examples

### Example 1: Complete Multi-Instrument Processing
```bash
# Step 1: Separate audio into stems
python -m src.cli separate "https://youtube.com/watch?v=..." -o output/

# Step 2: Transcribe each stem to MIDI
python -m src.cli transcribe output/song_drums.wav -o output/drums.mid
python -m src.cli transcribe output/song_bass.wav -o output/bass.mid
python -m src.cli transcribe output/song_vocals.wav -o output/vocals.mid
python -m src.cli transcribe output/song_other.wav -o output/guitar.mid

# Step 3: Analyze the full song
python -m src.cli analyze "https://youtube.com/watch?v=..."
```

### Example 2: Quick Transcription
```bash
# Direct transcription without separation
python -m src.cli transcribe song.wav -o output.mid -p
```

### Example 3: Analysis Only
```bash
# Get musical analysis without MIDI export
python -m src.cli analyze song.wav
```

---

## Key Principles

1. **Separation of Concerns**
   - `separate`: Audio processing only (Demucs)
   - `transcribe`: Note detection + MIDI export
   - `analyze`: Musical analysis + reporting

2. **Component Independence**
   - Each layer can work independently
   - Commands can be chained via files
   - Reusable components

3. **Clear Data Flow**
   - Input → Process → Output
   - No mixed responsibilities
   - Explicit dependencies

4. **Flexibility**
   - Can process stems separately
   - Can skip separation if not needed
   - Can analyze without transcribing
