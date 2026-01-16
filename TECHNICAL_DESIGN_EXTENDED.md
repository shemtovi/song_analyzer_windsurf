# Extended Technical Details
## Audio-to-Sheet-Music Transcription System

This document provides additional depth on signal processing, architectures, and implementation details.

---

## 1. Signal Processing Deep Dive

### 1.1 FFT/STFT Visualization

```
Time Domain                    Frequency Domain
     │                              │
  ───┼───────────────────       ────┼────────────────
     │   /\    /\    /\            │  │
     │  /  \  /  \  /  \           │  │ │
     │ /    \/    \/    \          │  │ │   │
  ───┴───────────────────       ───┴──┴─┴───┴────────
         samples                    Hz (frequency)
```

**STFT creates a spectrogram:**
```
  Freq │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
  (Hz) │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
   ↑   │▓▓▓▓▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓▓▓│
       │████████░░░░░░░░░░████████░░░░░░░░░░████████│
       │████████▓▓▓▓▓▓▓▓▓▓████████▓▓▓▓▓▓▓▓▓▓████████│
       └─────────────────────────────────────────────
                         Time →
```

### 1.2 Mel Scale Transformation

```
Linear:  |----|----|----|----|----|----|  (equal Hz spacing)
         0   1k   2k   3k   4k   5k   6k

Mel:     |--------|--------|-----|---|--|  (perceptual spacing)
         0       1k       2k   3k  4k 5k
```

**Formula:** `mel = 2595 * log10(1 + f/700)`

### 1.3 CREPE Architecture

```
Input: 1024 samples (64ms at 16kHz)
  ↓
6 × [Conv1D → ReLU → MaxPool → Dropout]
  ↓
Fully Connected (2048 → 360)
  ↓
Output: 360 pitch bins (20Hz resolution, C1-B7)
```

---

## 2. Onsets and Frames Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 ONSETS AND FRAMES ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Mel Spectrogram (229 bins, 32ms frames)                        │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Acoustic Model (CNN)                        │    │
│  │   Conv layers → features for onset/frame/velocity       │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌───────────┐        ┌───────────┐        ┌───────────┐        │
│  │  Onset    │        │  Frame    │        │ Velocity  │        │
│  │  Stack    │───────▶│  Stack    │        │  Stack    │        │
│  │ (BiLSTM)  │        │ (BiLSTM)  │        │ (Dense)   │        │
│  └───────────┘        └───────────┘        └───────────┘        │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  [88 onset probs]    [88 frame probs]    [88 velocities]        │
│                                                                  │
│  Key: Onset predictions gate frame predictions                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Data Flow

```
audio.wav
    │
    ▼
┌───────────────┐
│ Load & Decode │ ──▶ np.array [N samples]
└───────────────┘
        │
        ▼
┌───────────────┐
│  Preprocess   │ ──▶ np.array [N' samples] (resampled, normalized)
└───────────────┘
        │
        ▼
┌───────────────┐
│ Mel Spectro.  │ ──▶ np.array [T frames × F mel bins]
└───────────────┘
        │
        ▼
┌───────────────┐
│  Transcribe   │ ──▶ List[Note(pitch, onset, offset, velocity)]
│  (ML Model)   │
└───────────────┘
        │
        ▼
┌───────────────┐
│ Post-Process  │ ──▶ List[Note] (quantized) + tempo, key
└───────────────┘
        │
        ▼
┌───────────────┐
│  Export MIDI  │ ──▶ output.mid
└───────────────┘
        │
        ▼
┌───────────────┐
│ MIDI→MusicXML │ ──▶ output.musicxml
└───────────────┘
        │
        ▼
┌───────────────┐
│ Render PDF    │ ──▶ output.pdf
└───────────────┘
```

---

## 4. Class Structure

```
┌─────────────────────┐       ┌─────────────────────┐
│    AudioLoader      │       │   FeatureExtractor  │
├─────────────────────┤       ├─────────────────────┤
│ - target_sr: int    │       │ - n_mels: int       │
│ - mono: bool        │       │ - n_fft: int        │
├─────────────────────┤       │ - hop_length: int   │
│ + load(path) → arr  │       ├─────────────────────┤
│ + normalize(arr)    │       │ + mel_spec(audio)   │
│ + trim_silence(arr) │       │ + cqt(audio)        │
└─────────────────────┘       └─────────────────────┘
         │                             │
         ▼                             ▼
┌───────────────────────────────────────────────────┐
│              Transcriber (Abstract)                │
├───────────────────────────────────────────────────┤
│ + transcribe(features) → List[Note]               │
└───────────────────────────────────────────────────┘
         △                             △
         │                             │
┌────────┴────────┐         ┌─────────┴─────────┐
│MonophonicTranscr│         │PolyphonicTranscr  │
├─────────────────┤         ├───────────────────┤
│ - pitch_model   │         │ - model (O&F)     │
│   (CREPE)       │         ├───────────────────┤
├─────────────────┤         │ + transcribe()    │
│ + transcribe()  │         └───────────────────┘
└─────────────────┘

┌─────────────────────┐       ┌─────────────────────┐
│       Note          │       │    PostProcessor    │
├─────────────────────┤       ├─────────────────────┤
│ - pitch: int        │       │ - tempo: float      │
│ - onset: float      │       │ - time_sig: tuple   │
│ - offset: float     │       ├─────────────────────┤
│ - velocity: int     │       │ + quantize(notes)   │
│ - instrument: str   │       │ + detect_tempo()    │
└─────────────────────┘       │ + detect_key()      │
                              └─────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────────┐
                              │     Exporter        │
                              ├─────────────────────┤
                              │ + to_midi(notes)    │
                              │ + to_musicxml(midi) │
                              │ + to_pdf(musicxml)  │
                              └─────────────────────┘
```

---

## 5. Quantization Algorithm

```
Raw note onsets (seconds): 0.00, 0.48, 1.02, 1.51, 2.00

Tempo: 120 BPM → beat = 0.5s, 16th = 0.125s

Grid (16th notes):
0.000  0.125  0.250  0.375  0.500  0.625  ...
  ↑                           ↑
0.00                        0.48 → snaps to 0.500

Quantized: 0.000, 0.500, 1.000, 1.500, 2.000
Musical:   beat 1, beat 2, beat 3, beat 4, beat 1
```

---

## 6. MIDI Pitch Mapping

```
C4 (Middle C) = 60
A4 (440 Hz)   = 69

Formula: MIDI = 69 + 12 × log₂(f/440)

Examples:
  261.63 Hz → C4  → MIDI 60
  440.00 Hz → A4  → MIDI 69
  880.00 Hz → A5  → MIDI 81
```

---

## 7. Training Datasets

| Dataset | Size | Content | Use |
|---------|------|---------|-----|
| MAESTRO | 200h | Piano + MIDI | Piano transcription |
| MusicNet | 34h | Chamber music | Multi-instrument |
| MAPS | 18h | Piano | Piano transcription |
| Slakh2100 | 145h | Synthesized | Multi-instrument |
| GuitarSet | 3h | Guitar | Guitar transcription |

---

## 8. Evaluation Metrics

**Note-level metrics (mir_eval):**
- **Precision:** Correct notes / predicted notes
- **Recall:** Correct notes / ground truth notes
- **F1:** Harmonic mean of precision and recall

**Tolerances:**
- Onset: ±50ms
- Offset: ±50ms or 20% of note duration
- Pitch: exact match (MIDI number)

---

## 9. Fine-tuning Strategy

```
1. Start with pretrained model (Onsets & Frames)
2. Freeze early layers (feature extraction)
3. Fine-tune later layers on domain data
4. Data augmentation:
   - Pitch shifting (±2 semitones)
   - Time stretching (0.9-1.1x)
   - Add noise/reverb
   - Random EQ
5. Evaluate on held-out test set
```

---

## 10. Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Polyphony accuracy | High | High | Use O&F model |
| Tempo errors | Medium | Medium | Multi-algorithm voting |
| Octave errors | Medium | High | Continuity constraints |
| Noise sensitivity | Medium | Medium | Preprocessing + augmentation |
| GPU memory | Medium | Low | Chunked processing |
| Instrument bleeding | High | High | Source separation first |
