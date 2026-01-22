# Song Analyzer Workflows

Complete guide to using Song Analyzer for different music transcription tasks.

---

## Workflow 1: Quick Single-Instrument Transcription

**Use case:** Transcribe a monophonic melody or vocal line

```bash
# From local file
python -m src.cli transcribe song.wav -o output.mid

# From YouTube
python -m src.cli transcribe "https://youtube.com/watch?v=..." -o output.mid
```

**Result:** `output.mid` - MIDI file with detected melody

---

## Workflow 2: Polyphonic Transcription (Chords/Piano)

**Use case:** Transcribe piano, guitar, or music with chords

```bash
# Enable polyphonic mode with -p flag
python -m src.cli transcribe piano.wav -p -o output.mid

# With aggressive cleanup for better accuracy
python -m src.cli transcribe piano.wav -p --aggressive -o output.mid

# High sensitivity for quiet notes
python -m src.cli transcribe piano.wav -p --sensitivity high -o output.mid
```

**Result:** `output.mid` - MIDI file with chords and multiple simultaneous notes

---

## Workflow 3: Multi-Instrument Transcription

**Use case:** Separate full band recording into individual instruments

### Step 1: Separate Audio
```bash
# Separate into 4 stems: drums, bass, vocals, other
python -m src.cli separate full_band.wav -o stems/

# Or from YouTube
python -m src.cli separate "https://youtube.com/watch?v=..." -o stems/
```

**Output:**
```
stems/
  ├── full_band_drums.wav
  ├── full_band_bass.wav
  ├── full_band_vocals.wav
  └── full_band_other.wav
```

### Step 2: Transcribe Each Stem
```bash
# Drums (polyphonic - multiple drum hits)
python -m src.cli transcribe stems/full_band_drums.wav -p -o drums.mid

# Bass (monophonic)
python -m src.cli transcribe stems/full_band_bass.wav -o bass.mid

# Vocals (monophonic)
python -m src.cli transcribe stems/full_band_vocals.wav -o vocals.mid

# Guitar/Keys (polyphonic)
python -m src.cli transcribe stems/full_band_other.wav -p -o guitar.mid
```

**Result:** 4 MIDI files, one per instrument

---

## Workflow 4: Guitar/Piano Separation (6-stem model)

**Use case:** Separate guitar and piano into individual tracks

```bash
# Step 1: Use 6-stem model
python -m src.cli separate song.wav --model htdemucs_6s -o stems/
```

**Output:**
```
stems/
  ├── song_drums.wav
  ├── song_bass.wav
  ├── song_vocals.wav
  ├── song_guitar.wav    ← Separated guitar
  ├── song_piano.wav     ← Separated piano
  └── song_other.wav
```

```bash
# Step 2: Transcribe specific instruments
python -m src.cli transcribe stems/song_guitar.wav -p -o guitar.mid
python -m src.cli transcribe stems/song_piano.wav -p -o piano.mid
```

---

## Workflow 5: Musical Analysis

**Use case:** Get key, chords, and harmony information

```bash
# Analyze local file
python -m src.cli analyze song.wav

# Analyze from YouTube
python -m src.cli analyze "https://youtube.com/watch?v=..."
```

**Output:**
```
1. File Information
   Duration: 182.44s
   Format: WAV

2. Tempo Detection
   Tempo: 99.4 BPM

3. Key Detection
   Key: C major (confidence: 0.85)

4. Chord Detection
   Detected 24 chords

   Progression: I - IV - V - I

5. Harmony Analysis
   Harmonic rhythm: 1.5 changes/beat
   Average tension: 0.45
   Musical coherence: 0.78
```

---

## Workflow 6: Selective Stem Extraction

**Use case:** Only extract specific instruments

```bash
# Only extract bass and vocals
python -m src.cli separate song.wav --stems bass,vocals -o stems/

# Only extract guitar (requires 6-stem model)
python -m src.cli separate song.wav --stems guitar --model htdemucs_6s -o stems/
```

**Result:** Only the requested stem files are saved

---

## Workflow 7: YouTube to MIDI Pipeline

**Complete workflow from YouTube URL to MIDI files**

```bash
# 1. Download and separate (creates WAV stems)
python -m src.cli separate "https://youtube.com/watch?v=dQw4w9WgXcQ" -o stems/

# 2. Transcribe vocals
python -m src.cli transcribe stems/*_vocals.wav -o vocals.mid

# 3. Transcribe instruments
python -m src.cli transcribe stems/*_other.wav -p -o instruments.mid

# 4. Analyze for key/chords
python -m src.cli analyze "https://youtube.com/watch?v=dQw4w9WgXcQ"
```

---

## Advanced Options

### Transcription Sensitivity

Control how aggressively notes are detected:

```bash
# Low sensitivity (only loud, clear notes)
python -m src.cli transcribe audio.wav --sensitivity low -o output.mid

# Medium sensitivity (default)
python -m src.cli transcribe audio.wav --sensitivity medium -o output.mid

# High sensitivity (detect quiet notes)
python -m src.cli transcribe audio.wav --sensitivity high -o output.mid

# Ultra sensitivity (maximum detection, may include noise)
python -m src.cli transcribe audio.wav --sensitivity ultra -o output.mid
```

### Quantization Control

```bash
# With quantization (snap to grid)
python -m src.cli transcribe audio.wav --quantize -o output.mid

# Without quantization (preserve exact timings)
python -m src.cli transcribe audio.wav --no-quantize -o output.mid

# Manual tempo override
python -m src.cli transcribe audio.wav --tempo 120 -o output.mid
```

### Cleanup Options

```bash
# Standard cleanup (default)
python -m src.cli transcribe audio.wav -o output.mid

# Aggressive cleanup (remove harmonics, duplicates, outliers)
python -m src.cli transcribe audio.wav --aggressive -o output.mid

# Custom thresholds
python -m src.cli transcribe audio.wav --min-velocity 30 --min-duration 0.1 -o output.mid
```

---

## Common Use Cases

### 1. Cover Song Transcription
```bash
# Get just the melody from a cover song
python -m src.cli separate "youtube_url" -o stems/
python -m src.cli transcribe stems/song_vocals.wav -o melody.mid
```

### 2. Drum Programming Reference
```bash
# Extract drum pattern from a song
python -m src.cli separate song.wav --stems drums -o stems/
python -m src.cli transcribe stems/song_drums.wav -p -o drums.mid
```

### 3. Bass Line Extraction
```bash
# Get bass line for practice
python -m src.cli separate song.wav --stems bass -o stems/
python -m src.cli transcribe stems/song_bass.wav -o bass.mid
```

### 4. Chord Chart Creation
```bash
# Analyze song for chord progression
python -m src.cli analyze song.wav
# Shows: C - Am - F - G progression
```

### 5. Remix/Production Starting Point
```bash
# Get all stems for remixing
python -m src.cli separate "youtube_url" --model htdemucs_6s -o remix/
# Now have individual WAV files to import into DAW
```

---

## Tips and Best Practices

### 1. **Choose the Right Mode**
- Monophonic (default): Single melody lines, vocals, bass
- Polyphonic (`-p`): Chords, piano, guitar, drums

### 2. **Use Aggressive Cleanup for Dense Mixes**
```bash
python -m src.cli transcribe complex_mix.wav -p --aggressive -o output.mid
```

### 3. **Adjust Sensitivity Based on Recording Quality**
- Studio recordings: `low` or `medium`
- Live recordings: `medium` or `high`
- Lo-fi recordings: `high` or `ultra`

### 4. **Cache Stem Separations**
Demucs separation is slow. The results are cached automatically:
```bash
# First run: slow (separates audio)
python -m src.cli separate song.wav -o stems/

# Second run: fast (uses cache)
python -m src.cli separate song.wav -o stems/
```

Disable caching with `--no-cache` flag if needed.

### 5. **Keep Downloaded YouTube Files**
```bash
# Keep the downloaded audio file for later use
python -m src.cli separate "youtube_url" -o stems/ --keep
```

### 6. **Process Stems Selectively**
Don't transcribe what you don't need:
```bash
# Only separate and transcribe vocals
python -m src.cli separate song.wav --stems vocals -o stems/
python -m src.cli transcribe stems/song_vocals.wav -o vocals.mid
```

---

## Troubleshooting

### Issue: Transcription has too many false notes
**Solution:** Use aggressive cleanup
```bash
python -m src.cli transcribe audio.wav --aggressive -o output.mid
```

### Issue: Missing quiet notes
**Solution:** Increase sensitivity
```bash
python -m src.cli transcribe audio.wav --sensitivity high -o output.mid
```

### Issue: Timing is off
**Solution:** Check tempo detection or override manually
```bash
# Let it detect tempo
python -m src.cli transcribe audio.wav -v -o output.mid

# Or override manually
python -m src.cli transcribe audio.wav --tempo 120 -o output.mid
```

### Issue: Drums not separating well
**Solution:** Try different Demucs model
```bash
python -m src.cli separate song.wav --model htdemucs_ft -o stems/
```

### Issue: YouTube download fails
**Solution:** Ensure imageio-ffmpeg is installed
```bash
pip install imageio-ffmpeg
```

---

## Performance Notes

### Separation Speed
- **Demucs (GPU):** ~1-2x realtime (3-minute song = 3-6 minutes)
- **Demucs (CPU):** ~0.1-0.5x realtime (3-minute song = 6-30 minutes)

### Transcription Speed
- **Monophonic:** Very fast (~10x realtime)
- **Polyphonic:** Fast (~5x realtime)
- **Neural (if available):** Slow (~0.5x realtime)

### Tips for Faster Processing
1. Use GPU if available
2. Process only needed stems (`--stems bass,vocals`)
3. Use cached separations
4. Lower sensitivity for faster transcription
