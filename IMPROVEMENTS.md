# Polyphonic Transcription Improvements

## Problem Analysis

When transcribing the RHCP "_other.wav" stem (guitars, synths, etc.), the polyphonic transcriber detected 0 notes due to overly conservative filtering.

### Root Causes:
1. **Spectral flatness threshold too strict (0.8)**: Filters out complex timbres (guitars with distortion, synths)
2. **Min peak energy too high (0.1)**: Requires 10% of max energy
3. **No audio preprocessing**: No normalization or enhancement

### Diagnostic Results:
```
Audio RMS: 0.109855 (good)
Onsets detected: 656
Segments failed flatness: 19/20 (flatness ~0.96 > 0.8 threshold)
Segments failed RMS: 1/20
Segments failed energy: 0/20
```

## Solutions Implemented

### Solution 1: Relaxed Default Parameters
Modify default thresholds in `PolyphonicTranscriber.__init__()`:
- `min_peak_energy`: 0.1 → 0.03 (3% instead of 10%)
- `min_rms_threshold`: 0.003 → 0.001 (more permissive)
- Spectral flatness: 0.8 → 0.95 (allow complex timbres)

### Solution 2: Adaptive Thresholds
Calculate thresholds based on audio characteristics:
- RMS-based threshold: `max(0.0005, audio_rms * 0.05)`
- Energy-based threshold: `max(0.02, avg_cqt_energy * 0.3)`

### Solution 3: Audio Preprocessing
Add optional preprocessing pipeline:
- RMS normalization to target level
- Dynamic range compression
- Spectral enhancement

### Solution 4: CLI Options
Add command-line flags for users:
- `--sensitivity`: low/medium/high (controls thresholds)
- `--normalize`: Enable audio normalization
- `--no-flatness-filter`: Disable spectral flatness check

## Testing
Test with relaxed parameters showed ~7 notes detected (vs 0 before).
Further tuning and preprocessing should improve this significantly.
