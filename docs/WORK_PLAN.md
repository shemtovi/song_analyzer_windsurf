# Song Analyzer - Work Plan

This document tracks development progress and planned work for the Song Analyzer project.

**Last Updated:** 2026-01-19

---

## Current Focus: Stages 1-3 Improvements

**Goal:** Reduce false positives (extra notes/chords), improve noise filtering, and optimize separation performance.

### Identified Issues

| Issue | Layer | Root Cause | Impact |
|-------|-------|------------|--------|
| Too many false notes | Transcription | Low confidence thresholds (0.5 CREPE, 0.3 onset) | Extra notes in output |
| Ghost chords detected | Inference | No chord confidence threshold, min 2 notes | Spurious chord detections |
| Background noise passes through | Processing | Cleanup not aggressive by default | Noise becomes notes |
| Slow separation | Separation | htdemucs model, no fast alternatives | Long processing time |

---

## Improvement Roadmap

### Phase A: Note Filtering Improvements (Transcription Layer)

- [ ] **A.1 Increase CREPE confidence threshold**
  - File: `src/transcription/monophonic.py:19`
  - Current: `pitch_confidence_threshold: 0.5`
  - Target: `0.65-0.7` (configurable)
  - Test: Single note detection accuracy

- [ ] **A.2 Improve onset detection threshold**
  - File: `src/transcription/monophonic.py:17`
  - Current: `onset_threshold: 0.3`
  - Target: `0.4-0.5` (adaptive based on noise floor)
  - Test: Note onset accuracy

- [ ] **A.3 Add polyphonic peak validation**
  - File: `src/transcription/polyphonic.py:228-238`
  - Issue: CQT peak detection accepts any peak > 30% of segment max
  - Fix: Add harmonicity check, spectral flatness filter
  - Test: Chord detection precision

- [ ] **A.4 Add minimum note energy threshold**
  - File: `src/transcription/monophonic.py:197`
  - Issue: RMS-to-velocity mapping accepts very quiet signals
  - Fix: Add configurable `min_rms_threshold`
  - Test: Ghost note rejection

### Phase B: Chord Detection Improvements (Inference Layer)

- [ ] **B.1 Add chord confidence threshold**
  - File: `src/inference/chords.py:520-565`
  - Issue: Always returns best match, even if poor
  - Fix: Add `min_chord_confidence: 0.5` parameter
  - Test: Chord detection precision/recall

- [ ] **B.2 Increase minimum notes for chord**
  - File: `src/inference/chords.py:262`
  - Current: `min_notes_for_chord: 2`
  - Target: `3` (triads minimum)
  - Test: False chord rejection

- [ ] **B.3 Increase extra note penalty**
  - File: `src/inference/chords.py:552`
  - Current: `-0.05` per extra note
  - Target: `-0.1` to `-0.15`
  - Test: Extended chord vs noise differentiation

- [ ] **B.4 Add noise-based chord rejection**
  - Issue: Noisy segments still produce chord detections
  - Fix: Check spectral flatness before chord analysis
  - Test: Silence/noise handling

### Phase C: Enhanced Cleanup Pipeline (Processing Layer)

- [ ] **C.1 Enable aggressive cleanup by default for noisy audio**
  - File: `src/processing/cleanup.py:21-45`
  - Current: `enable_all: False`
  - Fix: Add noise detection, auto-enable aggressive mode
  - Test: Noisy audio cleanup stats

- [ ] **C.2 Improve harmonic detection**
  - File: `src/processing/cleanup.py:275-332`
  - Issue: Only checks 2-5x frequency ratios
  - Fix: Add spectral centroid validation, check more ratios
  - Test: Harmonic removal accuracy

- [ ] **C.3 Add adaptive velocity threshold**
  - File: `src/processing/cleanup.py:198-200`
  - Current: Fixed `min_velocity: 20`
  - Fix: Calculate from audio's noise floor
  - Test: Dynamic threshold effectiveness

- [ ] **C.4 Add note density filter**
  - Issue: Unrealistic note densities (too many notes per second)
  - Fix: Configurable `max_notes_per_second` filter
  - Test: Note density validation

### Phase D: Separation Performance (Separation Layer)

- [ ] **D.1 Add model speed presets**
  - File: `src/separation/demucs_separator.py:200-205`
  - Add: `speed` parameter ("fast", "balanced", "quality")
  - fast: `mdx` model
  - balanced: `htdemucs` (current default)
  - quality: `htdemucs_ft`
  - Test: Speed vs quality benchmarks

- [ ] **D.2 Optimize segment processing**
  - File: `src/separation/demucs_separator.py:219`
  - Current: `segment_length: 10.0`
  - Test: Optimal segment size for speed/quality

- [ ] **D.3 Add stem-specific processing**
  - Issue: All stems processed even if only one needed
  - Fix: Early exit if only specific stems requested
  - Test: Single-stem processing time

- [ ] **D.4 Improve fallback quality**
  - File: `src/separation/demucs_separator.py:661-771`
  - Issue: Simple bandpass filters are crude
  - Fix: Add spectral unmixing, better frequency bands
  - Test: Fallback quality metrics

### Phase E: Testing & Validation

- [ ] **E.1 Create noise rejection test suite**
  - Test white noise → expect 0 notes
  - Test pink noise → expect 0 notes
  - Test silence → expect 0 notes

- [ ] **E.2 Create precision/recall benchmarks**
  - Compare detected notes vs ground truth
  - Measure chord detection accuracy
  - Track false positive rates

- [ ] **E.3 Create performance benchmarks**
  - Separation time by model
  - Transcription time by length
  - Memory usage profiling

- [ ] **E.4 Add regression tests**
  - Test files with known issues
  - Prevent reintroduction of bugs

---

## Completed Work (Since 19ae231)

### Commit History

| Commit | Description | Files Changed |
|--------|-------------|---------------|
| `19ae231` | Added benchmarking framework and test fixtures | 6 files (+989 lines) |
| `d14883f` | Added ARCHITECTURE.md and improved README | 2 files (+478 lines) |
| `c266eb6` | Transcriber refactor - removed legacy code | 5 files (-626 lines) |
| `92cfcfd` | Added CleanupConfig and CleanupStats to processing | 2 files (+434 lines) |
| `49879fe` | Refactored CLI with improved commands | 1 file (+146 lines) |
| `cb6545d` | Added disk-based caching to SourceSeparator | 1 file (+274 lines) |
| `a08f184` | Added WORK_PLAN.md document | 1 file (+174 lines) |

---

## Test Status

- **Total Tests:** 112
- **Status:** All passing
- **Target:** Add 20+ tests for improvements

---

## Key Thresholds Reference

| Parameter | File | Current | Proposed |
|-----------|------|---------|----------|
| `pitch_confidence_threshold` | monophonic.py:19 | 0.5 | 0.65-0.7 |
| `onset_threshold` | monophonic.py:17 | 0.3 | 0.4-0.5 |
| `min_chord_duration` | chords.py:260 | 0.25s | 0.25s (keep) |
| `min_notes_for_chord` | chords.py:262 | 2 | 3 |
| `min_velocity` | cleanup.py:22 | 20 | adaptive |
| `frame_threshold` | polyphonic.py:25 | 0.1 | 0.2-0.3 |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_transcriber.py -v

# Run benchmarks
python tests/benchmarks/bench_transcription.py
```

---

## Implementation Order

1. **Start with A.1** - Increase CREPE threshold (quick win, low risk)
2. **Then B.1** - Add chord confidence threshold
3. **Then C.1** - Enable aggressive cleanup for noisy audio
4. **Then E.1** - Create noise rejection tests
5. **Continue with remaining items in order**

Each change should be:
1. Implemented
2. Tested locally
3. Committed with descriptive message
4. Verified with pytest
