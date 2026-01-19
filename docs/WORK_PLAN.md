# Song Analyzer - Work Plan

This document tracks development progress and planned work for the Song Analyzer project.

**Last Updated:** 2026-01-19

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

### Summary of Changes

1. **Testing Infrastructure (19ae231)**
   - Benchmarking framework (`tests/benchmarks/bench_framework.py`)
   - Transcription benchmarks (`tests/benchmarks/bench_transcription.py`)
   - Test fixture generator (`tests/fixtures/generate_fixtures.py`)
   - Baseline metrics (`tests/benchmarks/baselines.json`)

2. **Documentation (d14883f)**
   - Complete architecture documentation (`docs/ARCHITECTURE.md`)
   - Updated README with project status, CLI commands, architecture diagram

3. **Code Cleanup (c266eb6)**
   - Removed legacy `src/transcriber/` folder (duplicate of `src/transcription/`)
   - Cleaned up multi_instrument.py imports

4. **Processing Enhancement (92cfcfd)**
   - Added `CleanupConfig` dataclass for configurable note cleanup
   - Added `CleanupStats` dataclass for cleanup statistics
   - Enhanced `NoteCleanup` class with detailed statistics tracking

5. **CLI Refactor (49879fe)**
   - Improved command structure and options
   - Better error handling and output formatting

6. **Caching Feature (cb6545d)**
   - Disk-based caching for Demucs separation results
   - MD5-based cache keys from audio content
   - 24-hour TTL with configurable expiration
   - Cache management methods (clear_cache, get_cache_stats)
   - Automatic GPU detection with detailed device info

---

## Current Test Status

- **Total Tests:** 112
- **Status:** All passing
- **Warnings:** 3 (expected - related to Demucs fallback and librosa FFT size)

---

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Complete | Monophonic transcription (pYIN-based) |
| Phase 2 | Complete | Polyphonic transcription (CQT-based) |
| Phase 3 | Complete | Multi-instrument separation (Demucs) |
| Phase 4 | **Planned** | PDF sheet music rendering |

---

## Next Steps (Phase 4: PDF Sheet Music)

### High Priority

- [ ] **4.1 LilyPond Integration**
  - Add LilyPond export from Note list
  - Support for key signatures, time signatures
  - Basic engraving options

- [ ] **4.2 PDF Generation**
  - LilyPond to PDF pipeline
  - Sheet music layout configuration
  - Page sizing and margins

- [ ] **4.3 CLI Command**
  - Add `render` command for PDF output
  - Options: paper size, template, layout

### Medium Priority

- [ ] **4.4 Music Notation Improvements**
  - Automatic beaming
  - Tuplet detection and notation
  - Dynamic markings from velocity

- [ ] **4.5 Multi-voice Support**
  - Split polyphonic notes into voices
  - Automatic voice separation algorithm
  - Grand staff for piano

### Lower Priority

- [ ] **4.6 Score Presentation**
  - Title, composer, tempo markings
  - Rehearsal marks
  - Repeat signs and endings

---

## Technical Debt

- [ ] Add unit tests for new caching feature in SourceSeparator
- [ ] Add integration tests for CLI commands
- [ ] Improve error messages for missing dependencies
- [ ] Document cache configuration options

---

## Performance Improvements (Future)

- [ ] GPU batch processing for multiple files
- [ ] Streaming transcription for long audio
- [ ] Memory optimization for large files
- [ ] Parallel stem transcription

---

## How to Update This Plan

1. Mark completed items with `[x]`
2. Add new items as they are identified
3. Update the "Last Updated" date
4. Commit with descriptive message

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_multi_instrument.py -v

# Run benchmarks
python tests/benchmarks/bench_transcription.py
```

---

## Git Workflow

```bash
# Check current status
git status

# View recent commits
git log --oneline -10

# Commit incrementally
git add <file>
git commit -m "descriptive message"

# Run tests before pushing
pytest tests/ -v
```
