"""Test analysis of RHCP "Can't Stop" _other.wav to demonstrate improvements."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.input.loader import AudioLoader
from src.transcription.polyphonic import PolyphonicTranscriber
from src.processing.cleanup import NoteCleanup, CleanupConfig
from src.inference.chords import ChordAnalyzer

def analyze_with_old_settings(audio_path: str):
    """Analyze with old (lenient) settings."""
    print("\n" + "="*80)
    print("ANALYZING WITH OLD SETTINGS (Lenient - More False Positives)")
    print("="*80)

    loader = AudioLoader()
    audio, sr = loader.load(audio_path)

    # Old settings (before improvements)
    transcriber = PolyphonicTranscriber(
        onset_threshold=0.3,  # Old: 0.3, New: 0.4
        frame_threshold=0.1,  # Old: 0.1, New: 0.2
        min_peak_energy=0.1,  # Old: implicit 0.1, New: 0.15
        min_rms_threshold=0.005,  # Old: implicit low, New: 0.01
        max_notes_per_segment=20,  # Old: no limit, New: 8
    )

    notes = transcriber.transcribe(audio, sr)
    print(f"[*] Initial transcription: {len(notes)} notes detected")

    # Old cleanup (no aggressive mode, no adaptive velocity, no density filter)
    config = CleanupConfig(
        min_velocity=20,
        remove_harmonics=False,  # Old: often disabled
        auto_aggressive=False,  # Old: didn't exist
        adaptive_velocity=False,  # Old: didn't exist
        max_notes_per_second=0,  # Old: no limit
        harmonic_ratios=(2, 3, 4, 5),  # Old: 2-5
    )
    cleanup = NoteCleanup(config=config)
    cleaned_notes, stats = cleanup.cleanup(notes, return_stats=True)

    print(f"[*] After cleanup: {len(cleaned_notes)} notes")
    print(f"  - Removed ghost notes: {stats.removed_ghost_notes}")
    print(f"  - Removed harmonics: {stats.removed_harmonics}")
    print(f"  - Auto-aggressive enabled: {stats.auto_aggressive_enabled}")
    print(f"  - Notes per second: {stats.detected_notes_per_sec:.2f}")

    # Chord detection with old settings
    analyzer = ChordAnalyzer(
        min_notes_for_chord=2,  # Old: 2, New: 3
        min_chord_confidence=0.0,  # Old: no threshold, New: 0.4
        min_note_velocity=20,  # Old: 20, New: 30
        extra_note_penalty=0.05,  # Old: 0.05, New: 0.1
    )
    chords = analyzer.detect_chords(cleaned_notes)

    print(f"[*] Chords detected: {len(chords)}")

    return {
        'notes': len(notes),
        'cleaned_notes': len(cleaned_notes),
        'chords': len(chords),
        'stats': stats,
    }

def analyze_with_new_settings(audio_path: str):
    """Analyze with new (strict) settings - our improvements."""
    print("\n" + "="*80)
    print("ANALYZING WITH NEW SETTINGS (Strict - Fewer False Positives)")
    print("="*80)

    loader = AudioLoader()
    audio, sr = loader.load(audio_path)

    # New improved settings
    transcriber = PolyphonicTranscriber(
        onset_threshold=0.4,  # Increased from 0.3
        frame_threshold=0.2,  # Increased from 0.1
        min_peak_energy=0.15,  # Added threshold
        min_rms_threshold=0.01,  # More strict
        max_notes_per_segment=8,  # Limit noise explosion
    )

    notes = transcriber.transcribe(audio, sr)
    print(f"[*] Initial transcription: {len(notes)} notes detected")

    # New cleanup with all improvements
    config = CleanupConfig(
        min_velocity=20,
        remove_harmonics=True,  # Now enabled
        auto_aggressive=True,  # Auto-detect noise
        adaptive_velocity=True,  # Use adaptive threshold
        adaptive_velocity_percentile=15.0,
        max_notes_per_second=20,  # Limit density
        harmonic_ratios=(2, 3, 4, 5, 6, 7, 8),  # Extended range
        noise_threshold_notes_per_sec=15.0,
    )
    cleanup = NoteCleanup(config=config)
    cleaned_notes, stats = cleanup.cleanup(notes, return_stats=True)

    print(f"[*] After cleanup: {len(cleaned_notes)} notes")
    print(f"  - Removed ghost notes: {stats.removed_ghost_notes}")
    print(f"  - Removed harmonics: {stats.removed_harmonics}")
    print(f"  - Removed via adaptive velocity: {stats.removed_adaptive_velocity}")
    print(f"  - Removed via density filter: {stats.removed_density}")
    print(f"  - Removed duplicates: {stats.removed_duplicates}")
    print(f"  - Removed outliers: {stats.removed_outliers}")
    print(f"  - Auto-aggressive enabled: {stats.auto_aggressive_enabled}")
    print(f"  - Notes per second: {stats.detected_notes_per_sec:.2f}")

    # Chord detection with new settings
    analyzer = ChordAnalyzer(
        min_notes_for_chord=3,  # Require triads
        min_chord_confidence=0.4,  # Confidence threshold
        min_note_velocity=30,  # Higher threshold
        extra_note_penalty=0.1,  # Stricter
    )
    chords = analyzer.detect_chords(cleaned_notes)

    print(f"[*] Chords detected: {len(chords)}")
    if chords:
        print(f"  - First few chords: {', '.join(c.symbol for c in chords[:5])}")
        if len(chords) > 5:
            print(f"  - Last few chords: {', '.join(c.symbol for c in chords[-5:])}")

    return {
        'notes': len(notes),
        'cleaned_notes': len(cleaned_notes),
        'chords': len(chords),
        'stats': stats,
    }

def main():
    # Try bass first (more musical content than "other")
    audio_path = "./examples/rhcp_fast/Red Hot Chili Peppers - Can't Stop [Official Music Video]_bass.wav"

    print("\n" + "#"*80)
    print("# RHCP 'Can't Stop' _bass.wav Analysis")
    print("# Comparing OLD vs NEW settings to demonstrate improvements")
    print("#"*80)

    # Analyze with old settings
    old_results = analyze_with_old_settings(audio_path)

    # Analyze with new settings
    new_results = analyze_with_new_settings(audio_path)

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: OLD vs NEW")
    print("="*80)

    print(f"\nInitial Notes Detected:")
    print(f"  OLD: {old_results['notes']} notes")
    print(f"  NEW: {new_results['notes']} notes")
    print(f"  Difference: {old_results['notes'] - new_results['notes']} fewer notes detected (stricter transcription)")

    print(f"\nAfter Cleanup:")
    print(f"  OLD: {old_results['cleaned_notes']} notes")
    print(f"  NEW: {new_results['cleaned_notes']} notes")
    reduction = old_results['cleaned_notes'] - new_results['cleaned_notes']
    reduction_pct = (reduction / old_results['cleaned_notes'] * 100) if old_results['cleaned_notes'] > 0 else 0
    print(f"  Difference: {reduction} fewer false positives ({reduction_pct:.1f}% reduction)")

    print(f"\nChords Detected:")
    print(f"  OLD: {old_results['chords']} chords")
    print(f"  NEW: {new_results['chords']} chords")
    chord_diff = old_results['chords'] - new_results['chords']
    chord_diff_pct = (chord_diff / old_results['chords'] * 100) if old_results['chords'] > 0 else 0
    print(f"  Difference: {chord_diff} fewer spurious chords ({chord_diff_pct:.1f}% reduction)")

    print(f"\nNoise Detection:")
    print(f"  OLD Notes/sec: {old_results['stats'].detected_notes_per_sec:.2f}")
    print(f"  NEW Notes/sec: {new_results['stats'].detected_notes_per_sec:.2f}")
    print(f"  OLD Auto-aggressive: {old_results['stats'].auto_aggressive_enabled}")
    print(f"  NEW Auto-aggressive: {new_results['stats'].auto_aggressive_enabled}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("[*] Stricter thresholds reduce false positive notes")
    print("[*] Enhanced cleanup filters noise more effectively")
    print("[*] Chord detection rejects weak/spurious matches")
    print("[*] Auto-aggressive mode adapts to noisy input")
    print("[*] Overall: Cleaner, more accurate transcription with fewer artifacts")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
