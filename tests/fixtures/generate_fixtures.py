"""Generate synthetic test audio fixtures with known ground truth.

This module creates test audio files where we know exactly what notes
should be detected, enabling objective evaluation of transcription quality.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json


@dataclass
class GroundTruthNote:
    """A note with known ground truth for evaluation."""
    pitch: int  # MIDI pitch
    onset: float  # Start time in seconds
    offset: float  # End time in seconds
    velocity: int = 80

    def to_dict(self) -> dict:
        return {
            "pitch": self.pitch,
            "onset": self.onset,
            "offset": self.offset,
            "velocity": self.velocity,
        }


def midi_to_freq(midi: int) -> float:
    """Convert MIDI pitch to frequency."""
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def generate_sine_note(
    freq: float,
    duration: float,
    sr: int = 22050,
    amplitude: float = 0.8,
    with_harmonics: bool = True,
) -> np.ndarray:
    """Generate a sine wave note with optional harmonics and envelope."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Fundamental
    audio = np.sin(2 * np.pi * freq * t) * amplitude

    # Add harmonics for more realistic timbre
    if with_harmonics:
        audio += np.sin(2 * np.pi * freq * 2 * t) * amplitude * 0.3
        audio += np.sin(2 * np.pi * freq * 3 * t) * amplitude * 0.15
        audio += np.sin(2 * np.pi * freq * 4 * t) * amplitude * 0.08

    # Apply ADSR envelope
    attack = int(0.01 * sr)
    decay = int(0.05 * sr)
    release = int(0.05 * sr)

    envelope = np.ones(len(t))
    if len(envelope) > attack:
        envelope[:attack] = np.linspace(0, 1, attack)
    if len(envelope) > attack + decay:
        envelope[attack:attack + decay] = np.linspace(1, 0.8, decay)
    if len(envelope) > release:
        envelope[-release:] = np.linspace(envelope[-release - 1], 0, release)

    return audio * envelope


def generate_clean_melody(
    notes: List[int],
    note_duration: float = 0.5,
    sr: int = 22050,
) -> Tuple[np.ndarray, List[GroundTruthNote]]:
    """Generate a clean melody with known notes.

    Args:
        notes: List of MIDI pitches
        note_duration: Duration of each note in seconds
        sr: Sample rate

    Returns:
        Tuple of (audio_array, ground_truth_notes)
    """
    total_duration = len(notes) * note_duration
    audio = np.zeros(int(sr * total_duration), dtype=np.float32)
    ground_truth = []

    for i, midi_pitch in enumerate(notes):
        onset = i * note_duration
        offset = onset + note_duration * 0.95  # Small gap between notes

        freq = midi_to_freq(midi_pitch)
        note_audio = generate_sine_note(freq, note_duration * 0.95, sr)

        start_sample = int(onset * sr)
        end_sample = start_sample + len(note_audio)
        audio[start_sample:end_sample] += note_audio

        ground_truth.append(GroundTruthNote(
            pitch=midi_pitch,
            onset=onset,
            offset=offset,
        ))

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8

    return audio, ground_truth


def generate_chord_sequence(
    chords: List[List[int]],
    chord_duration: float = 1.0,
    sr: int = 22050,
) -> Tuple[np.ndarray, List[GroundTruthNote]]:
    """Generate a chord sequence with known notes.

    Args:
        chords: List of chords, each chord is a list of MIDI pitches
        chord_duration: Duration of each chord in seconds
        sr: Sample rate

    Returns:
        Tuple of (audio_array, ground_truth_notes)
    """
    total_duration = len(chords) * chord_duration
    audio = np.zeros(int(sr * total_duration), dtype=np.float32)
    ground_truth = []

    for i, chord_pitches in enumerate(chords):
        onset = i * chord_duration
        offset = onset + chord_duration * 0.95

        for midi_pitch in chord_pitches:
            freq = midi_to_freq(midi_pitch)
            note_audio = generate_sine_note(
                freq, chord_duration * 0.95, sr,
                amplitude=0.6 / len(chord_pitches),  # Scale by chord size
            )

            start_sample = int(onset * sr)
            end_sample = start_sample + len(note_audio)
            audio[start_sample:end_sample] += note_audio

            ground_truth.append(GroundTruthNote(
                pitch=midi_pitch,
                onset=onset,
                offset=offset,
            ))

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8

    return audio, ground_truth


def generate_with_harmonics(
    fundamental: int,
    duration: float = 2.0,
    sr: int = 22050,
) -> Tuple[np.ndarray, List[GroundTruthNote]]:
    """Generate a note with strong harmonics to test harmonic filtering.

    The ground truth only contains the fundamental - the harmonics
    should be filtered out by good transcription.
    """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    freq = midi_to_freq(fundamental)

    # Strong fundamental with prominent harmonics
    audio = np.sin(2 * np.pi * freq * t) * 0.5
    audio += np.sin(2 * np.pi * freq * 2 * t) * 0.4  # Strong 2nd harmonic
    audio += np.sin(2 * np.pi * freq * 3 * t) * 0.3  # Strong 3rd harmonic
    audio += np.sin(2 * np.pi * freq * 4 * t) * 0.2  # 4th harmonic

    # Apply envelope
    attack = int(0.02 * sr)
    release = int(0.1 * sr)
    envelope = np.ones(len(t))
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    audio *= envelope

    ground_truth = [GroundTruthNote(
        pitch=fundamental,
        onset=0.0,
        offset=duration - 0.1,
    )]

    return audio.astype(np.float32), ground_truth


def generate_noisy_melody(
    snr_db: float = 20.0,
    sr: int = 22050,
) -> Tuple[np.ndarray, List[GroundTruthNote]]:
    """Generate a melody with added noise to test robustness."""
    notes = [60, 62, 64, 65, 67]  # C4 to G4
    audio, ground_truth = generate_clean_melody(notes, note_duration=0.5, sr=sr)

    # Add white noise
    noise_power = np.mean(audio ** 2) / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    audio = audio + noise

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8

    return audio, ground_truth


def evaluate_transcription(
    detected_notes: List,
    ground_truth: List[GroundTruthNote],
    pitch_tolerance: int = 1,
    time_tolerance: float = 0.1,
) -> Dict[str, float]:
    """Evaluate transcription quality against ground truth.

    Args:
        detected_notes: List of detected Note objects
        ground_truth: List of ground truth notes
        pitch_tolerance: Allowed pitch deviation in semitones
        time_tolerance: Allowed timing deviation in seconds

    Returns:
        Dictionary with precision, recall, and F1 score
    """
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not detected_notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Match detected notes to ground truth
    matched_gt = set()
    true_positives = 0

    for detected in detected_notes:
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue

            pitch_match = abs(detected.pitch - gt.pitch) <= pitch_tolerance
            onset_match = abs(detected.onset - gt.onset) <= time_tolerance

            if pitch_match and onset_match:
                matched_gt.add(i)
                true_positives += 1
                break

    precision = true_positives / len(detected_notes) if detected_notes else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "detected_count": len(detected_notes),
        "ground_truth_count": len(ground_truth),
    }


def generate_all_fixtures(output_dir: Path, sr: int = 22050) -> Dict[str, List[dict]]:
    """Generate all test fixtures and save to directory.

    Returns mapping of filename to ground truth notes.
    """
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile not installed, cannot generate fixtures")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_map = {}

    # 1. Clean melody - C major scale
    print("Generating clean_melody.wav...")
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    audio, truth = generate_clean_melody(c_major, note_duration=0.5, sr=sr)
    sf.write(str(output_dir / "clean_melody.wav"), audio, sr)
    ground_truth_map["clean_melody.wav"] = [n.to_dict() for n in truth]

    # 2. Chord sequence - I-IV-V-I in C
    print("Generating chord_sequence.wav...")
    chords = [
        [60, 64, 67],  # C major
        [65, 69, 72],  # F major
        [67, 71, 74],  # G major
        [60, 64, 67],  # C major
    ]
    audio, truth = generate_chord_sequence(chords, chord_duration=1.0, sr=sr)
    sf.write(str(output_dir / "chord_sequence.wav"), audio, sr)
    ground_truth_map["chord_sequence.wav"] = [n.to_dict() for n in truth]

    # 3. Harmonics test - single note with strong overtones
    print("Generating harmonics_test.wav...")
    audio, truth = generate_with_harmonics(60, duration=2.0, sr=sr)  # C4
    sf.write(str(output_dir / "harmonics_test.wav"), audio, sr)
    ground_truth_map["harmonics_test.wav"] = [n.to_dict() for n in truth]

    # 4. Noisy melody - test robustness
    print("Generating noisy_melody.wav...")
    audio, truth = generate_noisy_melody(snr_db=20.0, sr=sr)
    sf.write(str(output_dir / "noisy_melody.wav"), audio, sr)
    ground_truth_map["noisy_melody.wav"] = [n.to_dict() for n in truth]

    # 5. Single note A4 for basic pitch test
    print("Generating single_a4.wav...")
    audio, truth = generate_clean_melody([69], note_duration=2.0, sr=sr)
    sf.write(str(output_dir / "single_a4.wav"), audio, sr)
    ground_truth_map["single_a4.wav"] = [n.to_dict() for n in truth]

    # Save ground truth JSON
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, "w") as f:
        json.dump(ground_truth_map, f, indent=2)
    print(f"Ground truth saved to {ground_truth_path}")

    return ground_truth_map


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "audio"
    ground_truth = generate_all_fixtures(fixtures_dir)
    print(f"\nGenerated {len(ground_truth)} test fixtures")
