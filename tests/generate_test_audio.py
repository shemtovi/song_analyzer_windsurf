"""Generate sample WAV files for testing."""

import numpy as np
import os
from scipy.io import wavfile

# Ensure examples directory exists
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
os.makedirs(EXAMPLES_DIR, exist_ok=True)


def generate_sine_wave(freq: float, duration: float, sr: int = 22050) -> np.ndarray:
    """Generate a sine wave at given frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def generate_note_sequence(
    frequencies: list, durations: list, sr: int = 22050
) -> np.ndarray:
    """Generate a sequence of notes."""
    audio = []
    for freq, dur in zip(frequencies, durations):
        note = generate_sine_wave(freq, dur, sr)
        # Apply simple envelope to avoid clicks
        envelope = np.ones_like(note)
        attack = int(0.01 * sr)
        release = int(0.01 * sr)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        audio.append(note * envelope)
    return np.concatenate(audio)


def generate_chord(frequencies: list, duration: float, sr: int = 22050) -> np.ndarray:
    """Generate a chord by summing multiple sine waves at different frequencies."""
    voices = [generate_sine_wave(f, duration, sr) for f in frequencies]
    chord = np.sum(voices, axis=0)
    
    # Apply envelope to create clear attack/release transients
    envelope = np.ones_like(chord)
    attack = int(0.02 * sr)   # 20ms attack
    release = int(0.02 * sr)  # 20ms release
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    chord = chord * envelope
    
    # Normalize to avoid clipping
    max_abs = np.max(np.abs(chord)) or 1.0
    chord = chord / max_abs
    return chord.astype(np.float32)


def generate_chord_progression(
    chords: list[list[float]], durations: list[float], sr: int = 22050
) -> np.ndarray:
    """Generate a sequence of chords (polyphonic)."""
    audio = []
    for freqs, dur in zip(chords, durations):
        chord = generate_chord(freqs, dur, sr)
        audio.append(chord)
    return np.concatenate(audio)


def save_wav(filename: str, audio: np.ndarray, sr: int = 22050):
    """Save audio as WAV file."""
    # Normalize to 16-bit range
    audio_16bit = (audio * 32767).astype(np.int16)
    filepath = os.path.join(EXAMPLES_DIR, filename)
    wavfile.write(filepath, sr, audio_16bit)
    print(f"Created: {filepath}")
    return filepath


def main():
    sr = 22050

    # 1. Single A4 note (440 Hz) - 2 seconds
    print("Generating single_a4.wav...")
    a4 = generate_sine_wave(440.0, 2.0, sr)
    save_wav("single_a4.wav", a4, sr)

    # 2. C Major scale (C4, D4, E4, F4, G4, A4, B4, C5)
    print("Generating c_major_scale.wav...")
    scale_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    scale_durs = [0.5] * 8  # 0.5 seconds each
    scale = generate_note_sequence(scale_freqs, scale_durs, sr)
    save_wav("c_major_scale.wav", scale, sr)

    # 3. Simple melody (Twinkle Twinkle pattern: C C G G A A G)
    print("Generating simple_melody.wav...")
    melody_freqs = [261.63, 261.63, 392.00, 392.00, 440.00, 440.00, 392.00]
    melody_durs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8]
    melody = generate_note_sequence(melody_freqs, melody_durs, sr)
    save_wav("simple_melody.wav", melody, sr)

    # 4. Short test tone (middle C, 1 second)
    print("Generating test_c4.wav...")
    c4 = generate_sine_wave(261.63, 1.0, sr)
    save_wav("test_c4.wav", c4, sr)

    # 6. Polyphonic: G chord -> F chord -> A chord
    # G major: G3 (196.00), B3 (246.94), D4 (293.66)
    # F major: F3 (174.61), A3 (220.00), C4 (261.63)
    # A major: A3 (220.00), C#4 (277.18), E4 (329.63)
    print("Generating chords_g_f_a.wav (polyphonic chords)...")
    chords = [
        [196.00, 246.94, 293.66],   # G major
        [174.61, 220.00, 261.63],   # F major
        [220.00, 277.18, 329.63],   # A major
    ]
    chord_durs = [1.0, 1.0, 1.0]
    chord_prog = generate_chord_progression(chords, chord_durs, sr)
    save_wav("chords_g_f_a.wav", chord_prog, sr)

    # 7. Polyphonic: G->F->A chords with a simple melody on top
    print("Generating chords_g_f_a_with_melody.wav...")

    # Reuse the same chord progression
    chords_bg = chord_prog

    # Simple melody: G4 -> A4 -> B4 over the three chords
    melody_freqs = [392.00, 440.00, 493.88]
    melody_durs = chord_durs
    melody = generate_note_sequence(melody_freqs, melody_durs, sr)

    # Mix chords (background) and melody (foreground) with different gains
    min_len = min(len(chords_bg), len(melody))
    mix = chords_bg[:min_len] * 0.6 + melody[:min_len] * 0.8
    # Normalize final mix
    max_abs = np.max(np.abs(mix)) or 1.0
    mix = (mix / max_abs).astype(np.float32)
    save_wav("chords_g_f_a_with_melody.wav", mix, sr)

    # 5. Silence (for edge case testing)
    print("Generating silence.wav...")
    silence = np.zeros(sr * 2, dtype=np.float32)
    save_wav("silence.wav", silence, sr)

    print("\nAll test audio files generated!")


if __name__ == "__main__":
    main()
