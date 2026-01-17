"""Global constants for Song Analyzer."""

# Pitch names
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Audio processing defaults
DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_FFT = 2048
DEFAULT_N_MELS = 128

# Musical defaults
DEFAULT_TEMPO = 120.0
DEFAULT_TIME_SIGNATURE = (4, 4)
DEFAULT_QUANTIZE_RESOLUTION = 16  # 16th notes

# MIDI ranges
MIDI_MIN = 0
MIDI_MAX = 127
PIANO_MIN = 21  # A0
PIANO_MAX = 108  # C8
