"""Monophonic transcription using CREPE and onset detection."""

import numpy as np
import librosa
from typing import List, Optional, Tuple

from .base import Transcriber
from ..core import Note


class MonophonicTranscriber(Transcriber):
    """Transcribes monophonic audio using CREPE pitch detection."""

    def __init__(
        self,
        hop_length: int = 512,
        onset_threshold: float = 0.4,
        min_note_duration: float = 0.05,
        pitch_confidence_threshold: float = 0.65,
        crepe_model: str = "tiny",
        min_rms_threshold: float = 0.01,
    ):
        """
        Initialize MonophonicTranscriber.

        Args:
            hop_length: Samples between analysis frames
            onset_threshold: Threshold for onset detection (0.3-0.5, higher = fewer onsets)
            min_note_duration: Minimum note duration in seconds
            pitch_confidence_threshold: Minimum CREPE/pYIN confidence (0.5-0.8)
            crepe_model: CREPE model size ('tiny', 'small', 'medium', 'large', 'full')
            min_rms_threshold: Minimum RMS energy to consider a segment (filters noise)
        """
        self.hop_length = hop_length
        self.onset_threshold = onset_threshold
        self.min_note_duration = min_note_duration
        self.pitch_confidence_threshold = pitch_confidence_threshold
        self.crepe_model = crepe_model
        self.min_rms_threshold = min_rms_threshold

    def transcribe(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Transcribe monophonic audio to notes.

        Args:
            audio: Audio array (mono)
            sr: Sample rate

        Returns:
            List of detected notes
        """
        # Detect pitch using CREPE
        times, frequencies, confidences = self._detect_pitch(audio, sr)

        # Detect onsets
        onset_times = self._detect_onsets(audio, sr)

        # Segment into notes
        notes = self._segment_notes(
            times, frequencies, confidences, onset_times, audio, sr
        )

        return notes

    def _detect_pitch(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect pitch using CREPE.

        Returns:
            Tuple of (times, frequencies, confidences)
        """
        try:
            import crepe

            # CREPE expects 16kHz audio
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio

            time, frequency, confidence, _ = crepe.predict(
                audio_16k,
                sr=16000,
                model_capacity=self.crepe_model,
                viterbi=True,  # Use Viterbi smoothing
                step_size=10,  # 10ms steps
            )

            return time, frequency, confidence

        except ImportError:
            # Fallback to pYIN if CREPE not available
            return self._detect_pitch_pyin(audio, sr)

    def _detect_pitch_pyin(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback pitch detection using pYIN."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=self.hop_length,
        )

        times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=self.hop_length
        )

        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)

        return times, f0, voiced_probs

    def _detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect note onsets using spectral flux.

        Returns:
            Array of onset times in seconds
        """
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=sr, hop_length=self.hop_length
        )

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=True,
            delta=self.onset_threshold,
        )

        onset_times = librosa.frames_to_time(
            onset_frames, sr=sr, hop_length=self.hop_length
        )

        return onset_times

    def _segment_notes(
        self,
        times: np.ndarray,
        frequencies: np.ndarray,
        confidences: np.ndarray,
        onset_times: np.ndarray,
        audio: np.ndarray,
        sr: int,
    ) -> List[Note]:
        """
        Segment pitch contour into discrete notes at onset boundaries.

        Returns:
            List of Note objects
        """
        notes = []
        duration = len(audio) / sr

        # Add end of audio as final boundary
        boundaries = np.concatenate([onset_times, [duration]])

        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]

            # Skip very short segments
            if end_time - start_time < self.min_note_duration:
                continue

            # Check RMS energy - skip quiet/noisy segments
            rms = self._get_segment_rms(audio, sr, start_time, end_time)
            if rms < self.min_rms_threshold:
                continue

            # Get pitch values in this segment
            mask = (times >= start_time) & (times < end_time)
            segment_freqs = frequencies[mask]
            segment_confs = confidences[mask]

            if len(segment_freqs) == 0:
                continue

            # Filter by confidence
            confident_mask = segment_confs >= self.pitch_confidence_threshold
            if not np.any(confident_mask):
                continue

            confident_freqs = segment_freqs[confident_mask]

            # Use median frequency for the note (robust to outliers)
            valid_freqs = confident_freqs[confident_freqs > 0]
            if len(valid_freqs) == 0:
                continue

            median_freq = np.median(valid_freqs)

            if median_freq <= 0 or np.isnan(median_freq):
                continue

            # Convert to MIDI pitch
            midi_pitch = Note.freq_to_midi(median_freq)

            # Clamp to valid MIDI range
            midi_pitch = max(0, min(127, midi_pitch))

            # Estimate velocity from RMS energy
            velocity = self._rms_to_velocity(rms)

            notes.append(
                Note(
                    pitch=midi_pitch,
                    onset=start_time,
                    offset=end_time,
                    velocity=velocity,
                )
            )

        return notes

    def _get_segment_rms(
        self,
        audio: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float,
    ) -> float:
        """Calculate RMS energy for an audio segment."""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            return 0.0

        return float(np.sqrt(np.mean(segment**2)))

    def _rms_to_velocity(self, rms: float) -> int:
        """Convert RMS energy to MIDI velocity (0-127)."""
        # Map RMS to velocity range [20, 127]
        # Assuming normalized audio, RMS typically 0.01-0.5
        velocity = int(np.clip(rms * 200, 20, 127))
        return velocity
