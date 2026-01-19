"""Polyphonic transcription using piano_transcription_inference or CQT-based fallback."""

import numpy as np
import librosa
from typing import List, Optional, Tuple
from pathlib import Path
import tempfile
import soundfile as sf

from .base import Transcriber
from ..core import Note


class PolyphonicTranscriber(Transcriber):
    """
    Polyphonic audio transcriber for detecting multiple simultaneous notes.

    Uses piano_transcription_inference (Onsets and Frames based) when available,
    falls back to CQT-based multi-pitch detection otherwise.
    """

    def __init__(
        self,
        onset_threshold: float = 0.4,
        frame_threshold: float = 0.2,
        min_note_duration: float = 0.05,
        device: str = "cpu",
        min_peak_energy: float = 0.15,
        min_rms_threshold: float = 0.01,
        max_notes_per_segment: int = 8,
    ):
        """
        Initialize PolyphonicTranscriber.

        Args:
            onset_threshold: Threshold for onset detection (0-1, higher = fewer onsets)
            frame_threshold: Threshold for frame-level detection (0-1, higher = fewer notes)
            min_note_duration: Minimum note duration in seconds
            device: Device for inference ('cpu' or 'cuda')
            min_peak_energy: Minimum CQT peak energy (0-1) to consider as note
            min_rms_threshold: Minimum RMS energy to process segment (filters noise)
            max_notes_per_segment: Maximum notes per segment (prevents noise explosion)
        """
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_duration = min_note_duration
        self.device = device
        self.min_peak_energy = min_peak_energy
        self.min_rms_threshold = min_rms_threshold
        self.max_notes_per_segment = max_notes_per_segment
        self._transcriptor = None
        self._use_neural = self._check_neural_available()

    def _check_neural_available(self) -> bool:
        """Check if neural transcription (piano_transcription_inference) is available."""
        try:
            import torch
            from piano_transcription_inference import PianoTranscription
            return True
        except ImportError:
            return False

    def _init_neural_transcriptor(self):
        """Initialize the neural transcriptor lazily."""
        if self._transcriptor is None and self._use_neural:
            try:
                import warnings
                import os
                import sys
                
                # Suppress stdout/stderr during model loading (wget warnings, etc.)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Redirect stderr to suppress wget errors on Windows
                    old_stderr = sys.stderr
                    sys.stderr = open(os.devnull, 'w')
                    try:
                        from piano_transcription_inference import PianoTranscription
                        self._transcriptor = PianoTranscription(device=self.device)
                    finally:
                        sys.stderr.close()
                        sys.stderr = old_stderr
            except Exception as e:
                # Silently fall back to CQT
                self._use_neural = False

    def transcribe(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Transcribe polyphonic audio to notes.

        Args:
            audio: Audio array (mono)
            sr: Sample rate

        Returns:
            List of detected notes with pitch, onset, offset, velocity
        """
        # Try to initialize neural transcriptor (may fall back to CQT)
        if self._use_neural:
            self._init_neural_transcriptor()
        
        # Check again after init attempt (may have fallen back)
        if self._use_neural and self._transcriptor is not None:
            return self._transcribe_neural(audio, sr)
        else:
            return self._transcribe_cqt(audio, sr)

    def _transcribe_neural(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Transcribe using piano_transcription_inference (neural network based).
        
        This provides high-quality polyphonic transcription using the
        Onsets and Frames model architecture.
        """
        self._init_neural_transcriptor()

        # piano_transcription_inference expects 16kHz audio
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # The library works with file paths, so we need to save temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, target_sr)

        try:
            # Transcribe
            transcribed_dict = self._transcriptor.transcribe(
                tmp_path,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
            )

            # Convert to Note objects
            notes = []
            for event in transcribed_dict.get("est_note_events", []):
                onset, offset, pitch, velocity = event
                
                # Filter by minimum duration
                if (offset - onset) < self.min_note_duration:
                    continue

                notes.append(
                    Note(
                        pitch=int(pitch),
                        onset=float(onset),
                        offset=float(offset),
                        velocity=int(velocity),
                    )
                )

            # Sort by onset time
            notes.sort(key=lambda n: (n.onset, n.pitch))
            return notes

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def _transcribe_cqt(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Fallback: CQT-based multi-pitch detection.

        Uses Constant-Q Transform with peak detection to find
        multiple simultaneous pitches. Less accurate than neural
        but works without PyTorch.
        """
        from scipy.signal import find_peaks

        # Parameters - focus on typical musical range (C2 to C7)
        hop_length = 512
        n_bins = 60  # 5 octaves (C2-C7)
        bins_per_octave = 12
        fmin = librosa.note_to_hz("C2")  # Start from C2 (MIDI 36)

        # Compute CQT
        C = np.abs(librosa.cqt(
            audio,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
        ))

        # Convert to dB and normalize
        C_db = librosa.amplitude_to_db(C, ref=np.max)
        C_norm = (C_db - C_db.min()) / (C_db.max() - C_db.min() + 1e-8)

        # Detect onsets using spectral flux (changes in spectrum)
        spectral_flux = np.zeros(C.shape[1])
        for i in range(1, C.shape[1]):
            diff = C_norm[:, i] - C_norm[:, i-1]
            spectral_flux[i] = np.sum(np.maximum(diff, 0))

        # Find peaks in spectral flux (these are onset candidates)
        # Increased threshold: mean + 1.5*std (was mean + std)
        flux_threshold = np.mean(spectral_flux) + 1.5 * np.std(spectral_flux)
        flux_peaks, _ = find_peaks(
            spectral_flux,
            height=flux_threshold,
            distance=int(0.1 * sr / hop_length)
        )

        # Convert to times
        times = librosa.times_like(C, sr=sr, hop_length=hop_length)
        onsets = times[flux_peaks] if len(flux_peaks) > 0 else np.array([])

        # Add start and end points
        duration = len(audio) / sr

        if len(onsets) == 0:
            onsets = np.array([0.0])
        onsets = np.concatenate([[0.0], onsets, [duration]])
        onsets = np.unique(onsets)

        notes = []

        # For each segment between onsets, detect active pitches
        for i in range(len(onsets) - 1):
            onset_time = onsets[i]
            offset_time = onsets[i + 1]

            # Skip very short segments
            if (offset_time - onset_time) < self.min_note_duration:
                continue

            # Check RMS energy of segment - skip quiet/noisy segments
            segment_rms = self._get_segment_rms(audio, sr, onset_time, offset_time)
            if segment_rms < self.min_rms_threshold:
                continue

            # Get frame indices for this segment
            start_frame = int(onset_time * sr / hop_length)
            end_frame = int(offset_time * sr / hop_length)

            if start_frame >= end_frame or start_frame >= C.shape[1]:
                continue

            end_frame = min(end_frame, C.shape[1])

            # Average energy across segment for each pitch bin
            segment_energy = np.mean(C_norm[:, start_frame:end_frame], axis=1)

            # Find peaks in the spectrum
            max_energy = np.max(segment_energy)
            if max_energy < self.min_peak_energy:
                continue  # Skip silent/noisy segments

            # Check spectral flatness - high flatness indicates noise
            spectral_flatness = self._compute_spectral_flatness(segment_energy)
            if spectral_flatness > 0.8:
                continue  # Skip noise-like segments

            # Find peaks with stricter thresholds
            peaks, properties = find_peaks(
                segment_energy,
                height=max(max_energy * 0.4, self.min_peak_energy),  # Increased from 0.3
                distance=2,
                prominence=0.08,  # Increased from 0.05
            )

            if len(peaks) == 0:
                # Fallback: only take bins with very high energy
                active_bins = np.where(segment_energy > max_energy * 0.6)[0]
            else:
                active_bins = peaks

            if len(active_bins) == 0:
                continue

            # Limit notes per segment to prevent noise explosion
            if len(active_bins) > self.max_notes_per_segment:
                # Keep only the strongest peaks
                peak_energies = segment_energy[active_bins]
                top_indices = np.argsort(peak_energies)[-self.max_notes_per_segment:]
                active_bins = active_bins[top_indices]

            # Convert bin indices to MIDI pitches
            midi_base = 36  # C2

            segment_notes = []
            for bin_idx in active_bins:
                midi_pitch = midi_base + bin_idx

                if midi_pitch < 21 or midi_pitch > 108:
                    continue

                # Only include notes with sufficient energy
                note_energy = segment_energy[bin_idx]
                if note_energy < self.min_peak_energy:
                    continue

                velocity = int(min(127, max(20, note_energy * 127)))

                segment_notes.append(
                    Note(
                        pitch=midi_pitch,
                        onset=onset_time,
                        offset=offset_time,
                        velocity=velocity,
                    )
                )

            notes.extend(segment_notes)

        notes = self._merge_duplicate_notes(notes)
        notes.sort(key=lambda n: (n.onset, n.pitch))

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

    def _compute_spectral_flatness(self, spectrum: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy).

        Returns value between 0 (tonal) and 1 (noise-like).
        High values indicate noise rather than musical content.
        """
        # Avoid log(0)
        spectrum = np.maximum(spectrum, 1e-10)

        # Geometric mean / Arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)

        if arithmetic_mean == 0:
            return 1.0

        flatness = geometric_mean / arithmetic_mean
        return float(np.clip(flatness, 0.0, 1.0))

    def _merge_duplicate_notes(self, notes: List[Note]) -> List[Note]:
        """Merge notes with same pitch and overlapping times."""
        if not notes:
            return notes

        pitch_groups = {}
        for note in notes:
            if note.pitch not in pitch_groups:
                pitch_groups[note.pitch] = []
            pitch_groups[note.pitch].append(note)

        merged = []
        for pitch, group in pitch_groups.items():
            group.sort(key=lambda n: n.onset)
            
            current = group[0]
            for next_note in group[1:]:
                if next_note.onset <= current.offset + 0.05:
                    current = Note(
                        pitch=current.pitch,
                        onset=current.onset,
                        offset=max(current.offset, next_note.offset),
                        velocity=max(current.velocity, next_note.velocity),
                    )
                else:
                    merged.append(current)
                    current = next_note
            merged.append(current)

        return merged

    @property
    def is_neural(self) -> bool:
        """Return whether neural transcription is being used."""
        return self._use_neural
