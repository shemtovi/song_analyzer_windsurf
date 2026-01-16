"""Polyphonic transcription using piano_transcription_inference or CQT-based fallback."""

import numpy as np
import librosa
from typing import List, Optional, Tuple
from pathlib import Path
import tempfile
import soundfile as sf

from .base import Note, Transcriber


class PolyphonicTranscriber(Transcriber):
    """
    Polyphonic audio transcriber for detecting multiple simultaneous notes.
    
    Uses piano_transcription_inference (Onsets and Frames based) when available,
    falls back to CQT-based multi-pitch detection otherwise.
    """

    def __init__(
        self,
        onset_threshold: float = 0.3,
        frame_threshold: float = 0.1,
        min_note_duration: float = 0.05,
        device: str = "cpu",
    ):
        """
        Initialize PolyphonicTranscriber.

        Args:
            onset_threshold: Threshold for onset detection (0-1)
            frame_threshold: Threshold for frame-level detection (0-1)
            min_note_duration: Minimum note duration in seconds
            device: Device for inference ('cpu' or 'cuda')
        """
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_duration = min_note_duration
        self.device = device
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
                from piano_transcription_inference import PianoTranscription
                self._transcriptor = PianoTranscription(device=self.device)
            except Exception as e:
                print(f"  [Warning] Failed to load neural model: {e}")
                print("  [Warning] Falling back to CQT-based transcription")
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
        # This works better for synthetic audio where amplitude may not change much
        # but the frequencies do change
        
        # Compute frame-by-frame spectral difference
        spectral_flux = np.zeros(C.shape[1])
        for i in range(1, C.shape[1]):
            # Compute difference between consecutive frames
            diff = C_norm[:, i] - C_norm[:, i-1]
            # Only count increases (new notes appearing)
            spectral_flux[i] = np.sum(np.maximum(diff, 0))
        
        # Find peaks in spectral flux (these are onset candidates)
        from scipy.signal import find_peaks as fp
        flux_peaks, _ = fp(spectral_flux, height=np.mean(spectral_flux) + np.std(spectral_flux), distance=int(0.1 * sr / hop_length))
        
        # Convert to times
        times = librosa.times_like(C, sr=sr, hop_length=hop_length)
        onsets = times[flux_peaks] if len(flux_peaks) > 0 else np.array([])

        # Add start and end points
        times = librosa.times_like(C, sr=sr, hop_length=hop_length)
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

            # Get frame indices for this segment
            start_frame = int(onset_time * sr / hop_length)
            end_frame = int(offset_time * sr / hop_length)
            
            if start_frame >= end_frame or start_frame >= C.shape[1]:
                continue

            end_frame = min(end_frame, C.shape[1])

            # Average energy across segment for each pitch bin
            segment_energy = np.mean(C_norm[:, start_frame:end_frame], axis=1)

            # Find peaks in the spectrum (not just threshold)
            # This helps avoid harmonics - we want local maxima
            max_energy = np.max(segment_energy)
            if max_energy < 0.1:
                continue  # Skip silent segments
                
            # Use relative threshold based on segment max
            peaks, properties = find_peaks(
                segment_energy, 
                height=max_energy * 0.3,  # 30% of max energy in segment
                distance=2,               # At least 2 semitones apart (reduces harmonics)
                prominence=0.05,          # Peak must stand out slightly
            )
            
            # If no peaks found with strict criteria, try simpler threshold
            if len(peaks) == 0:
                active_bins = np.where(segment_energy > max_energy * 0.5)[0]
            else:
                active_bins = peaks

            # Group nearby bins (within 1 semitone)
            if len(active_bins) == 0:
                continue

            # Convert bin indices to MIDI pitches
            # CQT bin 0 = fmin = C2 = MIDI 36
            midi_base = 36

            for bin_idx in active_bins:
                midi_pitch = midi_base + bin_idx
                
                # Skip if pitch is outside piano range
                if midi_pitch < 21 or midi_pitch > 108:
                    continue

                # Estimate velocity from energy
                velocity = int(min(127, max(1, segment_energy[bin_idx] * 127)))

                # Check minimum duration
                duration = offset_time - onset_time
                if duration < self.min_note_duration:
                    continue

                notes.append(
                    Note(
                        pitch=midi_pitch,
                        onset=onset_time,
                        offset=offset_time,
                        velocity=velocity,
                    )
                )

        # Remove duplicates and sort
        notes = self._merge_duplicate_notes(notes)
        notes.sort(key=lambda n: (n.onset, n.pitch))
        
        return notes

    def _merge_duplicate_notes(self, notes: List[Note]) -> List[Note]:
        """Merge notes with same pitch and overlapping times."""
        if not notes:
            return notes

        # Group by pitch
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
                # If notes overlap or are adjacent, merge them
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
