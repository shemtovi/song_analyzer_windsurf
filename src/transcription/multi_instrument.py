"""Multi-instrument transcription using source separation.

Combines source separation (Demucs) with per-stem transcription
to handle complex multi-instrument audio recordings.

Pipeline:
1. Separate audio into stems (drums, bass, vocals, other)
2. Classify instrument in each stem
3. Transcribe each stem with appropriate settings
4. Combine results with instrument labels
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

from .base import Transcriber
from .monophonic import MonophonicTranscriber
from .polyphonic import PolyphonicTranscriber
from ..core import Note
from ..separation import (
    SourceSeparator,
    SeparatedStems,
    StemType,
    StemAudio,
    InstrumentClassifier,
    InstrumentCategory,
    InstrumentInfo,
)


@dataclass
class StemTranscription:
    """Transcription result for a single stem."""
    stem_type: StemType
    instrument_info: InstrumentInfo
    notes: List[Note]
    confidence: float = 1.0
    
    @property
    def note_count(self) -> int:
        return len(self.notes)
    
    @property
    def duration(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.offset for n in self.notes) - min(n.onset for n in self.notes)
    
    @property
    def instrument_name(self) -> str:
        """Human-readable instrument name."""
        return self.instrument_info.category.value.title()


@dataclass
class MultiInstrumentTranscription:
    """Complete transcription result with all stems."""
    
    stems: Dict[StemType, StemTranscription] = field(default_factory=dict)
    separation_time: float = 0.0
    transcription_time: float = 0.0
    total_time: float = 0.0
    
    def get_all_notes(self) -> List[Note]:
        """Get all notes from all stems, sorted by onset."""
        all_notes = []
        for stem_trans in self.stems.values():
            all_notes.extend(stem_trans.notes)
        all_notes.sort(key=lambda n: (n.onset, n.pitch))
        return all_notes
    
    def get_melodic_notes(self) -> List[Note]:
        """Get notes from melodic stems only (exclude drums)."""
        notes = []
        for stem_type, stem_trans in self.stems.items():
            if stem_type != StemType.DRUMS and stem_trans.instrument_info.is_melodic:
                notes.extend(stem_trans.notes)
        notes.sort(key=lambda n: (n.onset, n.pitch))
        return notes
    
    def get_stem_notes(self, stem_type: StemType) -> List[Note]:
        """Get notes from a specific stem."""
        if stem_type in self.stems:
            return self.stems[stem_type].notes
        return []
    
    @property
    def total_notes(self) -> int:
        return sum(st.note_count for st in self.stems.values())
    
    @property
    def active_stems(self) -> List[StemType]:
        """Get stems that have detected notes."""
        return [st for st, trans in self.stems.items() if trans.note_count > 0]
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_notes": self.total_notes,
            "active_stems": len(self.active_stems),
            "stems": {
                st.value: {
                    "instrument": trans.instrument_name,
                    "notes": trans.note_count,
                    "confidence": trans.confidence,
                }
                for st, trans in self.stems.items()
            },
            "separation_time": self.separation_time,
            "transcription_time": self.transcription_time,
            "total_time": self.total_time,
        }


class MultiInstrumentTranscriber(Transcriber):
    """
    Transcriber for multi-instrument audio.
    
    Uses source separation to isolate instruments, then transcribes
    each stem independently with appropriate settings.
    
    Features:
    - Demucs-based source separation (4 stems)
    - Instrument classification per stem
    - Adaptive transcription (mono vs poly based on instrument)
    - Combined output with instrument labels
    
    Usage:
        transcriber = MultiInstrumentTranscriber()
        result = transcriber.transcribe_multi(audio, sr)
        
        # Get all notes
        all_notes = result.get_all_notes()
        
        # Get notes by stem
        bass_notes = result.get_stem_notes(StemType.BASS)
    """
    
    def __init__(
        self,
        demucs_model: str = "htdemucs",
        device: str = "auto",
        skip_drums: bool = False,
        min_stem_energy: float = 0.001,
        mono_transcriber_kwargs: Optional[Dict] = None,
        poly_transcriber_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize MultiInstrumentTranscriber.
        
        Args:
            demucs_model: Demucs model to use for separation
            device: Device for inference ('auto', 'cpu', 'cuda')
            skip_drums: Skip drum transcription (drums don't have pitched notes)
            min_stem_energy: Minimum RMS energy to process a stem
            mono_transcriber_kwargs: Kwargs for monophonic transcriber
            poly_transcriber_kwargs: Kwargs for polyphonic transcriber
        """
        self.demucs_model = demucs_model
        self.device = device
        self.skip_drums = skip_drums
        self.min_stem_energy = min_stem_energy
        
        # Initialize components
        self.separator = SourceSeparator(
            model_name=demucs_model,
            device=device,
        )
        self.classifier = InstrumentClassifier()
        
        # Transcribers (lazy initialization)
        self._mono_kwargs = mono_transcriber_kwargs or {}
        self._poly_kwargs = poly_transcriber_kwargs or {}
        self._mono_transcriber = None
        self._poly_transcriber = None
    
    def _get_mono_transcriber(self) -> MonophonicTranscriber:
        """Get or create monophonic transcriber."""
        if self._mono_transcriber is None:
            self._mono_transcriber = MonophonicTranscriber(**self._mono_kwargs)
        return self._mono_transcriber
    
    def _get_poly_transcriber(self) -> PolyphonicTranscriber:
        """Get or create polyphonic transcriber."""
        if self._poly_transcriber is None:
            device = self.device if self.device != "auto" else "cpu"
            self._poly_transcriber = PolyphonicTranscriber(
                device=device,
                **self._poly_kwargs
            )
        return self._poly_transcriber
    
    def transcribe(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Transcribe audio (Transcriber interface).
        
        Returns all notes from all stems combined.
        For more control, use transcribe_multi().
        """
        result = self.transcribe_multi(audio, sr)
        return result.get_all_notes()
    
    def transcribe_multi(
        self,
        audio: np.ndarray,
        sr: int,
        stems_to_process: Optional[List[StemType]] = None,
    ) -> MultiInstrumentTranscription:
        """
        Full multi-instrument transcription.
        
        Args:
            audio: Audio array (mono or stereo)
            sr: Sample rate
            stems_to_process: Which stems to process (default: all)
            
        Returns:
            MultiInstrumentTranscription with per-stem results
        """
        import time
        start_time = time.time()
        
        # Ensure mono for processing
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio
        
        # Step 1: Separate into stems
        separated = self.separator.separate(audio, sr, stems_to_process)
        separation_time = separated.separation_time
        
        # Step 2: Process each stem
        transcription_start = time.time()
        stem_transcriptions = {}
        
        for stem_type, stem_audio in separated.stems.items():
            # Skip if too quiet
            if stem_audio.is_silent(self.min_stem_energy):
                continue
            
            # Skip drums if requested
            if stem_type == StemType.DRUMS and self.skip_drums:
                continue
            
            # Classify instrument
            instrument_info = self.classifier.classify(
                stem_audio.to_mono(),
                stem_audio.sample_rate,
                stem_hint=stem_type.value,
            )
            
            # Transcribe based on instrument type
            notes = self._transcribe_stem(stem_audio, instrument_info)
            
            # Label notes with instrument
            for note in notes:
                note.instrument = instrument_info.category.value
            
            stem_transcriptions[stem_type] = StemTranscription(
                stem_type=stem_type,
                instrument_info=instrument_info,
                notes=notes,
                confidence=stem_audio.confidence * instrument_info.confidence,
            )
        
        transcription_time = time.time() - transcription_start
        total_time = time.time() - start_time
        
        return MultiInstrumentTranscription(
            stems=stem_transcriptions,
            separation_time=separation_time,
            transcription_time=transcription_time,
            total_time=total_time,
        )
    
    def _transcribe_stem(
        self,
        stem_audio: StemAudio,
        instrument_info: InstrumentInfo,
    ) -> List[Note]:
        """
        Transcribe a single stem with appropriate settings.
        """
        audio = stem_audio.to_mono()
        sr = stem_audio.sample_rate
        
        # Skip drums for now (no pitched transcription)
        if not instrument_info.is_melodic:
            return self._transcribe_drums(audio, sr)
        
        # Choose transcriber based on instrument
        if instrument_info.is_monophonic:
            transcriber = self._get_mono_transcriber()
        else:
            transcriber = self._get_poly_transcriber()
        
        # Transcribe
        notes = transcriber.transcribe(audio, sr)
        
        # Filter by instrument pitch range if available
        if instrument_info.pitch_range:
            min_pitch, max_pitch = instrument_info.pitch_range
            notes = [
                n for n in notes
                if min_pitch <= n.pitch <= max_pitch
            ]
        
        return notes
    
    def _transcribe_drums(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> List[Note]:
        """
        Basic drum onset detection (not pitched).
        
        Returns notes with special MIDI mappings for drum sounds.
        This is a simplified version - full drum transcription would
        require a specialized drum transcription model.
        """
        import librosa
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            units='frames',
            hop_length=512,
            backtrack=True,
        )
        
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        # Create notes for drum hits
        # Using General MIDI drum map: 36=kick, 38=snare, 42=hihat
        notes = []
        
        for i, onset in enumerate(onset_times):
            # Simple classification based on spectral content
            frame_start = int(onset * sr)
            frame_end = min(frame_start + int(0.1 * sr), len(audio))
            
            if frame_end <= frame_start:
                continue
            
            segment = audio[frame_start:frame_end]
            
            # Compute spectral centroid for this segment
            centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
            
            # Classify based on frequency content
            if centroid < 200:
                pitch = 36  # Kick drum
            elif centroid < 1000:
                pitch = 38  # Snare
            else:
                pitch = 42  # Hi-hat
            
            # Calculate offset (drums are typically short)
            offset = onset + 0.1
            if i + 1 < len(onset_times):
                offset = min(offset, onset_times[i + 1])
            
            notes.append(Note(
                pitch=pitch,
                onset=onset,
                offset=offset,
                velocity=80,
                instrument="drums",
            ))
        
        return notes
    
    def transcribe_file(
        self,
        file_path: str,
        output_stems: bool = False,
        output_dir: Optional[str] = None,
    ) -> Tuple[MultiInstrumentTranscription, Optional[Dict[str, str]]]:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            output_stems: Save separated stems to disk
            output_dir: Directory for stem output
            
        Returns:
            Tuple of (transcription, stem_paths or None)
        """
        import soundfile as sf
        from pathlib import Path
        
        # Load audio
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio.T  # [channels, samples]
        
        # Transcribe
        result = self.transcribe_multi(audio, sr)
        
        # Save stems if requested
        stem_paths = None
        if output_stems and output_dir:
            stem_paths = {}
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            separated = self.separator.separate(audio, sr)
            stem_name = Path(file_path).stem
            
            for stem_type, stem_audio in separated.stems.items():
                stem_path = output_dir / f"{stem_name}_{stem_type.value}.wav"
                sf.write(str(stem_path), stem_audio.audio, stem_audio.sample_rate)
                stem_paths[stem_type.value] = str(stem_path)
        
        return result, stem_paths
    
    @property
    def is_available(self) -> bool:
        """Check if multi-instrument transcription is available."""
        return self.separator.is_available
