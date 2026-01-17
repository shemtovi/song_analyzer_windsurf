"""Source separation using Demucs.

Demucs is a state-of-the-art music source separation model from Meta/Facebook.
Supports multiple models:
- htdemucs: 4 stems (drums, bass, vocals, other)
- htdemucs_6s: 6 stems (drums, bass, vocals, guitar, piano, other)

Reference: https://github.com/facebookresearch/demucs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")


class StemType(Enum):
    """Types of stems that can be separated."""
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    GUITAR = "guitar"      # Available with htdemucs_6s
    PIANO = "piano"        # Available with htdemucs_6s
    OTHER = "other"        # Remaining instruments (synths, strings, etc.)
    
    @classmethod
    def all_stems(cls) -> List["StemType"]:
        """Return all 4-stem types (standard Demucs)."""
        return [cls.DRUMS, cls.BASS, cls.VOCALS, cls.OTHER]
    
    @classmethod
    def all_stems_6(cls) -> List["StemType"]:
        """Return all 6-stem types (htdemucs_6s model)."""
        return [cls.DRUMS, cls.BASS, cls.VOCALS, cls.GUITAR, cls.PIANO, cls.OTHER]
    
    @classmethod
    def melodic_stems(cls) -> List["StemType"]:
        """Return stems that typically contain melodic/pitched content."""
        return [cls.BASS, cls.VOCALS, cls.GUITAR, cls.PIANO, cls.OTHER]
    
    @classmethod
    def harmonic_stems(cls) -> List["StemType"]:
        """Return stems that typically contain harmonic content (chords)."""
        return [cls.GUITAR, cls.PIANO, cls.OTHER]


@dataclass
class StemAudio:
    """Audio data for a single stem."""
    stem_type: StemType
    audio: np.ndarray  # Audio array (mono or stereo)
    sample_rate: int
    confidence: float = 1.0  # Separation confidence/quality estimate
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.audio) / self.sample_rate
    
    @property
    def is_stereo(self) -> bool:
        """Check if audio is stereo."""
        return len(self.audio.shape) > 1 and self.audio.shape[0] == 2
    
    def to_mono(self) -> np.ndarray:
        """Convert to mono if stereo."""
        if self.is_stereo:
            return np.mean(self.audio, axis=0)
        return self.audio
    
    @property
    def rms_energy(self) -> float:
        """Calculate RMS energy of the stem."""
        mono = self.to_mono()
        return float(np.sqrt(np.mean(mono ** 2)))
    
    def is_silent(self, threshold: float = 0.001) -> bool:
        """Check if stem is effectively silent."""
        return self.rms_energy < threshold


@dataclass
class SeparatedStems:
    """Container for all separated stems from an audio file."""
    
    stems: Dict[StemType, StemAudio] = field(default_factory=dict)
    original_sample_rate: int = 44100
    model_name: str = "htdemucs"
    separation_time: float = 0.0  # Time taken to separate in seconds
    
    def __getitem__(self, stem_type: StemType) -> Optional[StemAudio]:
        """Get a stem by type."""
        return self.stems.get(stem_type)
    
    def get_stem(self, stem_type: Union[StemType, str]) -> Optional[StemAudio]:
        """Get a stem by type (accepts string or enum)."""
        if isinstance(stem_type, str):
            stem_type = StemType(stem_type)
        return self.stems.get(stem_type)
    
    def get_melodic_stems(self) -> List[StemAudio]:
        """Get all melodic stems (bass, vocals, other)."""
        return [
            self.stems[st] for st in StemType.melodic_stems()
            if st in self.stems and not self.stems[st].is_silent()
        ]
    
    def get_active_stems(self, threshold: float = 0.001) -> List[StemAudio]:
        """Get all stems that have significant audio content."""
        return [
            stem for stem in self.stems.values()
            if not stem.is_silent(threshold)
        ]
    
    @property
    def has_vocals(self) -> bool:
        """Check if vocals stem has content."""
        vocals = self.stems.get(StemType.VOCALS)
        return vocals is not None and not vocals.is_silent()
    
    @property
    def has_drums(self) -> bool:
        """Check if drums stem has content."""
        drums = self.stems.get(StemType.DRUMS)
        return drums is not None and not drums.is_silent()
    
    def remix(
        self,
        include: Optional[List[StemType]] = None,
        exclude: Optional[List[StemType]] = None,
    ) -> np.ndarray:
        """
        Remix stems back together.
        
        Args:
            include: Only include these stems (default: all)
            exclude: Exclude these stems
            
        Returns:
            Mixed audio array
        """
        if include is None:
            include = list(self.stems.keys())
        
        if exclude:
            include = [st for st in include if st not in exclude]
        
        if not include:
            return np.zeros(1)
        
        # Get first stem to determine shape
        first_stem = self.stems[include[0]]
        mixed = np.zeros_like(first_stem.audio, dtype=np.float32)
        
        for stem_type in include:
            if stem_type in self.stems:
                mixed += self.stems[stem_type].audio
        
        return mixed


class SourceSeparator:
    """
    Audio source separator using Demucs.
    
    Separates audio into stems (drums, bass, vocals, other) for
    per-instrument transcription.
    
    Usage:
        separator = SourceSeparator()
        stems = separator.separate(audio, sr)
        bass_audio = stems.get_stem(StemType.BASS).to_mono()
    """
    
    # Available Demucs models
    MODELS = {
        "htdemucs": "4-stem: drums, bass, vocals, other (recommended)",
        "htdemucs_ft": "4-stem fine-tuned (highest quality, slower)",
        "htdemucs_6s": "6-stem: drums, bass, vocals, guitar, piano, other",
        "mdx_extra": "4-stem MDX-Net (good quality, faster)",
        "mdx": "4-stem MDX-Net base (fastest)",
    }
    
    # Models that output 6 stems
    SIX_STEM_MODELS = {"htdemucs_6s"}
    
    # Stem order for each model type
    STEM_ORDER_4 = ["drums", "bass", "other", "vocals"]
    STEM_ORDER_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
    
    DEFAULT_MODEL = "htdemucs"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        segment_length: float = 10.0,
        overlap: float = 0.25,
        jobs: int = 0,
    ):
        """
        Initialize SourceSeparator.
        
        Args:
            model_name: Demucs model to use (htdemucs, htdemucs_ft, mdx, mdx_extra)
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
            segment_length: Process audio in chunks of this length (seconds)
            overlap: Overlap between segments (0-1)
            jobs: Number of parallel jobs (0=auto)
        """
        self.model_name = model_name
        self.device = device
        self.segment_length = segment_length
        self.overlap = overlap
        self.jobs = jobs
        
        self._model = None
        self._demucs_available = self._check_demucs_available()
    
    def _check_demucs_available(self) -> bool:
        """Check if Demucs is available."""
        try:
            import torch
            import demucs
            return True
        except ImportError:
            return False
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.device != "auto":
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        
        return "cpu"
    
    def _load_model(self):
        """Load Demucs model lazily."""
        if self._model is not None:
            return
        
        if not self._demucs_available:
            raise ImportError(
                "Demucs not available. Install with: pip install demucs torch torchaudio"
            )
        
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import BagOfModels
        
        device = self._get_device()
        
        # Load model
        self._model = get_model(self.model_name)
        
        # Move to device
        if device != "cpu":
            self._model.to(device)
        
        self._model.eval()
        self._device = device
    
    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stems: Optional[List[StemType]] = None,
    ) -> SeparatedStems:
        """
        Separate audio into stems.
        
        Args:
            audio: Audio array (mono or stereo)
            sample_rate: Sample rate of the audio
            stems: Which stems to extract (default: all)
            
        Returns:
            SeparatedStems containing the separated audio
        """
        import time
        start_time = time.time()
        
        # Use fallback if Demucs not available
        if not self._demucs_available:
            warnings.warn(
                "Demucs not available, using frequency-band separation fallback. "
                "Install demucs for better results."
            )
            return self._separate_fallback(audio, sample_rate, stems)
        
        self._load_model()
        
        import torch
        import torchaudio
        from demucs.apply import apply_model
        
        # Convert to tensor - Demucs expects (channels, samples)
        audio = np.asarray(audio)
        if len(audio.shape) == 1:
            # Mono -> stereo by duplicating
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
        elif audio.shape[0] == 2:
            # Already (channels, samples)
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
        elif audio.shape[1] == 2:
            # Shape is (samples, channels) -> transpose to (channels, samples)
            audio_tensor = torch.tensor(audio.T, dtype=torch.float32)
        else:
            # Assume first dim is samples if it's larger
            if audio.shape[0] > audio.shape[1]:
                audio_tensor = torch.tensor(audio.T, dtype=torch.float32)
            else:
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure we have exactly 2 channels
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        elif audio_tensor.shape[0] > 2:
            audio_tensor = audio_tensor[:2, :]
        
        # Resample to model sample rate if needed
        model_sr = self._model.samplerate
        if sample_rate != model_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, model_sr
            )
        
        # Normalize audio (important for Demucs)
        ref = audio_tensor.mean(0)
        audio_tensor = (audio_tensor - ref.mean()) / (ref.std() + 1e-8)
        
        # Add batch dimension: (channels, samples) -> (1, channels, samples)
        audio_tensor = audio_tensor.unsqueeze(0)
        
        # Move to device
        if self._device != "cpu":
            audio_tensor = audio_tensor.to(self._device)
        
        # Apply model with split=True for long audio
        with torch.no_grad():
            sources = apply_model(
                self._model,
                audio_tensor,
                split=True,
                overlap=self.overlap,
                device=self._device,
            )
        
        # Denormalize output
        sources = sources * (ref.std() + 1e-8) + ref.mean()
        
        # Move back to CPU and convert to numpy
        sources = sources.cpu().numpy()
        
        # Resample back to original sample rate if needed
        separated_stems = {}
        stem_names = self._model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']
        
        for i, name in enumerate(stem_names):
            try:
                stem_type = StemType(name)
            except ValueError:
                continue  # Skip unknown stem types
            
            if stems is not None and stem_type not in stems:
                continue
            
            stem_audio = sources[0, i]  # Remove batch dimension
            
            # Resample back if needed
            if sample_rate != model_sr:
                stem_tensor = torch.tensor(stem_audio, dtype=torch.float32)
                stem_tensor = torchaudio.functional.resample(
                    stem_tensor, model_sr, sample_rate
                )
                stem_audio = stem_tensor.numpy()
            
            # Convert to mono for processing
            stem_mono = np.mean(stem_audio, axis=0)
            
            separated_stems[stem_type] = StemAudio(
                stem_type=stem_type,
                audio=stem_mono,
                sample_rate=sample_rate,
                confidence=1.0,
            )
        
        separation_time = time.time() - start_time
        
        return SeparatedStems(
            stems=separated_stems,
            original_sample_rate=sample_rate,
            model_name=self.model_name,
            separation_time=separation_time,
        )
    
    def _separate_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stems: Optional[List[StemType]] = None,
    ) -> SeparatedStems:
        """
        Fallback separation using simple frequency band filtering.
        
        This is a basic approximation when Demucs is not available.
        Results will be significantly worse than neural separation.
        """
        from scipy import signal
        import time
        
        start_time = time.time()
        
        if stems is None:
            stems = StemType.all_stems()
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        
        separated_stems = {}
        
        # Define frequency bands for approximate separation
        # These are rough approximations and won't work nearly as well as Demucs
        
        if StemType.BASS in stems:
            # Bass: 20-250 Hz
            sos = signal.butter(4, 250, btype='low', fs=sample_rate, output='sos')
            bass = signal.sosfilt(sos, audio)
            separated_stems[StemType.BASS] = StemAudio(
                stem_type=StemType.BASS,
                audio=bass.astype(np.float32),
                sample_rate=sample_rate,
                confidence=0.3,  # Low confidence for fallback
            )
        
        if StemType.DRUMS in stems:
            # Drums: use transient detection (high-pass + envelope following)
            sos_hp = signal.butter(4, 100, btype='high', fs=sample_rate, output='sos')
            sos_lp = signal.butter(4, 8000, btype='low', fs=sample_rate, output='sos')
            drums = signal.sosfilt(sos_hp, audio)
            drums = signal.sosfilt(sos_lp, drums)
            # Emphasize transients
            envelope = np.abs(signal.hilbert(drums))
            drums = drums * (envelope / (np.max(envelope) + 1e-8))
            separated_stems[StemType.DRUMS] = StemAudio(
                stem_type=StemType.DRUMS,
                audio=drums.astype(np.float32),
                sample_rate=sample_rate,
                confidence=0.2,
            )
        
        if StemType.VOCALS in stems:
            # Vocals: 300-4000 Hz (mid-range)
            sos = signal.butter(4, [300, 4000], btype='band', fs=sample_rate, output='sos')
            vocals = signal.sosfilt(sos, audio)
            separated_stems[StemType.VOCALS] = StemAudio(
                stem_type=StemType.VOCALS,
                audio=vocals.astype(np.float32),
                sample_rate=sample_rate,
                confidence=0.2,
            )
        
        if StemType.GUITAR in stems:
            # Guitar: mid-high frequency range with harmonic content
            sos = signal.butter(4, [200, 5000], btype='band', fs=sample_rate, output='sos')
            guitar = signal.sosfilt(sos, audio)
            separated_stems[StemType.GUITAR] = StemAudio(
                stem_type=StemType.GUITAR,
                audio=guitar.astype(np.float32),
                sample_rate=sample_rate,
                confidence=0.15,  # Very low - fallback can't isolate guitar well
            )
        
        if StemType.PIANO in stems:
            # Piano: wide frequency range
            sos = signal.butter(4, [100, 6000], btype='band', fs=sample_rate, output='sos')
            piano = signal.sosfilt(sos, audio)
            separated_stems[StemType.PIANO] = StemAudio(
                stem_type=StemType.PIANO,
                audio=piano.astype(np.float32),
                sample_rate=sample_rate,
                confidence=0.15,
            )
        
        if StemType.OTHER in stems:
            # Other: everything else (subtract bass and vocals approximation)
            other = audio.copy()
            if StemType.BASS in separated_stems:
                other = other - separated_stems[StemType.BASS].audio * 0.5
            if StemType.VOCALS in separated_stems:
                other = other - separated_stems[StemType.VOCALS].audio * 0.3
            separated_stems[StemType.OTHER] = StemAudio(
                stem_type=StemType.OTHER,
                audio=other.astype(np.float32),
                sample_rate=sample_rate,
                confidence=0.2,
            )
        
        separation_time = time.time() - start_time
        
        return SeparatedStems(
            stems=separated_stems,
            original_sample_rate=sample_rate,
            model_name="fallback_bandpass",
            separation_time=separation_time,
        )
    
    def separate_file(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        stems: Optional[List[StemType]] = None,
    ) -> Tuple[SeparatedStems, Optional[Path]]:
        """
        Separate an audio file into stems.
        
        Args:
            file_path: Path to audio file
            output_dir: Optional directory to save separated stems
            stems: Which stems to extract
            
        Returns:
            Tuple of (SeparatedStems, output_path or None)
        """
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(str(file_path))
        
        # Transpose if needed (sf returns [samples, channels])
        if len(audio.shape) > 1:
            audio = audio.T
        
        # Separate
        result = self.separate(audio, sr, stems)
        
        # Save if output dir specified
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            stem_name = Path(file_path).stem
            for stem_type, stem_audio in result.stems.items():
                stem_path = output_dir / f"{stem_name}_{stem_type.value}.wav"
                sf.write(
                    str(stem_path),
                    stem_audio.audio,
                    stem_audio.sample_rate,
                )
            output_path = output_dir
        
        return result, output_path
    
    @property
    def is_available(self) -> bool:
        """Check if Demucs separation is available."""
        return self._demucs_available
    
    @property
    def is_6stem(self) -> bool:
        """Check if current model separates guitar and piano."""
        return self.model_name in self.SIX_STEM_MODELS
    
    @property
    def supports_guitar(self) -> bool:
        """Check if current model can separate guitar."""
        return self.is_6stem
    
    @property
    def available_models(self) -> Dict[str, str]:
        """Get available Demucs models."""
        return self.MODELS.copy()
    
    def get_stem_types(self) -> List[StemType]:
        """Get stem types available for current model."""
        if self.is_6stem:
            return StemType.all_stems_6()
        return StemType.all_stems()
