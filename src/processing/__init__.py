"""Processing layer - Note-level post-processing.

This layer refines transcribed notes:
- Quantization (snap to grid)
- Note cleanup (merge, filter, harmonic removal)
- Outlier filtering
- Temporal smoothing
- Velocity normalization
"""

from .quantize import Quantizer
from .cleanup import NoteCleanup, CleanupConfig, CleanupStats

__all__ = [
    "Quantizer",
    "NoteCleanup",
    "CleanupConfig",
    "CleanupStats",
]
