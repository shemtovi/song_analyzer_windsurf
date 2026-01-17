"""Processing layer - Note-level post-processing.

This layer refines transcribed notes:
- Quantization (snap to grid)
- Note cleanup (merge, filter)
- Velocity normalization
"""

from .quantize import Quantizer
from .cleanup import NoteCleanup

__all__ = [
    "Quantizer",
    "NoteCleanup",
]
