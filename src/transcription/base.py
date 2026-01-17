"""Base classes for transcription."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from ..core import Note


class Transcriber(ABC):
    """Abstract base class for audio transcription."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Transcribe audio to notes.

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            List of detected notes
        """
        pass
