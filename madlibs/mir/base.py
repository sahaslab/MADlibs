import numpy as np
from abc import ABC, abstractmethod


class MIRAnalysis(ABC):
    @abstractmethod
    def process_audio(self, data: np.ndarray) -> dict:
        """Process the audio data via given music information retrieval method."""
