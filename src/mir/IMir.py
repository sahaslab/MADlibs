from abc import ABC, abstractmethod
import numpy as np

class IMir(ABC):
    @abstractmethod
    def process_audio(self, data: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """
        Process the audio data via given music information retrieval method

        Args:
            - data: the quantized audio, spectrogram, symbolic repr, etc

        Returns:
            - key value store with value: processed data
        """