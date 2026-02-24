from src.mir.IMir import IMir

import numpy as np
import librosa


class ChromagramSum(IMir):
    def process_audio(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """
        Analyze pitch by calculating chromagram, summing over time, and normalizing

        Args:
            - data: quantized audio numpy array
        Returns:
            - pitch array of size (12,)
        """
        chroma = librosa.feature.chroma_stft(y=data)
        sum_t = np.sum(chroma, axis=1)
        normalized = sum_t / np.sum(sum_t)
        return {"chromasum": normalized}
