import numpy as np
import librosa
from madlibs.mir.base import MIRAnalysis


class ChromagramSum(MIRAnalysis):
    def process_audio(self, data: tuple[np.ndarray, float]) -> dict[str, np.ndarray]:
        """
        Analyze pitch by calculating chromagram, summing over time, and normalizing

        Args:
            - data: quantized audio numpy array
        Returns:
            - pitch array of size (12,)
        """
        audio, _ = data
        chroma = librosa.feature.chroma_stft(y=audio)
        sum_t = np.sum(chroma, axis=1)
        normalized = sum_t / np.sum(sum_t)
        return {"chromasum": normalized}
