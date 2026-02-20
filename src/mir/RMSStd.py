from src.mir.IMir import IMir

import numpy as np
import librosa


class RMSStd(IMir):
    def process_audio(self, data: np.ndarray) -> dict[str, float]:
        # Calculate RMS for default librosa frames
        rms = librosa.feature.rms(y=data)
        std = np.std(rms)
        return {"rmsstd": std}
