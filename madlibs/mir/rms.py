import numpy as np
import librosa
from madlibs.mir.base import MIRAnalysis


class RMSStd(MIRAnalysis):
    def process_audio(self, data: tuple[np.ndarray, float]) -> dict[str, float]:
        # Calculate RMS for default librosa frames
        audio, _ = data
        rms = librosa.feature.rms(y=audio)
        std = np.std(rms)
        return {"rmsstd": std}
