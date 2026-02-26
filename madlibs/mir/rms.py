import numpy as np
import librosa
from madlibs.mir.base import MIRAnalysis


class RMSStd(MIRAnalysis):
    def process_audio(self, data: np.ndarray) -> dict[str, float]:
        # Calculate RMS for default librosa frames
        rms = librosa.feature.rms(y=data)
        std = np.std(rms)
        return {"rmsstd": std}
