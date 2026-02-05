from src.mir.IMir import IMir

import matplotlib.pyplot as plt
import numpy as np
import librosa

class RMSStd(IMir):
    def __init__(self):
        self.data = dict()

    def process_audio(self, id: str, audio: np.ndarray, sr: int = 44100) -> None:
        # Calculate RMS for default librosa frames
        rms = librosa.feature.rms(y=audio)
        std = np.std(rms)
        self.data[id] = std

    def plot_analysis(self) -> None:
        # Plot all the RMS std over all ids using bins
        rms_data = self.data.values()
        
        plt.figure()
        plt.hist(rms_data, bins=30, density=True)
        plt.xlabel("RMS Std")
        plt.ylabel("Density")
        plt.title("Distribution of RMS Std")
        plt.show()

    def get_single_analysis(self, id):
        return self.data[id]
    
    def get_analysis(self):
        return self.data