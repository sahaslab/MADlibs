import numpy as np
import librosa
import torch
from beat_this.inference import File2Beats

from src.mir.RMSStd import RMSStd
from src.mir.Tempo import Tempo


def process(audio_filename: str, metadata: dict = {}) -> dict:
    """
    Process all MIR analysis for an audio and store in a k, v store

    Args:
        - audio_filename: the filename for the audio
        - metadata: any attached metadata like song name, model source, etc
    Returns:
        - key, value store with key being type of analysis
            and val is the analysis output
    """
    # Hack fix for beat this torch error
    if not hasattr(torch.amp, "is_autocast_available"):

        def is_autocast_available(device_type):
            return False

        torch.amp.is_autocast_available = is_autocast_available

    # All external library pre processing TODO maybe clean up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file2beats = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = file2beats(audio_filename)
    audio, _ = librosa.load(audio_filename)

    audio_analysis = [RMSStd()]
    beat_analysis = [Tempo()]

    metadata["audio_filename"] = audio_filename
    processed = {"metadata": metadata}

    for analysis in audio_analysis:
        output = analysis.process_audio(data=audio)
        # flatten dict
        for k, v in output.items():
            processed[k] = v
    for analysis in beat_analysis:
        output = analysis.process_audio(data=(beats, downbeats))
        # flatten dict
        for k, v in output.items():
            processed[k] = v

    return processed
