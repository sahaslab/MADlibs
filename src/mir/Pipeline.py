import numpy as np

from src.mir.RMSStd import RMSStd


def process(audio: np.ndarray, sr: float, metadata: list = []) -> dict:
    """
    Process all MIR analysis for an audio and store in a k, v store

    Args:
        - audio: quantized audio
        - sr: sample rate
        - metadata: any attached metadata like song name, model source, etc
    Returns:
        - key, value store with key being type of analysis
            and val is the analysis output
    """
    audio_analysis = [RMSStd()]

    # TODO add other types of analysis i.e. spectrogram analysis, symbolic analysis

    processed = {"metadata": metadata}

    for analysis in audio_analysis:
        output = analysis.process_audio(data=audio, sr=sr)
        # flatten dict
        for k, v in output.items():
            processed[k] = v

    return processed
