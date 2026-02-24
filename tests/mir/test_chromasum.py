from src.mir.ChromagramSum import ChromagramSum

import librosa
import pytest
import numpy as np


def test_chromagram_sum():
    chroma = ChromagramSum()
    audio, sr = librosa.load("tests/mir/ace_step_test.wav")
    output = chroma.process_audio(data=audio)

    assert "chromasum" in output
    assert np.sum(output["chromasum"]) == pytest.approx(1, rel=1e-3)
