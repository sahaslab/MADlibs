import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pytest
import librosa
from madlibs.mir.longest_silence import LongestSilence


def test_longest_silence_on_ace_step():
    sil = LongestSilence()
    audio, sr = librosa.load("tests/mir/ace_step_test.wav")
    print(audio.shape)
    output = sil.process_audio(data=(audio, sr))
    assert list(output.keys())[0] == "longest_silence_ms" and len(output.keys()) == 1
    assert output["longest_silence_ms"] == 0


def test_longest_silence_on_all_silence():
    sil = LongestSilence()
    sr = 44100
    length = sr * 5
    audio = np.zeros(length)
    output = sil.process_audio(data=(audio, sr))
    assert output["longest_silence_ms"] == 2520


def test_longest_silence_on_one_chunk_silence():
    sil = LongestSilence()
    sr = 44100
    length = sr * 5
    audio = np.ones(length)

    # Do a little over to compensate for the coarse window
    silence_dur = int(sr * 50 / 1000)
    audio[sr * 3 : sr * 3 + silence_dur] = 0
    output = sil.process_audio(data=(audio, sr))
    assert output["longest_silence_ms"] == 30
