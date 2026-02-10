from src.mir.RMSStd import RMSStd

import pytest
import librosa

def test_rmsstd_on_ace_step():
    rms = RMSStd()
    audio, sr = librosa.load("tests/mir/ace_step_test.wav")
    output = rms.process_audio(data=audio, sr=sr)
    assert list(output.keys())[0] == "rmsstd" and len(output.keys()) == 1
    assert output["rmsstd"] == pytest.approx(0.07127465, rel=1e-3)