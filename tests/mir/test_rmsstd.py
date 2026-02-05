from src.mir.RMSStd import RMSStd

import pytest
import librosa

def test_rmsstd_on_ace_step():
    rms = RMSStd()
    audio, sr = librosa.load("tests/mir/ace_step_test.wav")
    rms.process_audio("test_audio", audio, sr=sr)
    assert rms.get_single_analysis("test_audio") == pytest.approx(0.07127465, rel=1e-3)

    assert type(rms.get_analysis()) == dict
    assert "test_audio" in rms.get_analysis()