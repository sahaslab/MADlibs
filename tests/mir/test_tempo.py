import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pytest
import librosa
import torch
from beat_this.inference import File2Beats
from madlibs.mir.tempo import Tempo


def test_tempo_on_ace_step():
    t = Tempo()
    # Hack fix for beat this torch error
    if not hasattr(torch.amp, "is_autocast_available"):

        def is_autocast_available(device_type):
            return False

        torch.amp.is_autocast_available = is_autocast_available

    # All external library pre processing TODO maybe clean up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file2beats = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = file2beats("tests/mir/ace_step_test.wav")

    output = t.process_audio(data=(beats, downbeats))
    assert list(output.keys())[0] == "tempo" and len(output.keys()) == 1
    assert output["tempo"] == pytest.approx(60, rel=1e-2)
