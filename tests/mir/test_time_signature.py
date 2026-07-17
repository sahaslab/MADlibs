import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pytest
import torch
from madlibs.mir.time_signature import TimeSignature


def test_time_signature():
    t = TimeSignature()

    beats = [1, 20, 80, 100, 101, 120, 180, 200, 201, 220, 280, 300, 301, 320, 380, 400]
    downbeats = [100, 200, 300]

    data = (beats, downbeats)
    output = t.process_audio(data)
    assert "time_signature" in output
    assert output["time_signature"] == 4


def test_time_signature_on_ace_step():
    t = TimeSignature()
    File2Beats = pytest.importorskip("beat_this.inference").File2Beats
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
    assert output["time_signature"] == 4


def test_time_signature_returns_none_for_sparse_beats():
    t = TimeSignature()

    assert t.process_audio(data=([], [])) == {"time_signature": None}
    assert t.process_audio(data=([], [1.0])) == {"time_signature": None}
