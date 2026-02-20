import librosa
from src.mir.Pipeline import process


def test_pipeline():
    # Simple test that pipeline executes without error
    output = process(audio_filename="tests/mir/ace_step_test.wav")
    assert output["metadata"] == []
