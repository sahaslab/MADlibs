import librosa
from src.mir.Pipeline import process


def test_pipeline():
    # Simple test that pipeline executes without error
    output = process(audio_filename="tests/mir/ace_step_test.wav")
    assert output["metadata"]["audio_filename"] == "tests/mir/ace_step_test.wav"
    # Test that all the measures are in the keys
    assert "rmsstd" in output
    assert "chromasum" in output
    assert "tempo" in output
