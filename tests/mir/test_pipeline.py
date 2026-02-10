import librosa
from src.mir.Pipeline import process

def test_pipeline():
    # Simple test that pipeline executes without error
    audio, sr = librosa.load("tests/mir/ace_step_test.wav")
    output = process(audio=audio, sr=sr)
    assert output["metadata"] == []