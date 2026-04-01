import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from madlibs.mir.beat_tracker import BeatThisTracker
from madlibs.mir.chroma import ChromagramSum
from madlibs.mir.pipeline import MIRPipeline
from madlibs.mir.rms import RMSStd
from madlibs.mir.tempo import Tempo


def test_pipeline():
    pipeline = MIRPipeline(
        beat_tracker=BeatThisTracker(checkpoint_path="final0", dbn=False),
        audio_analyses=[RMSStd(), ChromagramSum()],
        beat_analyses=[Tempo()],
        key_analyses=[],
    )
    output = pipeline.process(audio_filename="tests/mir/ace_step_test.wav")
    assert output["metadata"]["audio_filename"] == "tests/mir/ace_step_test.wav"
    assert "rmsstd" in output
    assert "chromasum" in output
    assert "tempo" in output
