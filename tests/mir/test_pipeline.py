import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from madlibs.mir.chroma import ChromagramSum
from madlibs.mir.gpu_pipeline import GPUPipeline
from madlibs.mir.mir_pipeline import MIRPipeline
from madlibs.mir.rms import RMSStd


def test_mir_pipeline():
    pipeline = MIRPipeline(audio_analyses=[RMSStd(), ChromagramSum()])
    output = pipeline.process(audio_filename="tests/mir/ace_step_test.wav")
    assert output["metadata"]["audio_filename"] == "tests/mir/ace_step_test.wav"
    assert "rmsstd" in output
    assert "chromasum" in output


def test_gpu_pipeline(tmp_path):
    class BeatTrackerStub:
        def track(self, audio_filename):
            assert audio_filename == "tests/mir/ace_step_test.wav"
            return [0.0, 0.5, 1.0], [0.0, 1.0]

    class BeatAnalysisStub:
        def process_audio(self, data):
            beats, downbeats = data
            return {"beat_count": len(beats), "downbeat_count": len(downbeats)}

    class KeyAnalysisStub:
        def process_audio(self, audio, sr):
            return {"sample_rate": sr, "sample_count": len(audio)}

    class InstrumentAnalysisStub:
        def process_audio(self, audio_path):
            return {"instrument_audio_path": audio_path}

    class ClapAnalysisStub:
        def process_audio(self, audio_path, prompt):
            return {"clap_audio_path": audio_path, "prompt_text": prompt}

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("test prompt", encoding="utf-8")

    pipeline = GPUPipeline(
        beat_tracker=BeatTrackerStub(),
        beat_analyses=[BeatAnalysisStub()],
        key_analyses=[KeyAnalysisStub()],
        instrument_analyses=[InstrumentAnalysisStub()],
        clap_analyses=[ClapAnalysisStub()],
    )
    output = pipeline.process(
        audio_filename="tests/mir/ace_step_test.wav",
        prompt_filename=str(prompt_path),
    )
    assert output["metadata"]["audio_filename"] == "tests/mir/ace_step_test.wav"
    assert output["beat_count"] == 3
    assert output["downbeat_count"] == 2
    assert output["sample_rate"] > 0
    assert output["sample_count"] > 0
    assert output["instrument_audio_path"] == "tests/mir/ace_step_test.wav"
    assert output["clap_audio_path"] == "tests/mir/ace_step_test.wav"
    assert output["prompt_text"] == "test prompt"
