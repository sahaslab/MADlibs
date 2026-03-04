import matplotlib.pyplot as plt

from madlibs.mir.beat_tracker import BeatThisTracker
from madlibs.mir.chroma import ChromagramSum
from madlibs.mir.pipeline import MIRPipeline
from madlibs.mir.rms import RMSStd
from madlibs.mir.tempo import Tempo


if __name__ == "__main__":
    pipeline = MIRPipeline(
        beat_tracker=BeatThisTracker(checkpoint_path="final0", dbn=False),
        audio_analyses=[RMSStd(), ChromagramSum()],
        beat_analyses=[Tempo()],
    )

    outputs = [
        pipeline.process("tests/mir/ace_step_test.wav"),
        pipeline.process("tests/mir/diffrhythm_test.wav"),
    ]
    rmsstd = [output["rmsstd"] for output in outputs]
    plt.figure()
    plt.hist(rmsstd, bins=30, density=True)
    plt.xlabel("RMS Std")
    plt.ylabel("Density")
    plt.title("Distribution of RMS Std")
    plt.show()
