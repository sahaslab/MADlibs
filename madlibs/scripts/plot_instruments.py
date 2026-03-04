from madlibs.mir.instrument_tagger import InstrumentTagger
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    tagger = InstrumentTagger(
        model_id="nvidia/music-flamingo-hf",
        cache_dir="/dartfs/rc/lab/S/SinghN/noah/.cache/huggingface/hub",
    )
    instruments = []
    files = ["tests/mir/ace_step_test.wav", "tests/mir/diffrhythm_test.wav"]

    for f in files:
        tagged = tagger.tag(f)
        instruments = instruments + tagged
    s = pd.Series(instruments)
    s.value_counts().plot(kind="bar")

    # Add title and labels
    plt.title("Frequency of Instruments")
    plt.xlabel("Instrument")
    plt.ylabel("Songs present")

    # Display the plot
    plt.savefig("tests/mir/instruments.png")
