from src.mir.Pipeline import process

import librosa
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # NOTE: this is a test to make sure plotting works
    audio_1, sr = librosa.load("tests/mir/ace_step_test.wav")
    audio_2, sr = librosa.load("tests/mir/diffrhythm_test.wav")
    outputs = []

    outputs.append(process(audio=audio_1, sr=sr))
    outputs.append(process(audio=audio_2, sr=sr))

    rmsstd = [output["rmsstd"] for output in outputs]

    plt.figure()
    plt.hist(rmsstd, bins=30, density=True)
    plt.xlabel("RMS Std")
    plt.ylabel("Density")
    plt.title("Distribution of RMS Std")
    plt.show()
