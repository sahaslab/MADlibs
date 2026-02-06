from src.mir.RMSStd import RMSStd

import librosa

if __name__ == "__main__":
    # NOTE: this is a test to make sure plotting works
    rms = RMSStd()
    audio_1, sr = librosa.load("tests/mir/ace_step_test.wav")
    audio_2, sr = librosa.load("tests/mir/diffrhythm_test.wav")

    rms.process_audio("audio_1", audio_1, sr=sr)
    rms.process_audio("audio_2", audio_2, sr=sr)

    rms.plot_analysis()