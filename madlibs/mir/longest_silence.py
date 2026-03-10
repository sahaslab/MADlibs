import numpy as np
import librosa
from madlibs.mir.base import MIRAnalysis


class LongestSilence(MIRAnalysis):
    def process_audio(self, data: tuple[np.ndarray, float]) -> dict[str, float]:
        """
        Find the longest continuous stretch of silence in the middle of the song
        Iterate over 30 ms chunks
        Silence is when RMS of that chunk is below some threshold
        Defining 1e-3 as the silence threshold

        NOTE: doing the middle part of the song (1/4 - 3/4) removes the natural silence variability at the end and beginning of the track

        NOTE: we're iterating over 30 ms chunks so there might be some slight overlap or underlap (<30ms) than the real silence

        Returns:
            The longest silence duration in ms
        """
        SILENCE_THRESHOLD = 1e-3

        audio, sr = data
        # 30 ms chunks
        chunk_ms = 30
        chunk_size = int(sr * chunk_ms / 1000)

        longest_silence_ms = 0
        current_silence_ms = 0

        song_len = audio.shape[0]
        start = int(song_len / 4)
        end = int(3 * song_len / 4)

        for chunk_start in range(start, end, chunk_size):
            chunk = audio[chunk_start : chunk_start + chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            if rms > SILENCE_THRESHOLD:
                # not silent
                longest_silence_ms = max(longest_silence_ms, current_silence_ms)
                current_silence_ms = 0
            else:
                current_silence_ms += chunk_ms

            # Handle case where longest silence is ongoing as of end of song
            longest_silence_ms = max(longest_silence_ms, current_silence_ms)

        return {"longest_silence_ms": longest_silence_ms}
