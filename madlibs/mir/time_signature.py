import numpy as np
from madlibs.mir.base import MIRAnalysis


class TimeSignature(MIRAnalysis):
    def process_audio(self, data) -> dict[str, float | None]:
        """
        Time signature is hard to do, we can do a best effort by taking the median count of beats between each downbeat (+ 1)

        NOTE: this is the numerator for time signature

        Args:
            - data: tuple of (beats, downbeats) arrays
        Return:
            - median of the number of beats between two downbeats
        """
        beats, downbeats = data

        if len(beats) == 0 or len(downbeats) == 0:
            return {"time_signature": 0}

        beat_counts = []
        beat_idx = 0

        for down in downbeats:
            count = 0
            while beat_idx < len(beats) and beats[beat_idx] < down:
                count += 1
                beat_idx += 1

            beat_counts.append(count)

        beat_counts.sort()
        return {"time_signature": beat_counts[len(beat_counts) // 2]}
