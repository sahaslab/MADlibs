import numpy as np
from madlibs.mir.base import MIRAnalysis

class TimeSignature(MIRAnalysis):
    def process_audio(self, data) -> dict[str, float]:
        """
        Time signature is hard to do, we can do a best effort by taking the median count of beats between each downbeat (+ 1)

        Args:
            - data: tuple of (beats, downbeats) arrays
        Return:
            - median of the number of beats between two downbeats
        """
        beats, downbeats = data

        beat_counts = []
        beat_idx = 0

        for down in downbeats:
            count = 0
            while beats[beat_idx] < down:
                count += 1
                beat_idx += 1
            
            beat_counts.append(count)

        # Return median
        beat_counts.sort()
        return {"time_signature": beat_counts[len(beat_counts) // 2]}
