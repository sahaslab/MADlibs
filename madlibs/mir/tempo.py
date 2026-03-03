import numpy as np
from madlibs.mir.base import MIRAnalysis


class Tempo(MIRAnalysis):
    def process_audio(self, data) -> dict[str, float]:
        """
        Tempo is kind of arbitrary. Calculate how many downbeats happen in a minute.

        Let d be the downbeat time
        Iterate from i= 0 to i=len(d)-2
        find diff = d[i + 1]-d[i]
        Approx dbpm is 60/diff

        Args:
            - data: tuple of (beats, downbeats) arrays
        Return:
            - mean of all the approx dbpm (would median also be good?)
        """
        _, downbeats = data

        framewise_dbpm = []
        for i in range(len(downbeats) - 1):
            downbeat_delta = downbeats[i + 1] - downbeats[i]
            dbpm = 60 / (downbeat_delta + 1e-8)
            framewise_dbpm.append(dbpm)

        # Return mean
        return {"tempo": sum(framewise_dbpm) / len(framewise_dbpm)}
