from madlibs.mir.base import MIRAnalysis


class Beats(MIRAnalysis):
    def process_audio(self, data) -> dict[str, float | None]:
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
        beats, downbeats = data

        return ({'beats': beats, 'downbeats': downbeats})
