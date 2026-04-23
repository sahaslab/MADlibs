import dataclasses
from typing import Any

import librosa

from madlibs.mir.base import MIRAnalysis


@dataclasses.dataclass
class BeatPipeline:
    beat_tracker: Any | None = None
    beat_analyses: list[MIRAnalysis] = dataclasses.field(default_factory=list)

    def process(
        self,
        audio_filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        metadata["audio_filename"] = audio_filename

        processed: dict[str, Any] = {"metadata": metadata}

        if self.beat_analyses:
            if self.beat_tracker is None:
                raise ValueError("beat_tracker is required when beat_analyses are set")
            beats, downbeats = self.beat_tracker.track(audio_filename)
            for analysis in self.beat_analyses:
                processed.update(analysis.process_audio(data=(beats, downbeats)))
            

        return processed
