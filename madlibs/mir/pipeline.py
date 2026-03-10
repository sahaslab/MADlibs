import dataclasses
import librosa
from madlibs.mir.base import MIRAnalysis
from typing import Any


@dataclasses.dataclass
class MIRPipeline:
    audio_analyses: list[MIRAnalysis]
    beat_analyses: list[MIRAnalysis]
    instrument_analyses: list[MIRAnalysis]
    beat_tracker: Any
    instrument_tagger: Any

    def process(
        self, audio_filename: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        metadata["audio_filename"] = audio_filename

        beats, downbeats = self.beat_tracker.track(audio_filename)
        instruments = self.instrument_tagger.tag(audio_filename)
        audio, sr = librosa.load(audio_filename)

        processed: dict[str, Any] = {"metadata": metadata}
        for analysis in self.audio_analyses:
            processed.update(analysis.process_audio(data=(audio, sr)))
        for analysis in self.beat_analyses:
            processed.update(analysis.process_audio(data=(beats, downbeats)))
        for analysis in self.instrument_analyses:
            processed.update(analysis.process_audio(data=instruments))
        return processed
