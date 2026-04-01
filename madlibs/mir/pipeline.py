import dataclasses
import librosa
from madlibs.mir.base import MIRAnalysis
from typing import Any


@dataclasses.dataclass
class MIRPipeline:
    audio_analyses: list[MIRAnalysis]
    beat_analyses: list[MIRAnalysis]
    key_analyses: list[MIRAnalysis]
    clap_analyses: list[MIRAnalysis]
    beat_tracker: Any

    def process(
        self,
        audio_filename: str,
        prompt_filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        metadata["audio_filename"] = audio_filename
        beats, downbeats = self.beat_tracker.track(audio_filename)
        audio, sr = librosa.load(audio_filename)
        with open(prompt_filename, "r", encoding="utf-8") as file:
            prompt = file.read()
        processed: dict[str, Any] = {"metadata": metadata}
        for analysis in self.audio_analyses:
            processed.update(analysis.process_audio(data=(audio, sr)))
        for analysis in self.beat_analyses:
            processed.update(analysis.process_audio(data=(beats, downbeats)))
        for analysis in self.key_analyses:
            processed.update(analysis.process_audio(audio=audio, sr=sr))
        for analysis in self.clap_analyses:
            processed.update(
                analysis.process_audio(audio_path=audio_filename, prompt=prompt)
            )
        return processed
