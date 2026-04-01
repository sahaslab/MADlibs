import dataclasses
from typing import Any

import librosa

from madlibs.mir.base import MIRAnalysis


@dataclasses.dataclass
class GPUPipeline:
    beat_tracker: Any | None = None
    beat_analyses: list[MIRAnalysis] = dataclasses.field(default_factory=list)
    key_analyses: list[MIRAnalysis] = dataclasses.field(default_factory=list)
    instrument_analyses: list[MIRAnalysis] = dataclasses.field(default_factory=list)
    clap_analyses: list[MIRAnalysis] = dataclasses.field(default_factory=list)

    def process(
        self,
        audio_filename: str,
        prompt_filename: str | None = None,
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

        if self.instrument_analyses:
            for analysis in self.instrument_analyses:
                processed.update(analysis.process_audio(audio_path=audio_filename))

        if self.key_analyses:
            audio, sr = librosa.load(audio_filename)
            for analysis in self.key_analyses:
                processed.update(analysis.process_audio(audio=audio, sr=sr))

        if self.clap_analyses:
            if prompt_filename is None:
                raise ValueError(
                    "prompt_filename is required when clap_analyses are set"
                )
            with open(prompt_filename, "r", encoding="utf-8") as file_obj:
                prompt = file_obj.read()
            for analysis in self.clap_analyses:
                processed.update(
                    analysis.process_audio(audio_path=audio_filename, prompt=prompt)
                )

        return processed
