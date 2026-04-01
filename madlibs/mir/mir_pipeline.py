import dataclasses
from typing import Any

import librosa

from madlibs.mir.base import MIRAnalysis


@dataclasses.dataclass
class MIRPipeline:
    audio_analyses: list[MIRAnalysis] = dataclasses.field(default_factory=list)

    def process(
        self,
        audio_filename: str,
        prompt_filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del prompt_filename

        if metadata is None:
            metadata = {}
        metadata["audio_filename"] = audio_filename

        processed: dict[str, Any] = {"metadata": metadata}
        if not self.audio_analyses:
            return processed

        audio, sr = librosa.load(audio_filename)
        for analysis in self.audio_analyses:
            processed.update(analysis.process_audio(data=(audio, sr)))
        return processed
