import dataclasses
from typing import Any

from madlibs.mir.base import MIRAnalysis


@dataclasses.dataclass
class CLAPPipeline:
    clap_analyses: list[MIRAnalysis]

    def process(self, audio_filename: str, prompt_filename: str) -> dict[str, Any]:
        with open(prompt_filename, "r", encoding="utf-8") as file:
            prompt = file.read()

        processed: dict[str, Any] = {}
        for analysis in self.clap_analyses:
            processed.update(
                analysis.process_audio(audio_path=audio_filename, prompt=prompt)
            )
        return processed
