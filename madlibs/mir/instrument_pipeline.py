import dataclasses
from typing import Dict, Any

from madlibs.mir.instrument_tagger import InstrumentTagger


@dataclasses.dataclass
class InstrumentPipeline:
    instrument_tagger: InstrumentTagger
    batch_size: int = 4

    def process(
        self, audio_filenames: list[str], conversations: list[list[dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        if len(audio_filenames) != len(conversations):
            raise ValueError("audio_filenames and conversations must have equal length")

        processed: dict[str, dict[str, Any]] = {}
        for idx in range(0, len(audio_filenames), self.batch_size):
            batch_audio = audio_filenames[idx : idx + self.batch_size]
            batch_conversations = conversations[idx : idx + self.batch_size]
            batch_results = self.instrument_tagger.process_audio_batch(
                audio_paths=batch_audio,
                conversations=batch_conversations,
            )

            for audio_filename, result in zip(batch_audio, batch_results):
                processed[audio_filename] = result

        return processed
