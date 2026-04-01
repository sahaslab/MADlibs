import os
import logging
import glob
import json
import hydra
import soundfile
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    audio_paths = sorted(glob.glob(cfg.run.audio_glob))
    assert len(audio_paths) > 0, f"No audio files found for glob: {cfg.run.audio_glob}"
    mir_output_path = cfg.run.mir_output_filename
    instrument_output_path = cfg.run.instrument_output_filename
    analysis_type = cfg.run.analysis_type
    prompt_filename = cfg.run.prompt_filename
    os.makedirs(os.path.dirname(mir_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(instrument_output_path), exist_ok=True)
    if analysis_type == "mir":
        pipeline = instantiate(cfg.analysis.pipeline)
        valid_audio_count = 0
        with open(mir_output_path, "w", encoding="utf-8") as file_obj:
            for audio_path in tqdm(audio_paths):
                # First check if file is valid
                info = soundfile.info(audio_path)
                prompt_path = os.path.join(os.path.dirname(audio_path), prompt_filename)
                if info.frames == 0:
                    logger.warning(f"Skipping {audio_path} because it has 0 frames")
                    continue

                try:
                    result = pipeline.process(
                        audio_filename=audio_path, prompt_filename=prompt_path
                    )
                except Exception as error:
                    logger.error(f"Error processing {audio_path}: {error}")
                    continue

                serializable = {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in result.items()
                }
                file_obj.write(json.dumps(serializable) + "\n")
                valid_audio_count += 1
        print(f"Wrote MIR results for {valid_audio_count} files to {mir_output_path}")
        return

    if analysis_type == "instrument":
        instrument_pipeline = instantiate(cfg.analysis.instrument_pipeline)
        valid_audio_paths = []
        for audio_path in tqdm(audio_paths):
            info = soundfile.info(audio_path)
            if info.frames == 0:
                logger.warning(f"Skipping {audio_path} because it has 0 frames")
                continue
            valid_audio_paths.append(audio_path)

        instrument_conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instrument_pipeline.instrument_tagger.prompt,
                        },
                        {"type": "audio", "path": audio_path},
                    ],
                }
            ]
            for audio_path in valid_audio_paths
        ]
        instrument_results = instrument_pipeline.process(
            audio_filenames=valid_audio_paths,
            conversations=instrument_conversations,
        )

        with open(instrument_output_path, "w", encoding="utf-8") as file_obj:
            for audio_path in valid_audio_paths:
                serializable = {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in instrument_results[audio_path].items()
                }
                serializable["audio_filename"] = audio_path
                file_obj.write(json.dumps(serializable) + "\n")
        print(
            f"Wrote instrument-tagging results for {len(valid_audio_paths)} files to "
            f"{instrument_output_path}"
        )
        return

    raise ValueError(
        f"Unsupported run.analysis_type={analysis_type}. "
        "Expected one of: mir, instrument."
    )


if __name__ == "__main__":
    main()
