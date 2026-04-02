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


def _resolve_pipeline_config(cfg: DictConfig):
    pipeline_key = cfg.analysis_type.pipeline_key
    return cfg.analysis[pipeline_key]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = instantiate(_resolve_pipeline_config(cfg))
    audio_paths = sorted(glob.glob(cfg.run.audio_glob))
    assert len(audio_paths) > 0, f"No audio files found for glob: {cfg.run.audio_glob}"
    prompt_filename = cfg.run.prompt_filename
    output_path = cfg.analysis_type.output_filename
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written_count = 0
    with open(output_path, "w", encoding="utf-8") as file_obj:
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
            written_count += 1
    print(
        f"Wrote {cfg.analysis_type.pipeline_key} results for {written_count} files to {output_path}"
    )


if __name__ == "__main__":
    main()
