import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from madlibs.generation.base import MusicGenerator
import random 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    with open(cfg.run.prompts_filename, "r", encoding="utf-8") as file_obj:
        prompt_lines = [line.rstrip("\r\n") for line in file_obj]
    if cfg.run.prompt_strategy_name == "blank":
        prompts = prompt_lines
    else:
        prompts = [line.strip() for line in prompt_lines if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {cfg.run.prompts_filename}")

    generator = instantiate(cfg.generator)
    if not isinstance(generator, MusicGenerator):
        raise TypeError(
            f"generator must instantiate MusicGenerator, got: {type(generator)}"
        )

    if cfg.run.max_prompts is not None and len(prompts)>cfg.run.max_prompts:
        prompts = random.sample(prompts,cfg.run.max_prompts)
    os.makedirs(cfg.run.generated_audio_dir, exist_ok=True)
    run_dir = generator.generate(
        prompts=prompts, output_dir=cfg.run.generated_audio_dir
    )
    print(f"Generated {len(prompts)} outputs in {run_dir}")


if __name__ == "__main__":
    main()
