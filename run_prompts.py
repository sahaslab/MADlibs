import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from madlibs.prompt.base import PromptStrategy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    strategy = instantiate(cfg.prompt_strategy)
    if not isinstance(strategy, PromptStrategy):
        raise TypeError(
            f"prompt_strategy must instantiate PromptStrategy, got: {type(strategy)}"
        )

    prompts = strategy.get_prompts(n_prompts=cfg.run.n_prompts, seed=cfg.run.seed)

    output_path = cfg.run.prompts_filename
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and not cfg.run.overwrite_prompts:
        raise FileExistsError(
            f"{output_path} already exists. Set run.overwrite_prompts=true to overwrite."
        )

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(prompts))

    print(f"Wrote {len(prompts)} prompts to {output_path}")


if __name__ == "__main__":
    main()
