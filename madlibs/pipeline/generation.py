from hydra.utils import instantiate
from madlibs.generation.base import MusicGenerator


def generate_model_outputs(
    generator_cfg: dict,
    prompts: list[str],
    output_dir: str,
) -> str:
    generator = instantiate(generator_cfg)
    if not isinstance(generator, MusicGenerator):
        raise TypeError(
            f"Hydra config must instantiate MusicGenerator, got: {type(generator)}"
        )
    return generator.generate(prompts=prompts, output_dir=output_dir)
