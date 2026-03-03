import random
import dataclasses
from madlibs.prompt.base import PromptStrategy


@dataclasses.dataclass
class KeywordMixPromptStrategy(PromptStrategy):
    """
    Simple prompt strategy that mixes and matches keywords from different categories to create prompts.
    """

    styles: list[str]
    moods: list[str]
    instruments: list[str]
    seed: int = 0

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        random.seed(seed if seed is not None else self.seed)
        prompts = []

        for _ in range(n_prompts):
            style = random.choice(self.styles)
            mood = random.choice(self.moods)
            instrument = random.choice(self.instruments)
            prompts.append(f"{mood} {style} featuring {instrument}")

        return prompts
