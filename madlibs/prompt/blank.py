import logging
import dataclasses
from madlibs.prompt.base import PromptStrategy


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BlankPromptStrategy(PromptStrategy):
    blank_prompt: str = ""

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        logger.info(f"Generating {n_prompts} blank prompts; will ignore seed {seed}.")
        return [self.blank_prompt for _ in range(n_prompts)]
