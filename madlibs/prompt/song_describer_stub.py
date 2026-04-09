import logging
import dataclasses
from madlibs.prompt.base import PromptStrategy


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SongDescriberStub(PromptStrategy):
    """
    Stub to run generation on (existing) SDD prompts. Doesn't do anything
    """

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        logger.info(f"Generating {n_prompts} SDD prompts; will ignore seed {seed}.")
        return [None for _ in range(n_prompts)]
