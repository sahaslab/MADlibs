from abc import ABC, abstractmethod


class PromptStrategy(ABC):
    @abstractmethod
    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        """Generate n_prompts randomly following defined strategy."""
        raise NotImplementedError
