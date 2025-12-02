from abc import ABC, abstractmethod
from typing import Optional

class IPrompt(ABC):
    @abstractmethod
    def get_prompts(self, n_prompts: int, seed: Optional[int] = None) -> list[str]:
        """
        Generate n_prompts randomly following some defined strategy

        Args:
            - n_prompts: the number of prompts to generate
            - seed: the random seed

        Returns:
            - A list of prompts
        """
        pass