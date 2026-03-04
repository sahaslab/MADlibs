import random
import dataclasses
from tqdm.auto import tqdm
from madlibs.prompt.base import PromptStrategy


@dataclasses.dataclass
class MadlibsPrompt(PromptStrategy):
    templates: list[str]
    prompt_starts: list[str]
    part_of_speech_filename: str
    seed: int = 0
    words: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        with open(self.part_of_speech_filename, "r", encoding="utf-8") as file_obj:
            lines = [line.strip() for line in file_obj if line.strip()]

        current_pos = None
        for line in lines:
            if line.startswith("*") and line.endswith("*"):
                current_pos = line.strip("*")
                self.words[current_pos] = []
                continue

            if current_pos is None:
                continue

            self.words[current_pos].extend(w.strip() for w in line.split(","))

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        random.seed(seed if seed is not None else self.seed)

        prompts = []
        for _ in tqdm(range(n_prompts), desc="Generating prompts"):
            prompt = random.choice(self.prompt_starts)
            template = random.choice(self.templates)
            prompt += template.format(
                article=random.choice(self.words["article"]),
                article2=random.choice(self.words["article"]),
                noun=random.choice(self.words["noun"]),
                noun2=random.choice(self.words["noun"]),
                verb=random.choice(self.words["verb"]),
                verb2=random.choice(self.words["verb"]),
                adv=random.choice(self.words["adv"]),
                adj=random.choice(self.words["adj"]),
                relative_adv=random.choice(self.words["relative_adv"]),
                pronoun=random.choice(self.words["pronoun"]),
                conjunction=random.choice(self.words["conjunction"]),
            )
            prompts.append(prompt)

        return prompts
