import dataclasses
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from tqdm.auto import tqdm
from madlibs.prompt.base import PromptStrategy


class PromptOutput(BaseModel):
    prompt: str


@dataclasses.dataclass
class LLMPromptStrategy(PromptStrategy):
    model_id: str = "openai/gpt-4o-mini"
    system_instruction: str = "Generate one brief music generation prompt, returning only the prompt text itself."
    user_seed_text: str = "Create a detailed prompt for instrumental music generation."
    max_new_tokens: int = 256
    temperature: float = 1.0
    num_workers: int = 16

    def _generate_one(self, idx: int, seed: int | None) -> str:
        messages = [
            {"content": self.system_instruction, "role": "system"},
            {"content": self.user_seed_text, "role": "user"}
        ]
        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            response_format=PromptOutput,
            seed=seed + idx if seed is not None else None
        )
        response = response["choices"][0]["message"]["content"].strip()
        summary_obj = PromptOutput.model_validate_json(response)
        return summary_obj.prompt

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        if n_prompts <= 0:
            return []

        max_workers = max(1, min(self.num_workers, n_prompts))
        prompts = [""] * n_prompts # Default to empty string

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._generate_one, idx, seed): idx
                for idx in range(n_prompts)
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=n_prompts,
                desc="Generating prompts"
            ):
                idx = future_to_idx[future]
                prompts[idx] = future.result()

        return prompts
