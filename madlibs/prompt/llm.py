import dataclasses
import json
import logging
import re
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Pattern
from tqdm.auto import tqdm
from pydantic import BaseModel
from madlibs.prompt.base import CorpusPromptStrategy
from madlibs.prompt.common import (
    PosLexicon,
    WordLookup,
    build_word_to_pos_lookup,
    load_part_of_speech_words,
    normalize_token,
    tokenize_with_regex
)


logging.getLogger("litellm").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class PromptOutput(BaseModel):
    prompt: str


class CaptionAlternatesBatchOutput(BaseModel):
    prompts: list[list[str]]


@dataclasses.dataclass
class LLMPromptStrategy(CorpusPromptStrategy):
    """
    Use an LLM to systematically vary base prompts from a caption corpus, or directly generate new prompts in seed mode.
    """

    model_id: str = "openai/gpt-4o-mini"
    system_instruction: str = "Generate one brief music generation prompt, returning only the prompt text itself."
    user_seed_text: str = "Create a detailed prompt for instrumental music generation."
    alternates_system_instruction: str = (
        "You generate musically coherent caption alternates. "
        "Return strict JSON with a `prompts` list and nothing else."
    )
    generation_mode: str = "caption_alternates"
    alternates_per_caption: int = 3
    captions_per_request: int = 8
    max_source_captions: int | None = None
    output_prefix: str = "Song where "
    slot_pos: list[str] = dataclasses.field(default_factory=lambda: ["adj", "noun", "verb", "adv"])
    token_pattern: str = r"[A-Za-z]+(?:[’'][A-Za-z]+)?(?:[-][A-Za-z]+(?:[’'][A-Za-z]+)?)*|[0-9]+|[.,!?;:]"
    slot_word_pattern: str = r"^[a-z][a-z\-']+$"
    min_slot_word_length: int = 2
    max_slot_word_length: int = 20
    part_of_speech_filename: str = ""
    max_new_tokens: int = 256
    temperature: float = 1.0
    num_workers: int = 16
    words: PosLexicon = dataclasses.field(default_factory=dict)
    word_to_pos_lookup: WordLookup = dataclasses.field(default_factory=dict, init=False, repr=False)
    slot_word_regex: Pattern[str] = dataclasses.field(init=False, repr=False)
    slot_pos_set: set[str] = dataclasses.field(default_factory=set, init=False, repr=False)
    token_regex: Pattern[str] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize lexical resources for slot-template conversion
        self.slot_word_regex = re.compile(self.slot_word_pattern)
        self.slot_pos_set = set(self.slot_pos)
        self.token_regex = re.compile(self.token_pattern)
        if self.part_of_speech_filename:
            self.words = load_part_of_speech_words(self.part_of_speech_filename, self.slot_word_regex, self.min_slot_word_length, self.max_slot_word_length)
            self.word_to_pos_lookup = build_word_to_pos_lookup(self.words)

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        _ = seed # Keeping seed in signature for compatibility

        if self.generation_mode == "seed": # Unconditional generation of prompts
            return self.get_seed_prompts(n_prompts)

        if self.generation_mode == "caption_alternates": # Generate prompts by varying captions from the corpus
            return self.get_caption_alternate_prompts()

        raise ValueError("generation_mode must be one of: seed, caption_alternates. Got %s." % (self.generation_mode))

    def get_seed_prompts(self, n_prompts: int) -> list[str]:
        max_workers = max(1, min(self.num_workers, n_prompts))
        prompts = [""] * n_prompts

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.generate_one_seed_prompt): idx for idx in range(n_prompts)}

            for future in tqdm(as_completed(future_to_idx), total=n_prompts, desc="Generating prompts"):
                idx = future_to_idx[future]
                prompts[idx] = future.result()

        return prompts

    def generate_one_seed_prompt(self) -> str:
        messages = [
            {"content": self.system_instruction, "role": "system"},
            {"content": self.user_seed_text, "role": "user"},
        ]

        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            response_format=PromptOutput
        )

        response_text = response["choices"][0]["message"]["content"].strip()
        output_obj = PromptOutput.model_validate_json(response_text)

        return output_obj.prompt.strip()

    def get_caption_alternate_prompts(self) -> list[str]:
        # Read corpus captions and build matching slot templates
        captions = self.read_captions_from_corpus()

        if self.max_source_captions is not None:
            captions = captions[: self.max_source_captions]

        caption_templates = [self.caption_to_slot_template(caption) for caption in captions]
        caption_batches = self.build_caption_batches(captions, caption_templates)

        # Run batch requests concurrently and stitch results back to original source order
        max_workers = max(1, min(self.num_workers, len(caption_batches)))
        prompt_blocks = [[] for _ in range(len(captions))]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_alternates_for_caption_batch, batch_captions, batch_templates): idx
                for idx, (_, batch_captions, batch_templates) in enumerate(caption_batches)
            }

            for future in tqdm(as_completed(future_to_idx), total=len(caption_batches), desc="Generating prompts"):
                idx = future_to_idx[future]
                batch_start, _, _ = caption_batches[idx]
                caption_block_prompts = future.result()

                for offset, caption_prompts in enumerate(caption_block_prompts):
                    prompt_blocks[batch_start + offset] = caption_prompts

        prompts = []

        for block in prompt_blocks:
            prompts.extend(block)

        return prompts

    def build_caption_batches(self, captions: list[str], caption_templates: list[str]) -> list[tuple[int, list[str], list[str]]]:
        batches = []

        for start_idx in range(0, len(captions), self.captions_per_request):
            end_idx = min(start_idx + self.captions_per_request, len(captions))
            batches.append((start_idx, captions[start_idx:end_idx], caption_templates[start_idx:end_idx]))

        return batches

    def generate_alternates_for_caption_batch(self, original_captions: list[str], slot_template_captions: list[str]) -> list[list[str]]:
        user_text = self.format_caption_alternates_batch_user_text(original_captions, slot_template_captions)

        messages = [
            {"content": self.alternates_system_instruction, "role": "system"},
            {"content": user_text, "role": "user"}
        ]

        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            response_format=CaptionAlternatesBatchOutput
        )

        response_text = response["choices"][0]["message"]["content"].strip()
        raw_caption_groups = self.extract_caption_groups(response_text)
        caption_groups = self.normalize_caption_groups(raw_caption_groups, len(original_captions), self.alternates_per_caption)

        batch_prompts = []
        for caption_idx, caption_prompts in enumerate(caption_groups):
            prompts = [prompt.strip() for prompt in caption_prompts if prompt.strip()]
            prompts = self.normalize_prompt_count(prompts, self.alternates_per_caption, original_captions[caption_idx])
            batch_prompts.append([self.apply_output_prefix(prompt) for prompt in prompts])

        return batch_prompts

    def extract_caption_groups(self, response_text: str) -> list[list[str]]:
        try:
            output_obj = CaptionAlternatesBatchOutput.model_validate_json(response_text)
            return output_obj.prompts
        except Exception:
            # Hacky, but noticing instances where the wrong number of alternates are generated
            data = json.loads(response_text)
            raw_prompts = data.get("prompts", [])

            if isinstance(raw_prompts, list) and raw_prompts and isinstance(raw_prompts[0], str):
                return [[str(prompt) for prompt in raw_prompts]]

            if isinstance(raw_prompts, list):
                groups = []
                for entry in raw_prompts:
                    if isinstance(entry, list):
                        groups.append([str(item) for item in entry])
                    else:
                        groups.append([str(entry)])

                return groups if groups else [[]]
        finally:
            return [[]]

    def normalize_caption_groups(
        self,
        caption_groups: list[list[str]],
        target_count: int,
        target_group_size: int
    ) -> list[list[str]]:
        if len(caption_groups) == target_count:
            return caption_groups

        if not caption_groups:
            return [[] for _ in range(target_count)]

        if len(caption_groups) == 1:
            first_group = caption_groups[0]
            required = target_count * target_group_size

            if len(first_group) >= required:
                split_groups = []
                for idx in range(target_count):
                    start = idx * target_group_size
                    split_groups.append(first_group[start : start + target_group_size])

                return split_groups

            return [list(first_group) for _ in range(target_count)]

        if len(caption_groups) > target_count:
            return caption_groups[:target_count]

        return [list(caption_groups[idx % len(caption_groups)]) for idx in range(target_count)]

    def normalize_prompt_count(self, prompts: list[str], target_count: int, fallback_prompt: str) -> list[str]:
        if len(prompts) >= target_count:
            return prompts[:target_count]

        if not prompts:
            prompts = [fallback_prompt]

        return [prompts[idx % len(prompts)] for idx in range(target_count)]

    def format_caption_alternates_batch_user_text(self, original_captions: list[str], slot_template_captions: list[str]) -> str:
        batch_size = len(original_captions)
        captions_json = json.dumps(original_captions, ensure_ascii=True)
        slot_templates_json = json.dumps(slot_template_captions, ensure_ascii=True)
        return (
            "Original captions JSON array:\n%s\n\n"
            "Slot-template captions JSON array:\n%s\n\n"
            "Task:\n"
            "For each caption, generate exactly %s alternate captions that are grammatical, musically plausible, "
            "and keep the same broad structure as its slot template.\n\n"
            "Rules:\n"
            "1) Return strict JSON with key `prompts` mapped to a list of caption-lists.\n"
            "2) Do not add any keys besides `prompts`.\n"
            "3) `prompts` must have exactly %s inner lists, aligned to the input order.\n"
            "4) Prefer concise, coherent captions and avoid random keyword dumping.\n"
            "5) Each generated caption must be one sentence and should start with `%s`.\n"
            "6) Each inner list must contain exactly %s captions.\n"
        ) % (
            captions_json,
            slot_templates_json,
            self.alternates_per_caption,
            batch_size,
            self.output_prefix,
            self.alternates_per_caption
        )

    def apply_output_prefix(self, prompt: str) -> str:
        cleaned_prompt = prompt.strip()

        if len(self.output_prefix) == 0:
            return cleaned_prompt

        if cleaned_prompt.lower().startswith(self.output_prefix.lower()):
            return cleaned_prompt

        return "%s%s" % (self.output_prefix, cleaned_prompt)

    def caption_to_slot_template(self, caption: str) -> str:
        tokens = tokenize_with_regex(caption, self.token_regex)
        template_tokens = []

        for token in tokens:
            if token in [".", ",", "!", "?", ";", ":"]:
                template_tokens.append(token)
                continue

            normalized_token = normalize_token(token)
            if not normalized_token:
                continue

            pos_name = self.word_to_pos_lookup.get(normalized_token)
            if pos_name is not None and pos_name in self.slot_pos_set:
                template_tokens.append("[%s]" % (pos_name))
                continue

            template_tokens.append(normalized_token)

        template_caption = " ".join(template_tokens)
        template_caption = re.sub(r"\s+([.,!?;:])", r"\1", template_caption)
        template_caption = template_caption.strip()

        if not template_caption:
            return self.output_prefix.strip()

        if self.output_prefix and not template_caption.lower().startswith(self.output_prefix.lower()):
            return "%s%s" % (self.output_prefix, template_caption)

        return template_caption
