import dataclasses
import random
import re
from collections import Counter
from tqdm.auto import tqdm
from madlibs.prompt.base import CorpusPromptStrategy
from madlibs.prompt.common import (
    PosLexicon,
    TemplateFieldWordWeights,
    WordLookup,
    build_word_to_pos_lookup,
    clean_slot_word,
    finalize_template_field_word_weights,
    is_valid_slot_word,
    load_part_of_speech_words,
    normalize_token,
    strip_numeric_suffix,
    template_fields_in_order,
    tokenize_with_regex,
)
from typing import Pattern


@dataclasses.dataclass
class DependencyGraphPromptStrategy(CorpusPromptStrategy):
    """
    Depedency graph-based prompt generation, where we learn structural templates from caption corpus.

    The basic strategy is:
    1. Tokenize caption text and map tokens to coarse PoS categories
    2. Pick root (first verb else first noun) to locally anchor a simple dependency graph
    3. Assign role labels by simple rules over token position, category, and root relationships, etc.
    4. Convert captions into role templates, accumulate frequencies, and then sample by weight
    """

    prompt_starts: list[str]
    part_of_speech_filename: str
    min_template_placeholders: int = 3
    min_template_tokens: int = 5
    max_template_tokens: int = 14
    max_graph_templates: int = 500
    template_field_vocab_min_count: int = 2
    token_pattern: str = r"[A-Za-z]+(?:[’'][A-Za-z]+)?(?:[-][A-Za-z]+(?:[’'][A-Za-z]+)?)*|[0-9]+|[.,!?;:]"
    slot_word_pattern: str = r"^[a-z][a-z\-']+$"
    min_slot_word_length: int = 2
    max_slot_word_length: int = 20
    role_to_pos: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "root_verb": "verb",
            "aux_verb": "verb",
            "comp_verb": "verb",
            "root_noun": "noun",
            "subj_noun": "noun",
            "obj_noun": "noun",
            "subj_adj": "adj",
            "obj_adj": "adj",
            "adj": "adj",
            "pre_adv": "adv",
            "post_adv": "adv",
        }
    )
    words: PosLexicon = dataclasses.field(default_factory=dict)
    templates: list[str] = dataclasses.field(default_factory=list)
    template_weights: list[float] = dataclasses.field(default_factory=list)
    template_field_word_weights: TemplateFieldWordWeights = dataclasses.field(
        default_factory=dict
    )
    word_to_pos_lookup: WordLookup = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )
    token_regex: Pattern[str] = dataclasses.field(init=False, repr=False)
    slot_word_regex: Pattern[str] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Build lexical resources and induce weighted role templates
        self.token_regex = re.compile(self.token_pattern)
        self.slot_word_regex = re.compile(self.slot_word_pattern)
        self.words = load_part_of_speech_words(
            self.part_of_speech_filename,
            self.slot_word_regex,
            self.min_slot_word_length,
            self.max_slot_word_length,
        )
        self.word_to_pos_lookup = build_word_to_pos_lookup(self.words)
        templates, weights, field_word_weights = self.build_dependency_templates()
        self.templates = templates
        self.template_weights = weights
        self.template_field_word_weights = field_word_weights

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        random.seed(seed if seed is not None else self.seed)
        prompts = []

        for _ in tqdm(range(n_prompts), desc="Generating prompts"):
            # Sample learned role-template and lexicalize it into text
            prompt_prefix = random.choice(self.prompt_starts)
            template = random.choices(
                self.templates, weights=self.template_weights, k=1
            )[0]
            template_values = self.sample_template_values(template)
            prompts.append("%s%s" % (prompt_prefix, template.format(**template_values)))

        return prompts

    def sample_template_values(self, template: str) -> dict[str, str]:
        template_values = {}

        for field_name in template_fields_in_order(template):
            if field_name in template_values:
                continue

            template_values[field_name] = self.sample_word_for_field(
                template, field_name
            )

        return template_values

    def sample_word_for_field(self, template: str, field_name: str) -> str:
        template_weights = self.template_field_word_weights.get(template, {})
        field_weights = template_weights.get(field_name)

        if field_weights is not None:
            candidate_words, candidate_weights = field_weights
            return random.choices(candidate_words, weights=candidate_weights, k=1)[0]

        role_name = strip_numeric_suffix(field_name)
        base_pos = self.role_to_pos[role_name]

        return random.choice(self.words[base_pos])

    def build_dependency_templates(
        self,
    ) -> tuple[list[str], list[float], TemplateFieldWordWeights]:
        template_counts = Counter()
        template_field_counts = {}
        captions = self.read_captions_from_corpus()
        for caption in captions:
            # Convert each caption into role sequence around an estimated root
            tokens = tokenize_with_regex(caption, self.token_regex)
            categories = [
                self.word_to_pos_lookup.get(normalize_token(token)) for token in tokens
            ]
            root_idx = self.find_root_index(categories)
            template, field_values, placeholder_count, word_token_count = (
                self.tokens_to_dependency_template(tokens, categories, root_idx)
            )

            if (
                (placeholder_count < self.min_template_placeholders)
                or (word_token_count < self.min_template_tokens)
                or (word_token_count > self.max_template_tokens)
            ):
                continue

            template_counts[template] += 1
            if template not in template_field_counts:
                template_field_counts[template] = {}

            for field_name, field_word in field_values.items():
                if field_name not in template_field_counts[template]:
                    template_field_counts[template][field_name] = Counter()

                template_field_counts[template][field_name][field_word] += 1

        # Keep top role-templates and their field-level lexical priors
        top_templates = template_counts.most_common(self.max_graph_templates)
        templates = [template for template, _ in top_templates]
        weights = [float(count) for _, count in top_templates]
        template_field_weights = finalize_template_field_word_weights(
            template_field_counts, templates, self.template_field_vocab_min_count
        )

        return templates, weights, template_field_weights

    def tokens_to_dependency_template(
        self, tokens: list[str], categories: list[str | None], root_idx: int
    ) -> tuple[str, dict[str, str], int, int]:
        noun_indices = [idx for idx, pos in enumerate(categories) if pos == "noun"]
        role_counts = {}
        field_values = {}
        template_tokens = []
        placeholder_count = 0
        word_token_count = 0

        for idx, token in enumerate(tokens):
            if token in [".", ",", "!", "?", ";", ":"]:
                template_tokens.append(token)
                continue

            normalized = normalize_token(token)

            if not normalized:
                continue

            word_token_count += 1

            # Assign structural role to each token, convert roles into placeholders
            role_name = self.dependency_role_for_token(
                idx, categories[idx], root_idx, noun_indices
            )
            if role_name is None:
                template_tokens.append(normalized)
                continue

            role_count = role_counts.get(role_name, 0) + 1
            role_counts[role_name] = role_count
            field_name = (
                role_name if role_count == 1 else "%s%s" % (role_name, role_count)
            )
            template_tokens.append("{%s}" % (field_name))
            cleaned_word = clean_slot_word(normalized)

            if is_valid_slot_word(
                cleaned_word,
                self.slot_word_regex,
                self.min_slot_word_length,
                self.max_slot_word_length,
            ):
                field_values[field_name] = cleaned_word

            placeholder_count += 1

        template = " ".join(template_tokens)
        template = re.sub(r"\s+([.,!?;:])", r"\1", template)

        if template and template[-1] not in [".", "!", "?"]:
            template = "%s." % (template)

        return template, field_values, placeholder_count, word_token_count

    def dependency_role_for_token(
        self,
        token_idx: int,
        token_pos: str | None,
        root_idx: int,
        noun_indices: list[int],
    ) -> str | None:
        # Choose a role by POS and position relative to root
        if token_pos == "verb":
            return {
                token_idx == root_idx: "root_verb",
                token_idx > root_idx: "comp_verb",
            }.get(True, "aux_verb")

        if token_pos == "noun":
            return {
                token_idx == root_idx: "root_noun",
                token_idx < root_idx: "subj_noun",
            }.get(True, "obj_noun")

        if token_pos == "adj":
            nearest_noun_idx = self.nearest_index(token_idx, noun_indices)
            is_missing_noun = nearest_noun_idx is None
            is_subject_adj = (
                nearest_noun_idx is not None and nearest_noun_idx < root_idx
            )
            return {is_missing_noun: "adj", is_subject_adj: "subj_adj"}.get(
                True, "obj_adj"
            )

        if token_pos == "adv":
            return {token_idx < root_idx: "pre_adv"}.get(True, "post_adv")

        return None

    def nearest_index(self, source_idx: int, candidates: list[int]) -> int | None:
        return (
            min(candidates, key=lambda idx: abs(idx - source_idx))
            if candidates
            else None
        )

    def find_root_index(self, categories: list[str | None]) -> int:
        for idx, pos in enumerate(categories):
            if pos in ["verb", "noun"]:
                return idx

        return 0
