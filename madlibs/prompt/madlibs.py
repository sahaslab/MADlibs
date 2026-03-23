import dataclasses
import random
import re
import nltk
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
    tokenize_with_regex
)
from typing import Pattern


@dataclasses.dataclass
class MadlibsPrompt(CorpusPromptStrategy):
    """
    Madlibs-style; use parts-of-speech templates to scaffold prompt generation, optionally induced from caption corpus data.

    Can also optionally use a PCFG-based generator to sample POS skeletons, which would then be lexicalized with random words of the appropriate category.
    """

    templates: list[str]
    prompt_starts: list[str]
    part_of_speech_filename: str
    induction_method: str = "none"
    induced_template_style: str = "hybrid_literals"
    replace_templates_with_induced: bool = False
    min_template_placeholders: int = 3
    min_template_tokens: int = 5
    max_template_tokens: int = 12
    max_induced_templates: int = 500
    use_nltk_pos_tagger: bool = False
    use_template_field_lexicon: bool = True
    template_field_vocab_min_count: int = 2
    slot_pos: list[str] = dataclasses.field(default_factory=lambda: ["adj", "noun", "verb", "adv"])
    token_pattern: str = r"[A-Za-z]+(?:[’'][A-Za-z]+)?(?:[-][A-Za-z]+(?:[’'][A-Za-z]+)?)*|[0-9]+|[.,!?;:]"
    slot_word_pattern: str = r"^[a-z][a-z\-']+$"
    min_slot_word_length: int = 2
    max_slot_word_length: int = 20
    field_to_pos: dict[str, str] = dataclasses.field(default_factory=dict)
    pos_tag_to_pos: dict[str, str] = dataclasses.field(default_factory=dict)
    words: PosLexicon = dataclasses.field(default_factory=dict)
    template_weights: list[float] = dataclasses.field(default_factory=list)
    template_field_word_weights: TemplateFieldWordWeights = dataclasses.field(default_factory=dict)
    word_to_pos_lookup: WordLookup = dataclasses.field(default_factory=dict, init=False, repr=False)
    token_regex: Pattern[str] = dataclasses.field(init=False, repr=False)
    slot_word_regex: Pattern[str] = dataclasses.field(init=False, repr=False)
    slot_pos_set: set[str] = dataclasses.field(default_factory=set, init=False, repr=False)
    pcfg_model: object | None = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize text processing primitives
        self.token_regex = re.compile(self.token_pattern)
        self.slot_word_regex = re.compile(self.slot_word_pattern)
        self.slot_pos_set = set(self.slot_pos)
        self.words = load_part_of_speech_words(self.part_of_speech_filename, self.slot_word_regex, self.min_slot_word_length, self.max_slot_word_length)
        self.word_to_pos_lookup = build_word_to_pos_lookup(self.words)
        self.template_weights = [1.0] * len(self.templates)

        # Select and fit the structural generator from config
        if self.induction_method == "none":
            return
        if self.induction_method == "templates":
            induced_templates, induced_weights, induced_field_word_weights = self.build_weighted_templates_from_corpus()
            self.merge_induced_templates(induced_templates, induced_weights, induced_field_word_weights)
            return
        if self.induction_method == "pcfg":
            category_sequences = self.induce_category_sequences_from_corpus()
            self.pcfg_model = self.build_pcfg(category_sequences)
            return
        raise ValueError("induction_method must be one of: none, templates, pcfg. Got %s." % (self.induction_method))

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        random.seed(seed if seed is not None else self.seed)
        prompts = []
        for _ in tqdm(range(n_prompts), desc="Generating prompts"):
            # Compose each prompt as a prefix + a generated body
            prompt_prefix = random.choice(self.prompt_starts)
            prompt_body = self.generate_prompt_body()
            prompts.append("%s%s" % (prompt_prefix, prompt_body))
        return prompts

    def generate_prompt_body(self) -> str:
        # If PCFG, sample a POS skeleton first and lexicalize it
        if self.pcfg_model is not None:
            sampled_categories = self.sample_categories_from_pcfg()
            sampled_text = self.fill_category_sequence(sampled_categories)
            if sampled_text:
                return sampled_text

        # Otherwise sample a template and fill its placeholders
        template = random.choices(self.templates, weights=self.template_weights, k=1)[0]
        template_values = self.sample_template_values(template)
        return template.format(**template_values)

    def sample_template_values(self, template: str) -> dict[str, str]:
        template_values = {}
        for field_name in template_fields_in_order(template):
            if field_name in template_values:
                continue

            template_values[field_name] = self.sample_word_for_field(template, field_name)
        return template_values

    def sample_word_for_field(self, template: str, field_name: str) -> str:
        if self.use_template_field_lexicon:
            template_weights = self.template_field_word_weights.get(template, {})
            field_weights = template_weights.get(field_name)

            if field_weights is not None:
                candidate_words, candidate_weights = field_weights
                return random.choices(candidate_words, weights=candidate_weights, k=1)[0]

        field_pos = self.field_to_pos.get(field_name, strip_numeric_suffix(field_name))

        return random.choice(self.words[field_pos])

    def build_weighted_templates_from_corpus(self) -> tuple[list[str], list[float], TemplateFieldWordWeights]:
        template_counts = Counter()
        template_field_counts = {}
        captions = self.read_captions_from_corpus()

        for caption in captions:
            # Convert each caption into a typed template plus slot realizations
            tokens = tokenize_with_regex(caption, self.token_regex)
            categories = self.tokens_to_categories_aligned(tokens)
            template, slot_values, slot_count, token_count = self.tokens_to_template(tokens, categories)

            if (token_count < self.min_template_tokens) or (token_count > self.max_template_tokens) or (slot_count < self.min_template_placeholders):
                continue

            template_counts[template] += 1
            template_field_counts.setdefault(template, {})

            for field_name, field_word in slot_values.items():
                if field_name not in template_field_counts[template]:
                    template_field_counts[template][field_name] = Counter()
                template_field_counts[template][field_name][field_word] += 1

        # Keep most frequent structures, then field-level lexical weights
        most_common_templates = template_counts.most_common(self.max_induced_templates)
        selected_templates = [template for template, _ in most_common_templates]
        selected_weights = [float(count) for _, count in most_common_templates]
        selected_field_word_weights = finalize_template_field_word_weights(template_field_counts, selected_templates, self.template_field_vocab_min_count)

        return selected_templates, selected_weights, selected_field_word_weights

    def tokens_to_template(self, tokens: list[str], categories: list[str | None]) -> tuple[str, dict[str, str], int, int]:
        field_counts = {}
        slot_values = {}
        template_tokens = []
        slot_count = 0
        token_count = 0
        for token, pos_name in zip(tokens, categories):
            if token in [".", ",", "!", "?", ";", ":"]:
                template_tokens.append(token)
                continue

            normalized_token = normalize_token(token)

            if not normalized_token:
                continue

            token_count += 1

            # Replace eligible POS tokens with placeholders and record observed fillers
            if pos_name is not None and pos_name in self.slot_pos_set:
                slot_count += 1
                field_occurrence = field_counts.get(pos_name, 0) + 1
                field_counts[pos_name] = field_occurrence
                field_name = pos_name if field_occurrence == 1 else "%s%s" % (pos_name, field_occurrence)
                template_tokens.append("{%s}" % (field_name))
                cleaned_word = clean_slot_word(normalized_token)
                if is_valid_slot_word(cleaned_word, self.slot_word_regex, self.min_slot_word_length, self.max_slot_word_length):
                    slot_values[field_name] = cleaned_word
                continue

            # Optional fully-placeholder mode keeps only the POS tag as template material
            if self.induced_template_style == "all_placeholders" and pos_name is not None:
                field_occurrence = field_counts.get(pos_name, 0) + 1
                field_counts[pos_name] = field_occurrence
                field_name = pos_name if field_occurrence == 1 else "%s%s" % (pos_name, field_occurrence)
                template_tokens.append("{%s}" % (field_name))
                cleaned_word = clean_slot_word(normalized_token)

                if is_valid_slot_word(cleaned_word, self.slot_word_regex, self.min_slot_word_length, self.max_slot_word_length):
                    slot_values[field_name] = cleaned_word
                continue

            template_tokens.append(normalized_token)

        template = " ".join(template_tokens)
        template = re.sub(r"\s+([.,!?;:])", r"\1", template)

        if template and template[-1] not in [".", "!", "?"]:
            template = "%s." % (template)

        return template, slot_values, slot_count, token_count

    def merge_induced_templates(self, templates: list[str], weights: list[float], field_word_weights: TemplateFieldWordWeights) -> None:
        if self.replace_templates_with_induced:
            self.templates = templates
            self.template_weights = weights
            self.template_field_word_weights = field_word_weights
            return

        self.templates = self.templates + templates
        self.template_weights = self.template_weights + weights
        merged_field_weights = dict(self.template_field_word_weights)
        merged_field_weights.update(field_word_weights)
        self.template_field_word_weights = merged_field_weights

    def induce_category_sequences_from_corpus(self) -> list[list[str]]:
        category_sequences = []
        captions = self.read_captions_from_corpus()
        for caption in captions:
            tokens = tokenize_with_regex(caption, self.token_regex)
            categories = self.tokens_to_categories(tokens)

            if (len(categories) < self.min_template_tokens) or (len(categories) > self.max_template_tokens) or len(categories) < self.min_template_placeholders:
                continue

            category_sequences.append(categories)

        return category_sequences

    def tokens_to_categories(self, tokens: list[str]) -> list[str]:
        categories = self.tokens_to_categories_aligned(tokens)
        return [category for category in categories if category is not None]

    def tokens_to_categories_aligned(self, tokens: list[str]) -> list[str | None]:
        if self.use_nltk_pos_tagger:
            return self.tokens_to_categories_aligned_with_nltk(tokens)

        categories = []
        for token in tokens:
            normalized_token = normalize_token(token)
            categories.append(self.word_to_pos_lookup.get(normalized_token) if normalized_token else None)

        return categories

    def tokens_to_categories_aligned_with_nltk(self, tokens: list[str]) -> list[str | None]:
        tagged_tokens = nltk.pos_tag(tokens)
        categories = []

        for token, pos_tag in tagged_tokens:
            normalized_token = normalize_token(token)

            if not normalized_token:
                categories.append(None)
                continue

            pos_name = self.word_to_pos_lookup.get(normalized_token)
            if pos_name is None:
                pos_name = self.pos_tag_to_pos.get(pos_tag)

            categories.append(pos_name)

        return categories

    def build_pcfg(self, category_sequences: list[list[str]]) -> object:
        productions = []
        for category_sequence in category_sequences:
            children = [nltk.Tree("POS_%s" % (pos_name.upper()), ["__%s__" % (pos_name)]) for pos_name in category_sequence]
            parse_tree = nltk.Tree("S", children)
            productions.extend(parse_tree.productions())

        start_symbol = nltk.Nonterminal("S")

        return nltk.induce_pcfg(start_symbol, productions)

    def sample_categories_from_pcfg(self) -> list[str]:
        sampled_terminals = []
        symbol_stack = [self.pcfg_model.start()]
        while symbol_stack:
            symbol = symbol_stack.pop()

            if isinstance(symbol, str):
                sampled_terminals.append(symbol)
                continue

            productions = self.pcfg_model.productions(lhs=symbol)
            production_weights = [production.prob() for production in productions]
            sampled_production = random.choices(productions, weights=production_weights, k=1)[0]
            rhs_symbols = sampled_production.rhs()

            for rhs_symbol in reversed(rhs_symbols):
                symbol_stack.append(rhs_symbol)

        sampled_categories = []
        for terminal in sampled_terminals:
            if terminal.startswith("__") and terminal.endswith("__"):
                sampled_categories.append(terminal[2:-2])

        return sampled_categories

    def fill_category_sequence(self, category_sequence: list[str]) -> str:
        generated_words = [random.choice(self.words[pos_name]) for pos_name in category_sequence if pos_name in self.words and self.words[pos_name]]
        return "%s." % (" ".join(generated_words))
