import dataclasses
import re
from collections import Counter
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from tqdm.auto import tqdm
from madlibs.prompt.base import CorpusPromptStrategy, PromptStrategy
from madlibs.prompt.common import normalize_token, tokenize_with_regex
from typing import Pattern


@dataclasses.dataclass
class RankedPromptStrategy(CorpusPromptStrategy):
    """
    Ranked prompt generation strategy that over-generates candidates from a base strategy and then scores and filters them.
    """

    base_strategy: PromptStrategy
    candidate_multiplier: int = 6
    lm_order: int = 3
    min_candidate_tokens: int = 4
    score_weight_lm: float = 1.0
    score_weight_diversity: float = 0.8
    score_weight_repetition_penalty: float = 1.0
    max_jaccard_similarity: float = 0.9
    token_pattern: str = r"[A-Za-z]+(?:[’'][A-Za-z]+)?(?:[-][A-Za-z]+(?:[’'][A-Za-z]+)?)*|[0-9]+|[.,!?;:]"
    token_regex: Pattern[str] = dataclasses.field(init=False, repr=False)
    lm_model: KneserNeyInterpolated = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.caption_corpus_filename = getattr(
            self.base_strategy, "caption_corpus_filename", self.caption_corpus_filename
        )
        self.caption_column = getattr(
            self.base_strategy, "caption_column", self.caption_column
        )
        self.caption_has_header = getattr(
            self.base_strategy, "caption_has_header", self.caption_has_header
        )
        self.caption_delimiter = getattr(
            self.base_strategy, "caption_delimiter", self.caption_delimiter
        )
        self.caption_filter_column = getattr(
            self.base_strategy, "caption_filter_column", self.caption_filter_column
        )
        self.caption_filter_value = getattr(
            self.base_strategy, "caption_filter_value", self.caption_filter_value
        )
        self.token_pattern = getattr(
            self.base_strategy, "token_pattern", self.token_pattern
        )

        self.token_regex = re.compile(self.token_pattern)
        self.train_language_model()

    def get_prompts(self, n_prompts: int, seed: int | None = None) -> list[str]:
        # Over-generate first, then score-and-filter
        candidate_count = max(n_prompts * self.candidate_multiplier, n_prompts)
        candidates = self.base_strategy.get_prompts(
            candidate_count, seed if seed is not None else self.seed
        )
        ranked_candidates = self.rank_candidates(candidates)

        return self.select_diverse_top_n(ranked_candidates, n_prompts)

    def rank_candidates(self, candidates: list[str]) -> list[tuple[float, str]]:
        scored = [
            (self.score_candidate(candidate), candidate)
            for candidate in tqdm(candidates, desc="Scoring candidates")
        ]
        return sorted(scored, key=lambda item: item[0], reverse=True)

    def select_diverse_top_n(
        self, ranked_candidates: list[tuple[float, str]], n_prompts: int
    ) -> list[str]:
        selected = []
        selected_token_sets = []
        for _, candidate in ranked_candidates:
            token_set = set(self.tokenize_words(candidate))

            # Enforce diversity by rejecting candidates too close to selected prompts
            similarities = [
                1.0 - jaccard_distance(token_set, existing_set)
                for existing_set in selected_token_sets
            ]
            if similarities and max(similarities) > self.max_jaccard_similarity:
                continue

            selected.append(candidate)
            selected_token_sets.append(token_set)

            if len(selected) >= n_prompts:
                return selected

        for _, candidate in ranked_candidates:
            if candidate in selected:
                continue

            selected.append(candidate)

            if len(selected) >= n_prompts:
                return selected

        return selected[:n_prompts]

    def score_candidate(self, candidate: str) -> float:
        tokens = self.tokenize_words(candidate)

        if len(tokens) < self.min_candidate_tokens:
            return float("-inf")

        # Composite objective is weighted combination of language model score, lexical diversity, and repetition penalty
        lm_score = self.language_model_score(tokens)
        diversity_score = len(set(tokens)) / float(len(tokens))
        repetition_penalty = self.repetition_penalty(tokens)
        return (
            (self.score_weight_lm * lm_score)
            + (self.score_weight_diversity * diversity_score)
            - (self.score_weight_repetition_penalty * repetition_penalty)
        )

    def language_model_score(self, tokens: list[str]) -> float:
        padded = (["<s>"] * (self.lm_order - 1)) + tokens + ["</s>"]
        scores = []

        for idx in range(self.lm_order - 1, len(padded)):
            start_idx = idx - self.lm_order + 1
            ngram_tokens = padded[start_idx : idx + 1]
            context = ngram_tokens[:-1]
            word = ngram_tokens[-1]
            scores.append(float(self.lm_model.logscore(word, context)))

        return sum(scores) / float(len(scores))

    def repetition_penalty(self, tokens: list[str]) -> float:
        bigram_counts = Counter(ngrams(tokens, 2))
        repeated_total = sum(
            [count - 1 for count in bigram_counts.values() if count > 1]
        )

        return repeated_total / float(len(tokens) - 1)

    def train_language_model(self) -> None:
        # Use caption corpus as an in-domain language prior for ranking
        corpus_lines = self.read_captions_from_corpus()
        tokenized_corpus = [self.tokenize_words(line) for line in corpus_lines]
        train_ngrams, vocab = padded_everygram_pipeline(self.lm_order, tokenized_corpus)
        model = KneserNeyInterpolated(order=self.lm_order)
        model.fit(train_ngrams, vocab)
        self.lm_model = model

    def tokenize_words(self, text: str) -> list[str]:
        tokens = tokenize_with_regex(text, self.token_regex)
        normalized_tokens = [normalize_token(token) for token in tokens]
        return [token for token in normalized_tokens if token]
