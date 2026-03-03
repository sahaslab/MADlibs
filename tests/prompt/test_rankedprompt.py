import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from madlibs.prompt.base import PromptStrategy
from madlibs.prompt.ranked import RankedPromptStrategy


class DummyPromptStrategy(PromptStrategy):
    def get_prompts(
        self,
        n_prompts: int,
        seed: int | None = None,
    ) -> list[str]:
        _ = seed
        prompts = [
            "Song where bright guitar plays softly at night.",
            "Song where piano melody drifts gently.",
            "zxqv qqqq zzzz qqqq zxqv",
            "aaaa aaaa aaaa aaaa",
            "Song where bright guitar plays softly at night.",
            "random words random words random",
        ]
        repeated = []
        while len(repeated) < n_prompts:
            repeated.extend(prompts)
        return repeated[:n_prompts]


def test_ranked_prompt_strategy_subsamples_top_n(
    tmp_path,
):
    caption_csv = tmp_path / "captions.csv"
    caption_csv.write_text(
        "caption_id,track_id,caption,is_valid_subset\n"
        "1,11,bright guitar melody at night,True\n"
        "2,22,mellow piano drifts softly,True\n"
        "3,33,ambient strings shimmer gently,True\n",
        encoding="utf-8",
    )

    strategy = RankedPromptStrategy(
        base_strategy=DummyPromptStrategy(),
        caption_corpus_filename=str(caption_csv),
        caption_column=2,
        caption_has_header=True,
        caption_filter_column=3,
        caption_filter_value="True",
        candidate_multiplier=4,
        lm_order=3,
        min_candidate_tokens=3,
        max_jaccard_similarity=0.95,
    )

    prompts = strategy.get_prompts(n_prompts=2, seed=0)
    assert len(prompts) == 2
    assert prompts[0] != prompts[1]
