import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from madlibs.prompt.dependency_graph import DependencyGraphPromptStrategy


def test_dependency_graph_prompt_generation(
    tmp_path,
):
    caption_csv = tmp_path / "captions.csv"
    caption_csv.write_text(
        "caption_id,track_id,caption,is_valid_subset\n"
        "1,11,a bright guitar plays softly in the room,True\n"
        "2,22,a mellow piano drifts gently at night,True\n"
        "3,33,loud crowd at market,False\n",
        encoding="utf-8",
    )

    strategy = DependencyGraphPromptStrategy(
        prompt_starts=["Song where "],
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
        caption_corpus_filename=str(caption_csv),
        caption_column=2,
        caption_has_header=True,
        caption_filter_column=3,
        caption_filter_value="True",
        min_template_placeholders=2,
        min_template_tokens=3,
        max_template_tokens=12,
        max_graph_templates=20,
    )

    prompts = strategy.get_prompts(n_prompts=4, seed=7)
    assert len(prompts) == 4
    for prompt in prompts:
        assert prompt.startswith("Song where ")
        assert "{" not in prompt
