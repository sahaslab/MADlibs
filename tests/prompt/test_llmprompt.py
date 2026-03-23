import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from madlibs.prompt.llm import LLMPromptStrategy
from madlibs.prompt import llm as llm_module


def test_caption_alternates_outputs_k_times_n(
    tmp_path,
    monkeypatch,
):
    caption_csv = tmp_path / "captions.csv"
    caption_csv.write_text(
        "caption_id,track_id,caption,is_valid_subset\n"
        "1,11,a big guitar playing loudly,True\n"
        "2,22,a soft tree snapping quickly,True\n"
        "3,33,a noisy crowd in a market,False\n",
        encoding="utf-8",
    )

    call_count = {"n": 0}

    def _fake_completion(**kwargs):
        call_count["n"] += 1
        messages = kwargs["messages"]
        user_text = messages[-1]["content"]
        if "Original captions JSON array" in user_text:
            payload = {
                "prompts": [
                    [
                        "Song where bright drums hit hard.",
                        "Song where calm piano moves softly.",
                    ],
                    [
                        "Song where warm bass glides smoothly.",
                        "Song where crisp hats bounce lightly.",
                    ],
                ]
            }
            return {"choices": [{"message": {"content": json.dumps(payload)}}]}
        payload = {"prompt": "Song where crisp synths pulse steadily."}
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    strategy = LLMPromptStrategy(
        generation_mode="caption_alternates",
        model_id="fake-model",
        caption_corpus_filename=str(caption_csv),
        caption_column=2,
        caption_has_header=True,
        caption_filter_column=3,
        caption_filter_value="True",
        alternates_per_caption=2,
        captions_per_request=8,
        output_prefix="Song where ",
        num_workers=1,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
    )

    prompts = strategy.get_prompts(n_prompts=1, seed=0)
    assert len(prompts) == 4
    assert call_count["n"] == 1
    for prompt in prompts:
        assert prompt.startswith("Song where ")

    template = strategy.caption_to_slot_template("a big guitar playing loudly")
    assert "[adj]" in template
    assert "[noun]" in template


def test_caption_alternates_normalizes_prompt_count(
    tmp_path,
    monkeypatch,
):
    caption_csv = tmp_path / "captions.csv"
    caption_csv.write_text(
        "caption_id,track_id,caption,is_valid_subset\n"
        "1,11,a big guitar playing loudly,True\n"
        "2,22,a soft tree snapping quickly,True\n",
        encoding="utf-8",
    )

    def _fake_completion(**kwargs):
        _ = kwargs
        payload = {
            "prompts": [
                [
                    "Song where bright drums hit hard.",
                    "Song where calm piano moves softly.",
                    "Song where warm bass glides smoothly.",
                    "Song where crisp hats bounce lightly.",
                    "Song where layered strings swell gently.",
                    "Song where bold synths pulse tightly.",
                ],
                [
                    "Song where airy pads drift slowly.",
                    "Song where deep kick thumps steadily.",
                    "Song where sparse piano rings clearly.",
                    "Song where mellow bass rolls warmly.",
                ],
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    strategy = LLMPromptStrategy(
        generation_mode="caption_alternates",
        model_id="fake-model",
        caption_corpus_filename=str(caption_csv),
        caption_column=2,
        caption_has_header=True,
        caption_filter_column=3,
        caption_filter_value="True",
        alternates_per_caption=5,
        captions_per_request=8,
        output_prefix="Song where ",
        num_workers=1,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
    )

    prompts = strategy.get_prompts(n_prompts=1, seed=0)
    assert len(prompts) == 10


def test_caption_alternates_single_group_batch_response_is_handled(
    tmp_path,
    monkeypatch,
):
    caption_csv = tmp_path / "captions.csv"
    caption_csv.write_text(
        "caption_id,track_id,caption,is_valid_subset\n"
        "1,11,a big guitar playing loudly,True\n"
        "2,22,a soft tree snapping quickly,True\n",
        encoding="utf-8",
    )

    def _fake_completion(**kwargs):
        _ = kwargs
        payload = {
            "prompts": [
                [
                    "Song where p1.",
                    "Song where p2.",
                    "Song where p3.",
                    "Song where p4.",
                    "Song where p5.",
                    "Song where p6.",
                    "Song where p7.",
                    "Song where p8.",
                    "Song where p9.",
                    "Song where p10.",
                ]
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    strategy = LLMPromptStrategy(
        generation_mode="caption_alternates",
        model_id="fake-model",
        caption_corpus_filename=str(caption_csv),
        caption_column=2,
        caption_has_header=True,
        caption_filter_column=3,
        caption_filter_value="True",
        alternates_per_caption=5,
        captions_per_request=8,
        output_prefix="Song where ",
        num_workers=1,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
    )

    prompts = strategy.get_prompts(n_prompts=1, seed=0)
    assert len(prompts) == 10
    assert prompts[0] == "Song where p1."
    assert prompts[4] == "Song where p5."
    assert prompts[5] == "Song where p6."
    assert prompts[9] == "Song where p10."


def test_seed_mode_still_works(
    monkeypatch,
):
    def _fake_completion(**kwargs):
        _ = kwargs
        payload = {"prompt": "Song where crisp synths pulse steadily."}
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    strategy = LLMPromptStrategy(
        generation_mode="seed",
        model_id="fake-model",
        num_workers=1,
    )

    prompts = strategy.get_prompts(n_prompts=3, seed=2)
    assert len(prompts) == 3
    for prompt in prompts:
        assert prompt == "Song where crisp synths pulse steadily."
