import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from madlibs.prompt.madlibs import MadlibsPrompt


_TEMPLATES = ["{article} {noun} {verb}s and {pronoun} {verb2}s."]
_PROMPT_STARTS = ["Please give me a song where "]


def test_prompt_len_20():
    prompt_gen = MadlibsPrompt(
        templates=_TEMPLATES,
        prompt_starts=_PROMPT_STARTS,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
    )
    prompts = prompt_gen.get_prompts(n_prompts=20)
    assert len(prompts) == 20


def test_output_with_set_seed():
    prompt_gen = MadlibsPrompt(
        templates=_TEMPLATES,
        prompt_starts=_PROMPT_STARTS,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
    )
    prompt1 = prompt_gen.get_prompts(n_prompts=1, seed=1)[0]
    prompt_gen = MadlibsPrompt(
        templates=_TEMPLATES,
        prompt_starts=_PROMPT_STARTS,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
        seed=1,
    )
    prompt2 = prompt_gen.get_prompts(n_prompts=1)[0]
    prompt3 = prompt_gen.get_prompts(n_prompts=1, seed=2)[0]

    assert prompt1 == prompt2
    assert prompt2 != prompt3
    assert prompt1.startswith("Please give me a song where ")


def test_induced_templates_from_caption_csv(tmp_path):
    caption_csv = tmp_path / "captions.csv"
    caption_csv.write_text(
        "caption\n"
        "the big tree snap quickly\n"
        "a soft guitar sing emotionally\n"
        "they play loudly\n",
        encoding="utf-8",
    )
    prompt_gen = MadlibsPrompt(
        templates=["{article} {noun} {verb}s."],
        prompt_starts=_PROMPT_STARTS,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
        caption_corpus_filename=str(caption_csv),
        caption_column=0,
        caption_has_header=True,
        caption_filter_column=None,
        induction_method="templates",
        induced_template_style="all_placeholders",
        replace_templates_with_induced=True,
        min_template_placeholders=3,
        min_template_tokens=3,
        max_template_tokens=8,
    )
    assert "{article} {adj} {noun} {verb} {adv}." in prompt_gen.templates
    prompt = prompt_gen.get_prompts(n_prompts=1, seed=7)[0]
    assert prompt.startswith("Please give me a song where ")
    assert "{" not in prompt


def test_pcfg_induction_with_nltk(tmp_path):
    caption_csv = tmp_path / "captions.csv"
    caption_csv.write_text(
        "caption\n"
        "the big tree snap quickly\n"
        "a soft guitar sing emotionally\n"
        "they play loudly\n",
        encoding="utf-8",
    )
    prompt_gen = MadlibsPrompt(
        templates=_TEMPLATES,
        prompt_starts=_PROMPT_STARTS,
        part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt",
        caption_corpus_filename=str(caption_csv),
        caption_column=0,
        caption_has_header=True,
        caption_filter_column=None,
        induction_method="pcfg",
        use_nltk_pos_tagger=False,
        min_template_placeholders=3,
        min_template_tokens=3,
        max_template_tokens=8,
    )
    prompts = prompt_gen.get_prompts(n_prompts=3, seed=9)
    assert len(prompts) == 3
    for prompt in prompts:
        assert prompt.startswith("Please give me a song where ")
