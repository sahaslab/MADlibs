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
    assert prompt1 == "Please give me a song where the tree snaps and they snaps."
