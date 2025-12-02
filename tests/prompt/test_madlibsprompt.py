from src.prompt.MadlibsPrompt import MadlibsPrompt


def test_prompt_len_20():
    prompt_gen = MadlibsPrompt(part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt")
    prompts = prompt_gen.get_prompts(n_prompts=20)
    assert len(prompts) == 20

def test_output_with_set_seed():
    prompt_gen = MadlibsPrompt(part_of_speech_filename="tests/prompt/text_files/parts_of_speech.txt")
    prompts = prompt_gen.get_prompts(n_prompts=1, seed=1)
    assert prompts[0] == 'Please give me a song where a river sings and I sings.'