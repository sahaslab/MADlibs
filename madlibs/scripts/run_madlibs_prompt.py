from madlibs.prompt.madlibs import MadlibsPrompt

if __name__ == "__main__":
    prompt_gen = MadlibsPrompt(
        templates=["{article} {noun} {verb}s."],
        prompt_starts=["Please give me a song where "],
        part_of_speech_filename="data/parts_of_speech.txt",
    )
    prompts = prompt_gen.get_prompts(5)

    print(prompts)
