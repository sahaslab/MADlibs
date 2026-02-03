from src.prompt.MadlibsPrompt import MadlibsPrompt

if __name__ == "__main__":
    prompt_gen = MadlibsPrompt(part_of_speech_filename="./src/prompt/text_files/parts_of_speech.txt")
    prompts = prompt_gen.get_prompts(5)

    print(prompts)