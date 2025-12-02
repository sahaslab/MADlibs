from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
import soundfile as sf
from src.prompt.MadlibsPrompt import MadlibsPrompt


def test_prompts(vocab_file):
    prompt_gen = MadlibsPrompt(part_of_speech_filename=vocab_file)
    prompts = prompt_gen.get_prompts(n_prompts=5)
    return prompts


def prompt_musicgen(prompts):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
    model.to("cuda")
    inputs = processor(
        text=prompts,
        padding=True,
        return_tensors="pt",
    )
    inputs.to("cuda")
    audio_values = model.generate(
        **inputs, do_sample=True, guidance_scale=3, max_new_tokens=256
    )
    return audio_values.detach().cpu(), model.config.audio_encoder.sampling_rate


def save_musicgen_outputs(prompts, audio_values, output_dir, sr):
    for i, prompt in enumerate(prompts):
        write_path = os.path.join(output_dir, f"prompt_{i}")
        os.makedirs(write_path, exist_ok=True)
        with open(os.path.join(write_path, "prompt.txt"), "w") as f:
            f.write(prompt)
        sf.write(os.path.join(write_path, "output.wav"), audio_values[i][0].numpy(), sr)


def main():
    output_dir = "outputs"
    vocab = "tests/prompt/text_files/parts_of_speech.txt"
    prompts = test_prompts(vocab)
    audio_values, sr = prompt_musicgen(prompts)
    save_musicgen_outputs(prompts, audio_values, output_dir, sr)


if __name__ == "__main__":
    main()
