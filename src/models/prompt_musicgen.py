from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
import soundfile as sf
from src.prompt.MadlibsPrompt import MadlibsPrompt
import uuid
import argparse



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
    run_id = uuid.uuid1()
    print(f"Saving outputs for run {run_id}")
    run_path = os.path.join(output_dir, run_id)
    os.mkdir(run_path)
    for i, prompt in enumerate(prompts):
        write_path = os.path.join(run_path, f"prompt_{i}")
        os.mkdir(write_path, exist_ok=True)
        with open(os.path.join(write_path, "prompt.txt"), "w") as f:
            f.write(prompt)
        sf.write(os.path.join(write_path, "output.wav"), audio_values[i][0].numpy(), sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to write musicgen outputs",
        defualt="outputs",
    )
    parser.add_argument(
        "-v",
        "--vocab",
        help="File for vocabulary (for madlibs style prompt)",
        defualt="outputs",
    )
    parser.add_argument(
        "-t", "--type", help="Prompt Type (MadLibs, ...)", defualt="madlibs"
    )
    parser.add_argument("-n", "--n_prompts", help="Number of Prompts", defualt="100")
    args = parser.parse_args()

    output_dir = args.output_dir
    vocab = args.vocab
    prompt_type = args.type
    n_prompts = args.n_prompts
    prompt_gen = None
    if prompt_type == "madlibs":
         prompt_gen = MadlibsPrompt(part_of_speech_filename=vocab)
    prompts = prompt_gen.get_prompts(n_prompts=n_prompts)
    audio_values, sr = prompt_musicgen(prompts)
    save_musicgen_outputs(prompts, audio_values, output_dir, sr)


if __name__ == "__main__":
    main()
