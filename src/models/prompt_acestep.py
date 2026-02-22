from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
import torch
import json
from src.prompt.MadlibsPrompt import MadlibsPrompt
import os
import uuid
import argparse
import random


def generate_acestep(prompt, save_path):
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    model_path = "/dartfs/rc/home/t/f00814t/lab/noah/imPROMPTu/ACE-Step1.5/"
    # Initialize services
    dit_handler.initialize_service(
        project_root=model_path, config_path="acestep-v15-turbo", device="cuda"
    )

    llm_handler.initialize(
        checkpoint_dir=model_path,
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend="vllm",
        device="cuda",
    )

    # Configure generation parameters
    params = GenerationParams(caption=prompt, instrumental=True)

    # Configure generation settings
    config = GenerationConfig(
        batch_size=1,
        audio_format="wav",
    )

    # Generate music
    result = generate_music(
        dit_handler, llm_handler, params, config, save_dir=save_path
    )


def save_acestep_outputs(prompts, output_dir, cfg):
    run_id = uuid.uuid1()
    print(f"Saving outputs for run {str(run_id)}")
    run_path = os.path.join(output_dir, str(run_id))
    os.mkdir(run_path)

    for i, prompt in enumerate(prompts):
        write_path = os.path.join(run_path, f"prompt_{i}")
        os.mkdir(write_path)
        with open(os.path.join(write_path, "prompt.txt"), "w") as f:
            f.write(prompt)
        generate_acestep(prompt, write_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to write musicgen outputs",
        default="outputs",
    )
    parser.add_argument(
        "-v",
        "--vocab",
        help="File for vocabulary (for madlibs style prompt)",
        default="tests/prompt/text_files/parts_of_speech.txt",
    )
    parser.add_argument(
        "-t", "--type", help="Prompt Type (MadLibs, ...)", default="madlibs"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="location of acestep default configuration",
        default="src/models/cfg/ace_step.json",
    )
    parser.add_argument("-n", "--n_prompts", help="Number of Prompts", default="100")
    args = parser.parse_args()
    output_dir = args.output_dir
    vocab = args.vocab
    prompt_type = args.type
    model_cfg = args.cfg
    n_prompts = args.n_prompts
    prompt_gen = None
    if prompt_type == "madlibs":
        prompt_gen = MadlibsPrompt(part_of_speech_filename=vocab)
    prompts = prompt_gen.get_prompts(n_prompts=int(n_prompts))
    save_acestep_outputs(prompts, output_dir, model_cfg)


if __name__ == "__main__":
    main()
