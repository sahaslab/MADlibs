from acestep.data_sampler import DataSampler
from acestep.pipeline_ace_step import ACEStepPipeline
import torch
import json
from src.prompt.MadlibsPrompt import MadlibsPrompt
import os
import uuid
import argparse
import random


def generate_acestep(json_data, prompt, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACEStepPipeline(
        checkpoint_dir="/dartfs/rc/home/t/f00814t/lab/noah/imPROMPTu/ACE-Step/ACE-Step-v1-3.5B",
        dtype="bfloat16",
        torch_compile=True,
        cpu_offload=False,
        overlapped_decode=True,
    )
    json_data["prompt"] = prompt
    json_data["actual_seeds"] = [random.randint(100000, 999999)]

    (
        audio_duration,
        prompt,
        lyrics,
        infer_step,
        guidance_scale,
        scheduler_type,
        cfg_type,
        omega_scale,
        manual_seeds,
        guidance_interval,
        guidance_interval_decay,
        min_guidance_scale,
        use_erg_tag,
        use_erg_lyric,
        use_erg_diffusion,
        oss_steps,
        guidance_scale_text,
        guidance_scale_lyric,
    ) = sample_data(json_data)

    model(
        audio_duration=audio_duration,
        prompt=prompt,
        lyrics=lyrics,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        cfg_type=cfg_type,
        omega_scale=omega_scale,
        manual_seeds=manual_seeds,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
        min_guidance_scale=min_guidance_scale,
        use_erg_tag=use_erg_tag,
        use_erg_lyric=use_erg_lyric,
        use_erg_diffusion=use_erg_diffusion,
        oss_steps=oss_steps,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        save_path=save_path,
    )


def sample_data(json_data):
    return (
        json_data["audio_duration"],
        json_data["prompt"],
        json_data["lyrics"],
        json_data["infer_step"],
        json_data["guidance_scale"],
        json_data["scheduler_type"],
        json_data["cfg_type"],
        json_data["omega_scale"],
        ", ".join(map(str, json_data["actual_seeds"])),
        json_data["guidance_interval"],
        json_data["guidance_interval_decay"],
        json_data["min_guidance_scale"],
        json_data["use_erg_tag"],
        json_data["use_erg_lyric"],
        json_data["use_erg_diffusion"],
        ", ".join(map(str, json_data["oss_steps"])),
        json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
        (
            json_data["guidance_scale_lyric"]
            if "guidance_scale_lyric" in json_data
            else 0.0
        ),
    )


def save_acestep_outputs(prompts, output_dir, cfg):
    run_id = uuid.uuid1()
    print(f"Saving outputs for run {str(run_id)}")
    run_path = os.path.join(output_dir, str(run_id))
    os.mkdir(run_path)
    f = open(cfg)
    json_data = json.load(f)

    for i, prompt in enumerate(prompts):
        write_path = os.path.join(run_path, f"prompt_{i}")
        os.mkdir(write_path)
        with open(os.path.join(write_path, "prompt.txt"), "w") as f:
            f.write(prompt)
        generate_acestep(json_data, prompt, write_path)


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
