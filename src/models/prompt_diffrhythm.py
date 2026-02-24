import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "diffrhythm")))
import time
import random
import argparse
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from einops import rearrange
from muq import MuQMuLan
from diffrhythm.model import DiT, CFM
from diffrhythm.infer.infer_utils import (
    decode_audio,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    load_checkpoint
)


DIFFRHYTHM_CFG = {
    "model_type": "diffrhythm",
    "model": {
        "dim": 2048,
        "depth": 16,
        "heads": 32,
        "ff_mult": 4,
        "text_dim": 512,
        "conv_layers": 4,
        "mel_dim": 64,
        "text_num_embeds": 363
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-prompt",
        type=str,
        help="reference prompt as style prompt for target song",
        required=False,
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="whether to use chunked decoding",
    )
    parser.add_argument(
        "--audio-length",
        type=int,
        default=95, # choices=[95, 285],
        help="length of generated song, upported values are exactly 95 or any value between 96 and 285 (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="output directory fo generated song",
    )
    parser.add_argument(
        "--batch-infer-num",
        type=int,
        default=1,
        required=False,
        help="number of songs per batch",
    )
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    audio_length = args.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif 95 < audio_length <= 285:
        max_frames = 6144
    else:
        raise ValueError(
            f"Invalid audio_length: {audio_length}. "
            "Supported values are exactly 95 or any value between 96 and 285 (inclusive)."
        )

    cfm, muq, vae = prepare_model(max_frames, device)
    style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)
    negative_style_prompt = torch.from_numpy(np.load("diffrhythm/infer/example/vocal.npy")).to(device).half()
    latent_prompt, pred_frames = get_reference_latent(device, max_frames, False, None, None, vae)

    s_t = time.time()
    generated_songs = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=torch.zeros((1, max_frames), dtype=torch.long).to(device),
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=torch.tensor([0.0], dtype=torch.float16).to(device),
        pred_frames=pred_frames,
        chunked=args.chunked,
        batch_infer_num=args.batch_infer_num,
        song_duration=torch.tensor([1.0], dtype=torch.float16).to(device)
    )
    e_t = time.time() - s_t
    print(f"inference cost {e_t:.2f} seconds")

    generated_song = random.sample(generated_songs, 1)[0]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "output.wav")
    torchaudio.save(output_path, generated_song, sample_rate=44100)


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    song_duration,
    chunked=False,
):
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            max_duration=duration,
            song_duration=song_duration,
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num
        )

        outputs = []
        for latent in latents:
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)  # [b d t]

            output = decode_audio(latent, vae_model, chunked=chunked)

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            # Peak normalize, clip, convert to int16, and save to file
            output = (
                output.to(torch.float32)
                .div(torch.max(torch.abs(output)))
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )
            outputs.append(output)

        return outputs


def prepare_model(max_frames: int, device: str, repo_id: str = "ASLP-lab/DiffRhythm-1_2") -> tuple:
    # prepare cfm model
    if max_frames == 2048:
        repo_id = "ASLP-lab/DiffRhythm-1_2"
    else:
        repo_id = "ASLP-lab/DiffRhythm-1_2-full"

    dit_ckpt_path = hf_hub_download(
        repo_id=repo_id, filename="cfm_model.pt", cache_dir="./pretrained"
    )

    cfm = CFM(
        transformer=DiT(**DIFFRHYTHM_CFG["model"], max_frames=max_frames),
        num_channels=DIFFRHYTHM_CFG["model"]["mel_dim"],
        max_frames=max_frames
    )
    cfm = cfm.to(device)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)

    # prepare muq
    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    muq = muq.to(device).eval()

    # prepare vae
    vae_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt"
    )
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)

    return cfm, muq, vae


if __name__ == "__main__":
    main()
