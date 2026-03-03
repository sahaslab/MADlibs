import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
        "diffrhythm"
    )
)
import dataclasses
import random
import numpy as np
import torch
import torchaudio
from einops import rearrange
from huggingface_hub import hf_hub_download
from muq import MuQMuLan
from diffrhythm.infer.infer_utils import (
    decode_audio,
    get_reference_latent,
    get_style_prompt,
    load_checkpoint
)
from diffrhythm.model import CFM, DiT
from madlibs.generation.base import MusicGenerator
from typing import Any


@dataclasses.dataclass
class DiffRhythmModelConfig:
    dim: int = 2048
    depth: int = 16
    heads: int = 32
    ff_mult: int = 4
    text_dim: int = 512
    conv_layers: int = 4
    mel_dim: int = 64
    text_num_embeds: int = 363


@dataclasses.dataclass
class DiffRhythmSamplingConfig:
    chunked: bool = False
    batch_infer_num: int = 1
    steps: int = 32
    cfg_strength: float = 4.0
    start_time: float = 0.0
    song_duration: float = 1.0
    audio_format: str = "wav"
    random_select_batch_output: bool = True


class DiffRhythm(MusicGenerator):
    def __init__(
        self,
        checkpoint_dir: str,
        audio_length: int = 95,
        repo_id_short: str = "ASLP-lab/DiffRhythm-1_2",
        repo_id_full: str = "ASLP-lab/DiffRhythm-1_2-full",
        cfm_filename: str = "cfm_model.pt",
        vae_repo_id: str = "ASLP-lab/DiffRhythm-vae",
        vae_filename: str = "vae_model.pt",
        muq_model_id: str = "OpenMuQ/MuQ-MuLan-large",
        negative_style_prompt_path: str = "diffrhythm/infer/example/vocal.npy",
        model: dict[str, Any] | DiffRhythmModelConfig | None = None,
        sampling: dict[str, Any] | DiffRhythmSamplingConfig | None = None,
        num_workers: int = 1,
        device: str = "cuda",
        **base_kwargs: Any
    ):
        super().__init__(num_workers=num_workers, device=device, **base_kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.audio_length = audio_length
        self.repo_id_short = repo_id_short
        self.repo_id_full = repo_id_full
        self.cfm_filename = cfm_filename
        self.vae_repo_id = vae_repo_id
        self.vae_filename = vae_filename
        self.muq_model_id = muq_model_id
        self.negative_style_prompt_path = negative_style_prompt_path

        if model is None:
            self.model = DiffRhythmModelConfig()
        elif isinstance(model, DiffRhythmModelConfig):
            self.model = model
        else:
            self.model = DiffRhythmModelConfig(**model)

        if sampling is None:
            self.sampling = DiffRhythmSamplingConfig()
        elif isinstance(sampling, DiffRhythmSamplingConfig):
            self.sampling = sampling
        else:
            self.sampling = DiffRhythmSamplingConfig(**sampling)

        self.max_frames = self._resolve_max_frames(audio_length=self.audio_length)

        self._cfm = None
        self._muq = None
        self._vae = None
        self._latent_prompt = None
        self._pred_frames = None
        self._negative_style_prompt = None

    def _get_init_kwargs(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "audio_length": self.audio_length,
            "repo_id_short": self.repo_id_short,
            "repo_id_full": self.repo_id_full,
            "cfm_filename": self.cfm_filename,
            "vae_repo_id": self.vae_repo_id,
            "vae_filename": self.vae_filename,
            "muq_model_id": self.muq_model_id,
            "negative_style_prompt_path": self.negative_style_prompt_path,
            "model": dataclasses.asdict(self.model),
            "sampling": dataclasses.asdict(self.sampling),
            **self._base_init_kwargs()
        }

    def _generate_audio(self, prompt: str, save_path: str) -> str:
        self._ensure_model()
        assert self._cfm is not None
        assert self._muq is not None
        assert self._vae is not None
        assert self._latent_prompt is not None
        assert self._pred_frames is not None
        assert self._negative_style_prompt is not None

        style_prompt = get_style_prompt(self._muq, prompt=prompt)
        text = torch.zeros((1, self.max_frames), dtype=torch.long, device=self.device)
        start_time = torch.tensor([self.sampling.start_time], dtype=torch.float16, device=self.device)
        song_duration = torch.tensor([self.sampling.song_duration], dtype=torch.float16, device=self.device)

        with torch.inference_mode():
            latents, _ = self._cfm.sample(
                cond=self._latent_prompt,
                text=text,
                duration=self.max_frames,
                style_prompt=style_prompt,
                max_duration=self.max_frames,
                song_duration=song_duration,
                negative_style_prompt=self._negative_style_prompt,
                steps=self.sampling.steps,
                cfg_strength=self.sampling.cfg_strength,
                start_time=start_time,
                latent_pred_segments=self._pred_frames,
                batch_infer_num=self.sampling.batch_infer_num
            )

            outputs = []
            for latent in latents:
                latent = latent.to(torch.float32).transpose(1, 2)
                output = decode_audio(latent, self._vae, chunked=self.sampling.chunked)
                output = rearrange(output, "b d n -> d (b n)").to(torch.float32)
                peak = torch.max(torch.abs(output))
                if peak > 0:
                    output = output.div(peak)
                output = output.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                outputs.append(output)

        selected = outputs[0]
        if self.sampling.random_select_batch_output and len(outputs) > 1:
            selected = random.choice(outputs)

        output_path = os.path.join(save_path, f"output.{self.sampling.audio_format}")
        torchaudio.save(output_path, selected, sample_rate=44100)
        return output_path

    def _ensure_model(self) -> None:
        if self._cfm is not None:
            return

        repo_id = self.repo_id_short if self.max_frames == 2048 else self.repo_id_full
        cfm_ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=self.cfm_filename,
            cache_dir=self.checkpoint_dir
        )

        cfm = CFM(
            transformer=DiT(**dataclasses.asdict(self.model), max_frames=self.max_frames),
            num_channels=self.model.mel_dim,
            max_frames=self.max_frames,
        ).to(self.device)
        self._cfm = load_checkpoint(cfm, cfm_ckpt_path, device=str(self.device), use_ema=False)

        self._muq = MuQMuLan.from_pretrained(self.muq_model_id).to(self.device).eval()

        vae_ckpt_path = hf_hub_download(
            repo_id=self.vae_repo_id,
            filename=self.vae_filename,
            cache_dir=self.checkpoint_dir
        )
        self._vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(self.device)

        self._negative_style_prompt = torch.from_numpy(np.load(self.negative_style_prompt_path)).to(self.device).half()

        self._latent_prompt, self._pred_frames = get_reference_latent(
            str(self.device),
            self.max_frames,
            False,
            None,
            None,
            self._vae
        )

    def _resolve_max_frames(self, audio_length: int) -> int:
        if audio_length == 95:
            return 2048
        if 96 <= audio_length <= 285:
            return 6144
        raise ValueError(
            f"Invalid audio_length: {audio_length}. "
            "Supported values are exactly 95 or any value between 96 and 285 (inclusive)."
        )

    def _estimate_worker_vram_bytes(self, probe_prompt: str | None = None) -> int | None:
        if self.device.type != "cuda":
            return None
        if probe_prompt:
            return self._estimate_peak_generation_vram_bytes(prompt=probe_prompt)

        self._ensure_model()
        assert self._cfm is not None
        assert self._muq is not None
        assert self._vae is not None

        model_modules = [self._cfm, self._muq]
        param_bytes = sum(
            param.numel() * param.element_size()
            for module in model_modules
            for param in module.parameters()
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size()
            for module in model_modules
            for buffer in module.buffers()
        )

        # JIT modules expose parameters/buffers via iterators too.
        param_bytes += sum(param.numel() * param.element_size() for param in self._vae.parameters())
        buffer_bytes += sum(buffer.numel() * buffer.element_size() for buffer in self._vae.buffers())

        return param_bytes + buffer_bytes
