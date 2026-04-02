import math
import os
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from madlibs.generation.base import MusicGenerator
from tqdm.auto import tqdm
from typing import Any


class MusicGen(MusicGenerator):
    TOKENS_PER_SECOND = 50
    MAX_DURATION_SECONDS = 30.0

    def __init__(
        self,
        model_id: str = "facebook/musicgen-large",
        guidance_scale: float = 3.0,
        max_new_tokens: int | None = None,
        duration_seconds: float = 30.0,
        do_sample: bool = True,
        audio_format: str = "wav",
        batch_size: int = 1,
        torch_dtype: str | None = "float16",
        num_workers: int = 1,
        device: str = "cuda",
        **base_kwargs: Any,
    ):
        super().__init__(num_workers=num_workers, device=device, **base_kwargs)
        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self.duration_seconds = duration_seconds
        self.max_new_tokens = self._resolve_max_new_tokens(
            max_new_tokens=max_new_tokens, duration_seconds=duration_seconds
        )
        self.do_sample = do_sample
        self.audio_format = audio_format
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.processor = None
        self.model = None

    def _get_init_kwargs(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "guidance_scale": self.guidance_scale,
            "max_new_tokens": self.max_new_tokens,
            "duration_seconds": self.duration_seconds,
            "do_sample": self.do_sample,
            "audio_format": self.audio_format,
            "batch_size": self.batch_size,
            "torch_dtype": self.torch_dtype,
            **self._base_init_kwargs(),
        }

    def _generate_audio(self, prompt: str, save_path: str) -> str:
        self._ensure_model()
        assert self.processor is not None
        assert self.model is not None

        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            audio_values = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                guidance_scale=self.guidance_scale,
                max_new_tokens=self.max_new_tokens,
            )
        generated = audio_values.detach().to(torch.float32).cpu()
        sample_rate = self.model.config.audio_encoder.sampling_rate

        output_path = os.path.join(save_path, f"output.{self.audio_format}")
        sf.write(output_path, generated[0][0].numpy(), sample_rate)
        return output_path

    def generate(self, prompts: list[str], output_dir: str) -> str:
        run_path = self._create_run_path(output_dir=output_dir)
        batch_size = max(1, self.batch_size)
        self._maybe_log_gpu_usage(worker_count=1)

        for batch_start in tqdm(
            range(0, len(prompts), batch_size), desc="Generating audio"
        ):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            self._generate_batch(
                prompts=batch_prompts, run_path=run_path, start_idx=batch_start
            )
        return run_path

    def _generate_batch(
        self, prompts: list[str], run_path: str, start_idx: int
    ) -> None:
        self._ensure_model()
        assert self.processor is not None
        assert self.model is not None

        save_paths = []
        for offset, prompt in enumerate(prompts):
            save_path = self._create_prompt_path(
                run_path=run_path, idx=start_idx + offset
            )
            with open(
                os.path.join(save_path, "prompt.txt"), "w", encoding="utf-8"
            ) as file_obj:
                file_obj.write(prompt)
            save_paths.append(save_path)

        inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            audio_values = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                guidance_scale=self.guidance_scale,
                max_new_tokens=self.max_new_tokens,
            )

        generated = audio_values.detach().to(torch.float32).cpu()
        sample_rate = self.model.config.audio_encoder.sampling_rate
        for idx, save_path in enumerate(save_paths):
            output_path = os.path.join(save_path, f"output.{self.audio_format}")
            sf.write(output_path, generated[idx][0].numpy(), sample_rate)

    def _ensure_model(self) -> None:
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.model is None:
            model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
            resolved_dtype = self._resolve_torch_dtype()
            if resolved_dtype is not None:
                model_kwargs["torch_dtype"] = resolved_dtype

            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            ).to(self.device)
            self.model.eval()

    def _resolve_max_new_tokens(
        self, max_new_tokens: int | None, duration_seconds: float
    ) -> int:
        if max_new_tokens is not None:
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive when provided.")
            return max_new_tokens

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        if duration_seconds > self.MAX_DURATION_SECONDS:
            raise ValueError(
                f"MusicGen supports up to {self.MAX_DURATION_SECONDS:.0f} seconds per clip."
            )
        return math.ceil(duration_seconds * self.TOKENS_PER_SECOND)

    def _resolve_torch_dtype(self) -> torch.dtype | None:
        if self.torch_dtype is None:
            return torch.float16 if self.device.type == "cuda" else None

        dtype_name = self.torch_dtype.lower()
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(
            f"Unsupported torch_dtype '{self.torch_dtype}'. "
            "Expected one of: float16, bfloat16, float32."
        )

    def _estimate_worker_vram_bytes(
        self, probe_prompt: str | None = None
    ) -> int | None:
        if self.device.type != "cuda":
            return None
        if probe_prompt:
            return self._estimate_peak_generation_vram_bytes(prompt=probe_prompt)

        self._ensure_model()
        assert self.model is not None

        model_param_bytes = sum(
            param.numel() * param.element_size() for param in self.model.parameters()
        )
        model_buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in self.model.buffers()
        )
        return model_param_bytes + model_buffer_bytes
