import os
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from madlibs.generation.base import MusicGenerator
from typing import Any


class MusicGen(MusicGenerator):
    def __init__(
        self,
        model_id: str = "facebook/musicgen-large",
        guidance_scale: float = 3.0,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        audio_format: str = "wav",
        num_workers: int = 1,
        device: str = "cuda",
        **base_kwargs: Any,
    ):
        super().__init__(num_workers=num_workers, device=device, **base_kwargs)
        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.audio_format = audio_format
        self.processor = None
        self.model = None

    def _get_init_kwargs(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "guidance_scale": self.guidance_scale,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "audio_format": self.audio_format,
            **self._base_init_kwargs(),
        }

    def _generate_audio(self, prompt: str, save_path: str) -> str:
        self._ensure_model()
        assert self.processor is not None
        assert self.model is not None

        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(
            self.device
        )
        audio_values = self.model.generate(
            **inputs,
            do_sample=self.do_sample,
            guidance_scale=self.guidance_scale,
            max_new_tokens=self.max_new_tokens,
        )
        generated = audio_values.detach().cpu()
        sample_rate = self.model.config.audio_encoder.sampling_rate

        output_path = os.path.join(save_path, f"output.{self.audio_format}")
        sf.write(output_path, generated[0][0].numpy(), sample_rate)
        return output_path

    def _ensure_model(self) -> None:
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.model is None:
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_id
            ).to(self.device)

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
