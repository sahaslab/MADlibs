import os
import dataclasses
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler
from madlibs.generation.base import MusicGenerator
from typing import Any


@dataclasses.dataclass
class ACEStepParams:
    lyrics: str = "[Instrumental]"
    instrumental: bool = True
    duration: float = 90.0
    inference_steps: int = 8
    guidance_scale: float = 7.0
    seed: int = -1
    thinking: bool = False
    use_cot_metas: bool = False
    use_cot_caption: bool = False
    use_cot_language: bool = False


@dataclasses.dataclass
class ACEStepGenerationConfig:
    batch_size: int = 1
    audio_format: str = "wav"
    use_random_seed: bool = True
    seeds: list[int] | None = None
    lm_batch_chunk_size: int = 8


class ACEStep(MusicGenerator):
    def __init__(
        self,
        checkpoint_dir: str,
        config_path: str = "acestep-v15-turbo",
        lm_model_path: str = "acestep-5Hz-lm-0.6B",
        llm_backend: str = "vllm",
        init_llm: bool = True,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        params: dict[str, Any] | ACEStepParams | None = None,
        generation_config: dict[str, Any] | ACEStepGenerationConfig | None = None,
        num_workers: int = 1,
        device: str = "cuda",
        **base_kwargs: Any,
    ):
        super().__init__(num_workers=num_workers, device=device, **base_kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path
        self.lm_model_path = lm_model_path
        self.llm_backend = llm_backend
        self.init_llm = init_llm
        self.compile_model = compile_model
        self.offload_to_cpu = offload_to_cpu
        self.offload_dit_to_cpu = offload_dit_to_cpu

        if params is None:
            self.params = ACEStepParams()
        elif isinstance(params, ACEStepParams):
            self.params = params
        else:
            self.params = ACEStepParams(**params)

        if generation_config is None:
            self.generation_config = ACEStepGenerationConfig()
        elif isinstance(generation_config, ACEStepGenerationConfig):
            self.generation_config = generation_config
        else:
            self.generation_config = ACEStepGenerationConfig(**generation_config)

        self._dit_handler = None
        self._llm_handler = None

    def _get_init_kwargs(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "config_path": self.config_path,
            "lm_model_path": self.lm_model_path,
            "llm_backend": self.llm_backend,
            "init_llm": self.init_llm,
            "compile_model": self.compile_model,
            "offload_to_cpu": self.offload_to_cpu,
            "offload_dit_to_cpu": self.offload_dit_to_cpu,
            "params": dataclasses.asdict(self.params),
            "generation_config": dataclasses.asdict(self.generation_config),
            **self._base_init_kwargs(),
        }

    def _generate_audio(self, prompt: str, save_path: str) -> str:
        self._ensure_model()
        assert self._dit_handler is not None

        params = GenerationParams(**dataclasses.asdict(self.params), caption=prompt)
        config = GenerationConfig(**dataclasses.asdict(self.generation_config))
        result = generate_music(
            self._dit_handler,
            self._llm_handler,
            params,
            config,
            save_dir=save_path,
        )
        if not result.success:
            raise RuntimeError(
                f"ACE-Step generation failed: {result.error or result.status_message}"
            )

        if result.audios and result.audios[0].get("path"):
            return str(result.audios[0]["path"])
        return save_path

    def _ensure_model(self) -> None:
        if self._dit_handler is not None:
            return

        dit_handler = AceStepHandler()
        status_msg, success = dit_handler.initialize_service(
            project_root=self.checkpoint_dir,
            config_path=self.config_path,
            device=str(self.device),
            compile_model=self.compile_model,
            offload_to_cpu=self.offload_to_cpu,
            offload_dit_to_cpu=self.offload_dit_to_cpu,
        )
        if not success:
            raise RuntimeError(
                f"Failed to initialize ACE-Step DiT service: {status_msg}"
            )
        self._dit_handler = dit_handler

        if not self.init_llm:
            self._llm_handler = None
            return

        llm_handler = LLMHandler()
        checkpoint_dir = os.path.join(self.checkpoint_dir, "checkpoints")
        lm_status, lm_success = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=self.lm_model_path,
            backend=self.llm_backend,
            device=str(self.device),
            offload_to_cpu=self.offload_to_cpu,
            dtype=None,
        )
        if not lm_success:
            raise RuntimeError(
                f"Failed to initialize ACE-Step LLM service: {lm_status}"
            )
        self._llm_handler = llm_handler
