import asyncio
import dataclasses
import importlib
import os
import wave
from typing import Any
from madlibs.generation.base import MusicGenerator


@dataclasses.dataclass
class LyriaGenerationConfig:
    temperature: float | None = 1.1
    top_k: int | None = 40
    seed: int | None = None
    guidance: float | None = 4.0
    bpm: int | None = None
    density: float | None = None
    brightness: float | None = None
    scale: str | None = None
    mute_bass: bool | None = None
    mute_drums: bool | None = None
    only_bass_and_drums: bool | None = None
    music_generation_mode: str | None = None


class Lyria(MusicGenerator):
    def __init__(
        self,
        model_id: str = "models/lyria-realtime-exp",
        api_version: str = "v1alpha",
        api_key: str | None = None,
        api_key_env_vars: list[str] | None = None,
        output_seconds: float = 30.0,
        receive_timeout_seconds: float | None = None,
        prompt_weight: float = 1.0,
        sample_rate: int = 48000,
        channels: int = 2,
        sample_width_bytes: int = 2,
        audio_format: str = "wav",
        config: dict[str, Any] | LyriaGenerationConfig | None = None,
        num_workers: int = 1,
        device: str = "cpu",
        **base_kwargs: Any,
    ) -> None:
        super().__init__(num_workers=num_workers, device=device, **base_kwargs)
        self.model_id = model_id
        self.api_version = api_version
        self.api_key = api_key
        self.api_key_env_vars = api_key_env_vars or ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
        self.output_seconds = output_seconds
        self.prompt_weight = prompt_weight
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.audio_format = audio_format

        if config is None:
            self.config = LyriaGenerationConfig()
        elif isinstance(config, LyriaGenerationConfig):
            self.config = config
        else:
            self.config = LyriaGenerationConfig(**config)

        if receive_timeout_seconds is None:
            self.receive_timeout_seconds = max(30.0, self.output_seconds * 4.0)
        else:
            self.receive_timeout_seconds = receive_timeout_seconds

        if self.output_seconds <= 0:
            raise ValueError("output_seconds must be positive.")
        if self.receive_timeout_seconds <= 0:
            raise ValueError("receive_timeout_seconds must be positive.")
        if self.prompt_weight == 0:
            raise ValueError("prompt_weight must be non-zero.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        if self.sample_width_bytes <= 0:
            raise ValueError("sample_width_bytes must be positive.")
        if self.audio_format.lower() != "wav":
            raise ValueError("Lyria output must be saved as wav.")

    def _get_init_kwargs(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "api_version": self.api_version,
            "api_key": self.api_key,
            "api_key_env_vars": list(self.api_key_env_vars),
            "output_seconds": self.output_seconds,
            "receive_timeout_seconds": self.receive_timeout_seconds,
            "prompt_weight": self.prompt_weight,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "audio_format": self.audio_format,
            "config": dataclasses.asdict(self.config),
            **self._base_init_kwargs(),
        }

    def _generate_audio(self, prompt: str, save_path: str) -> str:
        return asyncio.run(
            self._generate_audio_async(prompt=prompt, save_path=save_path)
        )

    async def _generate_audio_async(self, prompt: str, save_path: str) -> str:
        genai_module, types_module = self._load_sdk()
        client = self._build_client(genai_module=genai_module)
        weighted_prompts = self._build_weighted_prompts(
            prompt=prompt,
            types_module=types_module,
        )
        music_config = self._build_music_generation_config(types_module=types_module)

        async with client.aio.live.music.connect(model=self.model_id) as session:
            await session.set_weighted_prompts(prompts=weighted_prompts)
            await session.set_music_generation_config(config=music_config)
            await session.play()
            try:
                audio_bytes = await self._collect_audio_bytes(session=session)
            finally:
                await session.stop()

        output_path = os.path.join(save_path, "output.%s" % (self.audio_format))
        self._write_wav(output_path=output_path, audio_bytes=audio_bytes)
        return output_path

    def _load_sdk(self) -> tuple[Any, Any]:
        try:
            genai_module = importlib.import_module("google.genai")
        except ImportError as error:
            raise ImportError(
                "google-genai is required for the Lyria generator. Install it with `pip install google-genai`."
            ) from error

        types_module = getattr(genai_module, "types", None)
        if types_module is None:
            raise ImportError("google.genai.types is unavailable in the installed SDK.")
        return genai_module, types_module

    def _build_client(self, genai_module: Any) -> Any:
        api_key = self._resolve_api_key()
        if api_key is None:
            raise ValueError(
                "Lyria requires an API key. Set one of %s or pass generator.api_key=..."
                % (self.api_key_env_vars)
            )
        return genai_module.Client(
            api_key=api_key,
            http_options={"api_version": self.api_version},
        )

    def _resolve_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key

        for env_var in self.api_key_env_vars:
            if env_var in os.environ and os.environ[env_var]:
                return os.environ[env_var]

        return None

    def _build_weighted_prompts(self, prompt: str, types_module: Any) -> list[Any]:
        return [
            types_module.WeightedPrompt(
                text=prompt,
                weight=self.prompt_weight,
            )
        ]

    def _build_music_generation_config(self, types_module: Any) -> Any:
        config_kwargs = dataclasses.asdict(self.config)
        if config_kwargs["scale"] is not None:
            config_kwargs["scale"] = self._resolve_enum(
                enum_cls=types_module.Scale,
                value=config_kwargs["scale"],
            )
        if config_kwargs["music_generation_mode"] is not None:
            config_kwargs["music_generation_mode"] = self._resolve_enum(
                enum_cls=types_module.MusicGenerationMode,
                value=config_kwargs["music_generation_mode"],
            )
        return types_module.LiveMusicGenerationConfig(**config_kwargs)

    def _resolve_enum(self, enum_cls: Any, value: str) -> Any:
        if not hasattr(enum_cls, value):
            raise ValueError(
                "Unsupported enum value `%s` for %s." % (value, enum_cls.__name__)
            )
        return getattr(enum_cls, value)

    async def _collect_audio_bytes(self, session: Any) -> bytes:
        target_bytes = self._target_audio_bytes()
        audio_buffer = bytearray()

        try:
            async with asyncio.timeout(self.receive_timeout_seconds):
                async for message in session.receive():
                    filtered_prompt = getattr(message, "filtered_prompt", None)
                    if filtered_prompt is not None:
                        filtered_text = getattr(filtered_prompt, "text", None) or ""
                        filtered_reason = (
                            getattr(filtered_prompt, "filtered_reason", None) or ""
                        )
                        raise RuntimeError(
                            "Lyria filtered prompt `%s` with reason `%s`."
                            % (filtered_text, filtered_reason)
                        )

                    server_content = getattr(message, "server_content", None)
                    if server_content is None:
                        continue

                    audio_chunks = getattr(server_content, "audio_chunks", None) or []
                    for chunk in audio_chunks:
                        chunk_bytes = getattr(chunk, "data", None) or b""
                        if len(chunk_bytes) == 0:
                            continue
                        audio_buffer.extend(chunk_bytes)
                        if len(audio_buffer) >= target_bytes:
                            return bytes(audio_buffer[:target_bytes])
        except TimeoutError as error:
            raise TimeoutError(
                "Timed out after %.1f seconds while collecting %.1f seconds of Lyria audio."
                % (self.receive_timeout_seconds, self.output_seconds)
            ) from error

        raise RuntimeError(
            "Lyria stream ended before %.1f seconds of audio were received."
            % (self.output_seconds)
        )

    def _target_audio_bytes(self) -> int:
        frame_count = max(1, int(round(self.output_seconds * self.sample_rate)))
        return frame_count * self.channels * self.sample_width_bytes

    def _write_wav(self, output_path: str, audio_bytes: bytes) -> None:
        with wave.open(output_path, "wb") as file_obj:
            file_obj.setnchannels(self.channels)
            file_obj.setsampwidth(self.sample_width_bytes)
            file_obj.setframerate(self.sample_rate)
            file_obj.writeframes(audio_bytes)
