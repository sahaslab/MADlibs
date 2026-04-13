import base64
import dataclasses
import importlib
import io
import json
import os
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
    NATIVE_SAMPLE_RATE = 44100
    NATIVE_CHANNELS = 2
    MAX_PRO_SECONDS = 240.0
    RESPONSE_TEXT_FILENAME = "response.txt"
    RESPONSE_DEBUG_FILENAME = "response_debug.json"

    def __init__(
        self,
        model_id: str = "lyria-3-pro-preview",
        api_version: str = "v1beta",
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
        self.api_key_env_vars = api_key_env_vars or [
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ]
        self.output_seconds = output_seconds
        self.receive_timeout_seconds = receive_timeout_seconds
        self.prompt_weight = prompt_weight
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.audio_format = audio_format.lower()

        if config is None:
            self.config = LyriaGenerationConfig()
        elif isinstance(config, LyriaGenerationConfig):
            self.config = config
        else:
            self.config = LyriaGenerationConfig(**config)

        self._validate_config()

    def _validate_config(self) -> None:
        if self.output_seconds <= 0:
            raise ValueError("output_seconds must be positive.")
        if (
            self.receive_timeout_seconds is not None
            and self.receive_timeout_seconds <= 0
        ):
            raise ValueError("receive_timeout_seconds must be positive.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        if self.sample_width_bytes <= 0:
            raise ValueError("sample_width_bytes must be positive.")
        if self.audio_format not in {"wav", "mp3"}:
            raise ValueError("audio_format must be either `wav` or `mp3`.")
        if self.prompt_weight == 0:
            raise ValueError("prompt_weight must be non-zero.")
        if (
            self.config.only_bass_and_drums
            and (self.config.mute_bass or self.config.mute_drums)
        ):
            raise ValueError(
                "only_bass_and_drums cannot be combined with mute_bass or mute_drums."
            )
        if (self.config.music_generation_mode or "").upper() == "VOCALIZATION":
            raise ValueError("The Lyria 3 generator only supports instrumental audio.")
        if self._is_clip_model() and abs(self.output_seconds - 30.0) > 1e-6:
            raise ValueError(
                "lyria-3-clip-preview always returns 30-second clips. "
                "Use lyria-3-pro-preview for other durations."
            )
        if self._is_pro_model() and self.output_seconds > self.MAX_PRO_SECONDS:
            raise ValueError("lyria-3-pro-preview supports tracks up to 240 seconds.")

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
        lyrics, audio_bytes, source_mime = self._request_track(
            prompt=prompt,
            save_path=save_path,
        )

        if lyrics:
            self._write_response_text(save_path=save_path, lyrics=lyrics)

        output_path = os.path.join(save_path, f"output.{self.audio_format}")
        self._write_audio_file(
            output_path=output_path,
            audio_bytes=audio_bytes,
            source_mime=source_mime,
        )
        return output_path

    def _request_track(self, prompt: str, save_path: str) -> tuple[list[str], bytes, str]:
        genai_module, types_module = self._load_sdk()
        client = self._build_client(genai_module=genai_module, types_module=types_module)

        try:
            response = client.models.generate_content(
                model=self.model_id,
                contents=self._build_generation_prompt(prompt),
                config=self._build_generation_config(types_module),
            )
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()

        return self._parse_response(response=response, save_path=save_path)

    def _load_sdk(self) -> tuple[Any, Any]:
        try:
            genai_module = importlib.import_module("google.genai")
        except ImportError as error:
            raise ImportError(
                "google-genai is required for the Lyria generator. "
                "Install it with `pip install google-genai`."
            ) from error

        types_module = getattr(genai_module, "types", None)
        if types_module is None:
            raise ImportError("google.genai.types is unavailable in the installed SDK.")
        return genai_module, types_module

    def _build_client(self, genai_module: Any, types_module: Any) -> Any:
        api_key = self._resolve_api_key()
        if api_key is None:
            raise ValueError(
                "Lyria requires an API key. Set one of %s or pass generator.api_key=..."
                % (self.api_key_env_vars)
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        http_options_cls = getattr(types_module, "HttpOptions", None)
        if self.api_version:
            if http_options_cls is None:
                client_kwargs["http_options"] = {"api_version": self.api_version}
            else:
                client_kwargs["http_options"] = http_options_cls(
                    api_version=self.api_version
                )
        return genai_module.Client(**client_kwargs)

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key

        for env_var in self.api_key_env_vars:
            value = os.environ.get(env_var)
            if value:
                return value
        return None

    def _build_generation_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        instructions = [
            self._duration_instruction(),
            "Generate instrumental music only.",
            "Do not include vocals, lyrics, singing, chanting, choir, or spoken word.",
        ]

        if self.config.only_bass_and_drums:
            instructions.append("Use only bass and drums/percussion.")
        else:
            if self.config.mute_bass:
                instructions.append("Do not use bass instruments.")
            if self.config.mute_drums:
                instructions.append("Do not use drums or percussion.")

        if self.config.bpm is not None:
            instructions.append(f"Target tempo: about {self.config.bpm} BPM.")
        if self.config.scale:
            instructions.append(
                f"Prefer the tonal center or scale {self._humanize_scale(self.config.scale)}."
            )
        if self.config.density is not None:
            instructions.append(
                f"Arrangement density: {self._describe_density(self.config.density)}."
            )
        if self.config.brightness is not None:
            instructions.append(
                f"Tonal brightness: {self._describe_brightness(self.config.brightness)}."
            )

        mode = (self.config.music_generation_mode or "").upper()
        if mode == "QUALITY":
            instructions.append("Prioritize polish and cohesion.")
        elif mode == "DIVERSITY":
            instructions.append("Favor variety and evolving sections.")

        return "\n".join(
            [
                "Follow all of these instructions when creating the track.",
                *instructions,
                "Creative direction:",
                prompt,
            ]
        )

    def _duration_instruction(self) -> str:
        if self._is_clip_model():
            return "Create a 30-second clip."

        return (
            "Create a complete track that lasts about %s and resolves naturally by %s."
            % (
                self._format_duration(self.output_seconds),
                self._format_timestamp(self.output_seconds),
            )
        )

    def _build_generation_config(self, types_module: Any) -> Any:
        config_kwargs = {"response_modalities": ["AUDIO", "TEXT"]}
        config_cls = getattr(types_module, "GenerateContentConfig", None)
        if config_cls is None:
            return config_kwargs
        return config_cls(**config_kwargs)

    def _parse_response(
        self, response: Any, save_path: str
    ) -> tuple[list[str], bytes, str]:
        lyrics: list[str] = []
        audio_bytes: bytes | None = None
        audio_mime: str | None = None

        for part in self._iter_response_parts(response):
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                lyrics.append(text.strip())
                continue

            inline_data = getattr(part, "inline_data", None) or getattr(
                part, "inlineData", None
            )
            if inline_data is None:
                continue

            data = getattr(inline_data, "data", None)
            if data is None:
                continue

            if isinstance(data, str):
                data = base64.b64decode(data)

            audio_bytes = bytes(data)
            audio_mime = getattr(inline_data, "mime_type", None) or getattr(
                inline_data, "mimeType", None
            )

        if audio_bytes is None:
            self._write_response_debug(save_path=save_path, response=response)
            raise RuntimeError(
                "Lyria 3 returned no audio data for the prompt. "
                "Saved response details to %s."
                % (os.path.join(save_path, self.RESPONSE_DEBUG_FILENAME))
            )

        return lyrics, audio_bytes, audio_mime or "audio/mp3"

    def _iter_response_parts(self, response: Any) -> list[Any]:
        parts = list(getattr(response, "parts", None) or [])
        if parts:
            return parts

        candidates = list(getattr(response, "candidates", None) or [])
        if not candidates:
            return []

        content = getattr(candidates[0], "content", None)
        return list(getattr(content, "parts", None) or [])

    def _write_response_debug(self, save_path: str, response: Any) -> None:
        payload = self._serialize_response(response)
        output_path = os.path.join(save_path, self.RESPONSE_DEBUG_FILENAME)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=True, default=repr)
            file_obj.write("\n")

    def _serialize_response(self, response: Any) -> Any:
        for method_name in ("model_dump", "to_json_dict"):
            method = getattr(response, method_name, None)
            if callable(method):
                try:
                    return method()
                except TypeError:
                    continue
        return repr(response)

    def _write_response_text(self, save_path: str, lyrics: list[str]) -> None:
        output_path = os.path.join(save_path, self.RESPONSE_TEXT_FILENAME)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n\n".join(lyrics).strip() + "\n")

    def _write_audio_file(
        self, output_path: str, audio_bytes: bytes, source_mime: str
    ) -> None:
        if self.audio_format == "mp3":
            self._write_mp3(output_path=output_path, audio_bytes=audio_bytes)
            return

        waveform, sample_rate = self._decode_audio_bytes(
            audio_bytes=audio_bytes,
            source_mime=source_mime,
        )
        waveform = self._convert_channels(waveform)
        if sample_rate != self.sample_rate:
            torchaudio = importlib.import_module("torchaudio")
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
            sample_rate = self.sample_rate

        self._write_wav(
            output_path=output_path,
            waveform=waveform,
            sample_rate=sample_rate,
        )

    def _write_mp3(self, output_path: str, audio_bytes: bytes) -> None:
        if (
            self.sample_rate != self.NATIVE_SAMPLE_RATE
            or self.channels != self.NATIVE_CHANNELS
        ):
            raise ValueError(
                "mp3 output can only be used with native 44.1kHz stereo settings. "
                "Use audio_format=wav for sample-rate or channel conversion."
            )
        with open(output_path, "wb") as file_obj:
            file_obj.write(audio_bytes)

    def _decode_audio_bytes(
        self, audio_bytes: bytes, source_mime: str
    ) -> tuple[Any, int]:
        try:
            torchaudio = importlib.import_module("torchaudio")
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise ImportError(
                "torchaudio and torch are required to decode Lyria 3 audio."
            ) from error

        audio_buffer = io.BytesIO(audio_bytes)
        format_hint = "mp3" if source_mime in {"audio/mp3", "audio/mpeg"} else "wav"
        try:
            waveform, sample_rate = torchaudio.load(audio_buffer, format=format_hint)
        except TypeError:
            audio_buffer.seek(0)
            waveform, sample_rate = torchaudio.load(audio_buffer)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.to(torch.float32), int(sample_rate)

    def _convert_channels(self, waveform: Any) -> Any:
        current_channels = int(waveform.shape[0])
        if current_channels == self.channels:
            return waveform
        if self.channels == 1:
            return waveform.mean(dim=0, keepdim=True)
        if current_channels == 1:
            return waveform.repeat(self.channels, 1)
        if current_channels >= self.channels:
            return waveform[: self.channels, :]

        mono = waveform.mean(dim=0, keepdim=True)
        return mono.repeat(self.channels, 1)

    def _write_wav(self, output_path: str, waveform: Any, sample_rate: int) -> None:
        try:
            soundfile = importlib.import_module("soundfile")
        except ImportError as error:
            raise ImportError(
                "soundfile is required to write wav output for the Lyria generator."
            ) from error

        soundfile.write(
            output_path,
            waveform.transpose(0, 1).cpu().numpy(),
            sample_rate,
            format="WAV",
            subtype=self._wav_subtype(),
        )

    def _wav_subtype(self) -> str:
        if self.sample_width_bytes == 1:
            return "PCM_U8"
        if self.sample_width_bytes == 2:
            return "PCM_16"
        if self.sample_width_bytes == 3:
            return "PCM_24"
        if self.sample_width_bytes == 4:
            return "PCM_32"
        raise ValueError(
            "Unsupported sample_width_bytes `%s` for wav output."
            % (self.sample_width_bytes)
        )

    def _describe_density(self, density: float) -> str:
        if density <= 0.2:
            return "very sparse"
        if density <= 0.4:
            return "sparse"
        if density <= 0.6:
            return "balanced"
        if density <= 0.8:
            return "dense"
        return "very dense"

    def _describe_brightness(self, brightness: float) -> str:
        if brightness <= 0.2:
            return "very dark"
        if brightness <= 0.4:
            return "warm and restrained"
        if brightness <= 0.6:
            return "balanced"
        if brightness <= 0.8:
            return "bright"
        return "very bright and shimmering"

    def _humanize_scale(self, scale: str) -> str:
        return scale.replace("_", " ").title()

    def _format_duration(self, seconds: float) -> str:
        total_seconds = max(1, int(round(seconds)))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        if minutes == 0:
            return f"{remaining_seconds} seconds"
        if remaining_seconds == 0:
            return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"
        minute_label = "minute" if minutes == 1 else "minutes"
        second_label = "second" if remaining_seconds == 1 else "seconds"
        return f"{minutes} {minute_label} {remaining_seconds} {second_label}"

    def _format_timestamp(self, seconds: float) -> str:
        total_seconds = max(1, int(round(seconds)))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        return f"{minutes}:{remaining_seconds:02d}"

    def _is_clip_model(self) -> bool:
        return "lyria-3-clip" in self.model_id.lower()

    def _is_pro_model(self) -> bool:
        return "lyria-3-pro" in self.model_id.lower()
