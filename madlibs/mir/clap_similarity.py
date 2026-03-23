import torch
import torch.nn.functional as F
import torchaudio
from transformers import (
    ClapProcessor,
    ClapTextModelWithProjection,
    ClapAudioModelWithProjection,
)
from madlibs.mir.base import MIRAnalysis
from typing import Tuple, Dict


class CLAPSimilarity(MIRAnalysis):
    def __init__(
        self,
        model_id: str = "laion/clap-htsat-fused",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.text_model = (
            ClapTextModelWithProjection.from_pretrained(model_id).to(self.device).eval()
        )
        self.audio_model = (
            ClapAudioModelWithProjection.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def encode_text(self, prompt: str) -> torch.Tensor:
        text_inputs = self.processor(text=[prompt], return_tensors="pt", padding=True)
        text_inputs = {
            k: v.to(self.device) for k, v in text_inputs.items() if hasattr(v, "to")
        }
        text_embeds = self.text_model(**text_inputs).text_embeds
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds

    @torch.inference_mode()
    def encode_audio_chunk(self, audio_48k: torch.Tensor) -> torch.Tensor:
        """
        audio_48k: mono waveform [time] at 48k
        """
        audio_inputs = self.processor(
            audios=[audio_48k.cpu().numpy()], return_tensors="pt"
        )
        audio_inputs = {
            k: v.to(self.device) for k, v in audio_inputs.items() if hasattr(v, "to")
        }
        audio_embeds = self.audio_model(**audio_inputs).audio_embeds
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

    def process_audio(
        self,
        audio_path: str,
        prompt: str,
        win_size: float = 10.0,
        hop_size: float = 5.0,
        aggregate: str = "mean",
    ) -> Dict:
        audio_48k, sr = _prepare_audio(audio_path)
        text_embed = self.encode_text(prompt)

        scores = []
        for chunk in _sliding_windows(
            audio_48k, sr, win_size=win_size, hop_size=hop_size
        ):
            audio_embed = self.encode_audio_chunk(chunk)
            score = (
                (text_embed @ audio_embed.T).squeeze().item()
            )  # cosine since normalized
            scores.append(score)

        if aggregate == "max":
            agg = max(scores)
        elif aggregate == "mean":
            agg = sum(scores) / len(scores)
        else:
            raise ValueError("aggregate must be 'mean' or 'max'")

        return {
            "aggregate_score": agg,
            "window_scores": scores,
        }


def _prepare_audio(path: str) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 48000:
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
        sr = 48000

    return waveform.squeeze(0), sr


def _sliding_windows(
    audio: torch.Tensor, sr: int, win_size: float = 10.0, hop_size: float = 5.0
):
    win = int(win_size * sr)
    hop = int(hop_size * sr)

    if audio.numel() <= win:
        yield audio
        return

    for start in range(0, audio.numel() - win + 1, hop):
        yield audio[start : start + win]

    # include tail if needed
    tail_start = audio.numel() - win
    if tail_start > 0 and (audio.numel() - win) % hop != 0:
        yield audio[tail_start : tail_start + win]
