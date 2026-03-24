from skey.key_detection import infer_key
from skey.key_detection import load_checkpoint, load_model_components
import torch
from madlibs.mir.base import MIRAnalysis
from typing import Dict
import numpy as np

class KeyDetector(MIRAnalysis):
    def __init__(
        self,
        model_path: str = "/dartfs/rc/lab/S/SinghN/noah/imPROMPTu/skey/skey.pt",
        device: str = "cpu",
    ) -> None:
        ckpt = load_checkpoint(model_path)
        self.pitch_hcqt, self.pitch_chromanet, self.pitch_crop_fn = (
            load_model_components(ckpt, device)
        )
        self.device = device

    def process_audio(self, audio: np.ndarray, sr: int) -> Dict:
        keys_dict = {}
        keys_dict["key_info"] = {}
        chunk_size = 30
        chunk_size_samples = int(sr * chunk_size)
        audio_mono = torch.mean(torch.Tensor(audio), dim=0).cpu()
        keys = []
        positions = []
        position = 0
        for chunk in _iterate_tensor_chunks(audio_mono, chunk_size_samples):
            key = infer_key(
                self.pitch_hcqt,
                self.pitch_chromanet,
                self.pitch_crop_fn,
                chunk,
                self.device,
            )
            keys.append(key)
            positions.append(position)
            position += chunk_size
        keys_dict["key_info"]["keys"] = keys
        keys_dict["key_info"]["positions"] = positions
        return keys_dict


def _iterate_tensor_chunks(tensor, chunk_size):
    n = tensor.size(0)

    for start in range(0, n, chunk_size):
        end = start + chunk_size
        chunk = tensor[start:end]
        yield chunk
