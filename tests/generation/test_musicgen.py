import os
import sys
import types

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

if "soundfile" not in sys.modules:
    sys.modules["soundfile"] = types.SimpleNamespace(
        write=lambda *args, **kwargs: None
    )
if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.SimpleNamespace(
        AutoProcessor=object,
        MusicgenForConditionalGeneration=object,
    )
if "tqdm.auto" not in sys.modules:
    fake_tqdm_module = types.SimpleNamespace(
        tqdm=lambda iterable=None, **kwargs: iterable
    )
    sys.modules["tqdm"] = types.SimpleNamespace(auto=fake_tqdm_module)
    sys.modules["tqdm.auto"] = fake_tqdm_module

from madlibs.generation.musicgen import MusicGen
from madlibs.generation import musicgen as musicgen_module


def test_musicgen_duration_defaults_to_30_seconds():
    generator = MusicGen(
        duration_seconds=30.0,
        max_new_tokens=None,
        device="cpu",
        torch_dtype="float32",
    )

    assert generator.max_new_tokens == 1500


def test_musicgen_batch_generation_writes_one_file_per_prompt(tmp_path, monkeypatch):
    recorded_writes = []

    class FakeInputs(dict):
        def to(self, device):
            self["device"] = str(device)
            return self

    class FakeProcessor:
        def __call__(self, text, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            return FakeInputs({"text": list(text)})

    class FakeAudioEncoder:
        sampling_rate = 32000

    class FakeConfig:
        audio_encoder = FakeAudioEncoder()

    class FakeModel:
        config = FakeConfig()

        def generate(self, **kwargs):
            batch_size = len(kwargs["text"])
            return torch.arange(batch_size * 8, dtype=torch.bfloat16).reshape(
                batch_size, 1, 8
            )

    def _fake_write(path, audio, sample_rate):
        recorded_writes.append((path, audio.tolist(), sample_rate))

    monkeypatch.setattr(musicgen_module.sf, "write", _fake_write)

    generator = MusicGen(
        batch_size=2,
        duration_seconds=30.0,
        max_new_tokens=None,
        device="cpu",
        torch_dtype="float32",
    )
    generator.processor = FakeProcessor()
    generator.model = FakeModel()

    run_dir = generator.generate(
        prompts=["first prompt", "second prompt", "third prompt"],
        output_dir=str(tmp_path),
    )

    assert len(recorded_writes) == 3
    assert all(sample_rate == 32000 for _, _, sample_rate in recorded_writes)
    assert os.path.exists(os.path.join(run_dir, "prompt_0", "prompt.txt"))
    assert os.path.exists(os.path.join(run_dir, "prompt_1", "prompt.txt"))
    assert os.path.exists(os.path.join(run_dir, "prompt_2", "prompt.txt"))
