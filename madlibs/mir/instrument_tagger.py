import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from typing import List, Dict
import ast
from madlibs.mir.base import MIRAnalysis


class InstrumentTagger(MIRAnalysis):
    def __init__(
        self,
        model_id: str = "nvidia/music-flamingo-hf",
        cache_dir: str = "/dartfs/rc/lab/S/SinghN/noah/.cache/huggingface/hub",
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,  # ensures weights are bf16 consistently
        )

        self.prompt = """You are an instrument-tagging system.
                            Task: identify all musical instruments present in the audio.
                            Output rules (must follow):

                            Output exactly one line.

                            Output only a comma-and-space separated list of instrument names (example: piano, bass guitar, drums).

                            No extra words, labels, punctuation (besides commas), explanations, or newlines.

                            Do not use “and”.

                            Use lowercase, singular instrument names.

                            Remove duplicates.
                            If no instruments are detected, output none."""

    def process_audio(self, audio_path: str) -> Dict:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "audio", "path": audio_path},
                    ],
                }
            ]
        ]
        return self.process_audio_batch(
            audio_paths=[audio_path], conversations=conversations
        )[0]

    def process_audio_batch(
        self, audio_paths: List[str], conversations: List[List[Dict]]
    ) -> List[Dict]:
        if len(audio_paths) != len(conversations):
            raise ValueError("audio_paths and conversations must have equal length")
        if len(audio_paths) == 0:
            return []

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "return_tensors": "pt",
                "padding": True,
            },
        )

        # Move to device, and cast float tensors to the model dtype (bf16),
        # while leaving integer tensors (input_ids, attention_mask) alone.
        device = self.model.device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                v = v.to(device)
                if v.is_floating_point():
                    v = v.to(dtype=self.model.dtype)
                inputs[k] = v

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=32, do_sample=False, num_beams=1
            )

        input_lengths = inputs["attention_mask"].sum(dim=1)
        decoded = []
        for row_idx, input_len in enumerate(input_lengths):
            generated_tokens = outputs[row_idx, int(input_len) :]
            decoded.append(
                self.processor.decode(generated_tokens, skip_special_tokens=True)
            )

        return [{"instruments": _instruments_output_to_list(text)} for text in decoded]


def _instruments_output_to_list(model_output: str) -> List[str]:
    """
    Convert model output into a Python list of instruments.

    Handles formats like:
    - "piano, bass guitar, drums"
    - ["bass", "guitar"]
    - '["bass", "guitar"]'
    - none
    """

    if not model_output:
        return []

    s = model_output.strip()

    # Handle "none"
    if s.lower() == "none":
        return []

    # Try parsing as a Python list (handles ["bass", "guitar"])
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass

    # Fallback: treat as comma-separated string
    items = [x.strip().lower() for x in s.split(",") if x.strip()]

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result
