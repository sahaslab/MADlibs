import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from madlibs.mir.clap_similarity import CLAPSimilarity
import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CLAPSimilarity(
        model_id="laion/clap-htsat-fused",
        device=device,
    )
    aud_file = "tests/mir/lyria_test.mp3"
    prompt = "a mellow lo-fi beat with vinyl crackle"
    similarity = scorer.process_audio(audio_path=aud_file, prompt=prompt)
    print(similarity)


if __name__ == "__main__":
    main()
