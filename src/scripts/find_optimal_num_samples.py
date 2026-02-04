from src.prompt.MadlibsPrompt import MadlibsPrompt
import torch
import laion_clap
from apricot import SaturatedCoverageSelection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prompt_gen = MadlibsPrompt(part_of_speech_filename="./src/prompt/text_files/parts_of_speech.txt")
    prompts = prompt_gen.get_prompts(10000)

    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        device="cpu"
    )
    model.load_ckpt()

    with torch.no_grad():
        embeddings = model.get_text_embedding(prompts)

    S = cosine_similarity(embeddings)
    m = np.maximum(S, 0.0)

    selector = SaturatedCoverageSelection(n_samples=1000)
    selector.fit(m)

    rank = selector.ranking
    print(rank)

    gains = selector.gains

    cumulative = np.cumsum(gains)

    plt.plot(cumulative)
    plt.xlabel("Subset size")
    plt.ylabel("Facility Location objective")
    plt.show()