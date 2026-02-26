from madlibs.prompt.madlibs import MadlibsPrompt
import torch
import laion_clap
from apricot import SaturatedCoverageSelection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import time

prompt_gen = MadlibsPrompt(
    part_of_speech_filename="./src/prompt/text_files/parts_of_speech.txt"
)
prompts = prompt_gen.get_prompts(10000)

model = laion_clap.CLAP_Module(enable_fusion=False, device="cuda")
model.load_ckpt()
start_time = time.time()
print("starting embeddings at time", start_time)

with torch.no_grad():
    embeddings = model.get_text_embedding(prompts)
end_time = time.time()
print(
    "done getting embeddings at time", end_time, "elapsed time", end_time - start_time
)

S = cosine_similarity(embeddings)
m = np.maximum(S, 0.0)

selector = SaturatedCoverageSelection(n_samples=1200)
selector.fit(m)

rank = selector.ranking
print(rank)

gains = selector.gains

cumulative = np.cumsum(gains)

plt.plot(cumulative)
plt.xlabel("Subset size")
plt.ylabel("Facility Location objective")
plt.show()
