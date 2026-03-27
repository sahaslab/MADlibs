import json
import matplotlib.pyplot as plt
import numpy as np

def graph(analyze_key: str, mir_results_filename: str = "results/outputs/madlibs/musicgen/mir/mir_results.jsonl") -> None:
    plt.figure()
    
    data = []
    with open(mir_results_filename, 'r') as f:
        for line in f:
            data.append(json.loads(line)[analyze_key])

    plt.hist(data, bins=20, edgecolor='black')

    plt.title(f"{analyze_key} counts")
    plt.xlabel(f"{analyze_key}")
    plt.ylabel("Count")

    plt.savefig(f"mtg-jamendo-{analyze_key}_plot.png")
    plt.close()

def graph_bar(analyze_key: str, mir_results_filename: str = "results/outputs/madlibs/musicgen/mir/mir_results.jsonl"):
    plt.figure()
    
    data = []
    with open(mir_results_filename, 'r') as f:
        for line in f:
            data.append(json.loads(line)[analyze_key])

    data = np.array(data)
    data = np.sum(data, axis=0)
    bars = np.arange(1, 13)
    
    plt.bar(bars, data)

    plt.title(f"{analyze_key} Agg")
    plt.xlabel("Note")
    plt.ylabel("Count")

    plt.savefig(f"mtg-jamendo-{analyze_key}_plot.png")
    plt.close()

if __name__ == "__main__":
    mir_results_filename = "results/outputs/madlibs/musicgen/mir/mir_results.jsonl"
    analyze_keys = [
        "longest_silence_ms",
        "tempo",
        "time_signature",
        "rmsstd",
    ]
    analyze_keys_bar = [
        "chromasum",
    ]

    for key in analyze_keys:
        graph(key)
    for key in analyze_keys_bar:
        graph_bar(key)