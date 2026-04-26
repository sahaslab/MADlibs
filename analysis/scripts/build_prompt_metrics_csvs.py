import os
import re
import math
import argparse
import collections
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Any
from population_analysis_common import (
    load_song_manifest,
    attach_prompt_text,
    select_unique_prompts,
    l2_normalize_rows,
    write_csv,
    sanitize_text
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mir_csv", default="analysis_data/mir_features.csv")
    parser.add_argument("--gpu_csv", default="analysis_data/gpu_features.csv")
    parser.add_argument("--output_dir", default="analysis_data")
    parser.add_argument("--include_strategies", nargs="*", default=None)
    parser.add_argument("--prompt_workers", type=int, default=16)
    parser.add_argument("--text_embedding_backend", choices=["none", "gemini"], default="gemini")
    parser.add_argument("--gemini_model", default="gemini/text-embedding-004")
    parser.add_argument("--gemini_batch_size", type=int, default=128)
    parser.add_argument("--cluster_ks", nargs="*", type=int, default=[8, 12, 16, 24])
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument("--embedding_cache_npy", default="analysis_data/prompt_gemini_embeddings.npy")
    parser.add_argument("--embedding_cache_csv", default="analysis_data/prompt_gemini_embedding_manifest.csv")
    args = parser.parse_args()

    manifest = load_song_manifest(
        mir_csv=args.mir_csv,
        gpu_csv=args.gpu_csv,
        include_strategies=args.include_strategies
    )
    manifest = attach_prompt_text(
        manifest,
        max_workers=max(1, args.prompt_workers)
    )
    manifest = add_prompt_metrics(manifest)

    unique_prompts = select_unique_prompts(manifest)
    unique_prompts = add_prompt_metrics(unique_prompts)

    row_output = manifest.loc[
        :,
        [
            "join_key",
            "strategy",
            "model",
            "prompt_id",
            "prompt_key",
            "prompt_path",
            "prompt_text",
            "prompt_char_count",
            "prompt_word_count",
            "prompt_sentence_count",
            "prompt_unique_word_count",
            "prompt_type_token_ratio",
            "prompt_avg_word_length",
            "prompt_token_entropy",
            "prompt_digit_share",
            "prompt_upper_share",
            "prompt_non_ascii_share",
            "prompt_punctuation_share",
            "prompt_comma_count",
            "prompt_colon_count",
            "prompt_semicolon_count",
            "prompt_quote_count",
            "has_generate_prefix",
            "has_featuring",
            "has_with",
            "has_vocals",
            "has_male",
            "has_female",
            "mean_similarity"
        ]
    ].copy()

    summary_output = summarize_prompt_metrics(unique_prompts)

    write_csv(
        row_output,
        os.path.join(args.output_dir, "prompt_metrics_rows.csv")
    )
    write_csv(
        unique_prompts,
        os.path.join(args.output_dir, "prompt_metrics_unique_prompts.csv")
    )
    write_csv(
        summary_output,
        os.path.join(args.output_dir, "prompt_metrics_summary.csv")
    )

    if args.text_embedding_backend == "none":
        return

    text_embedding_matrix = load_or_compute_prompt_embeddings(
        unique_prompts=unique_prompts,
        text_embedding_backend=args.text_embedding_backend,
        gemini_model=args.gemini_model,
        gemini_batch_size=args.gemini_batch_size,
        embedding_cache_npy=args.embedding_cache_npy,
        embedding_cache_csv=args.embedding_cache_csv
    )
    text_embedding_matrix = l2_normalize_rows(text_embedding_matrix)

    cluster_assignments = build_prompt_cluster_assignments(
        unique_prompts=unique_prompts,
        embedding_matrix=text_embedding_matrix,
        cluster_ks=args.cluster_ks,
        text_embedding_backend=args.text_embedding_backend
    )
    cluster_summary = summarize_prompt_cluster_assignments(cluster_assignments)
    neighbor_summary = summarize_prompt_neighbors(
        unique_prompts=unique_prompts,
        embedding_matrix=text_embedding_matrix,
        neighbor_k=args.neighbor_k,
        text_embedding_backend=args.text_embedding_backend
    )

    write_csv(
        cluster_assignments,
        os.path.join(args.output_dir, "prompt_embedding_cluster_assignments.csv")
    )
    write_csv(
        cluster_summary,
        os.path.join(args.output_dir, "prompt_embedding_cluster_summary.csv")
    )
    write_csv(
        neighbor_summary,
        os.path.join(args.output_dir, "prompt_embedding_neighbor_summary.csv")
    )


def add_prompt_metrics(table: pd.DataFrame) -> pd.DataFrame:
    metrics = table["prompt_text"].fillna("").map(compute_prompt_metrics)
    metrics_table = pd.DataFrame(metrics.tolist())
    return pd.concat([table.reset_index(drop=True), metrics_table.reset_index(drop=True)], axis=1)


def compute_prompt_metrics(prompt_text: str) -> dict[str, Any]:
    clean_text = sanitize_text(prompt_text)
    lower_text = clean_text.lower()
    words = re.findall(r"[A-Za-z0-9']+", clean_text)
    lower_words = [word.lower() for word in words]
    unique_words = sorted(set(lower_words))
    char_count = len(clean_text)
    word_count = len(words)
    sentence_count = max(1, len([part for part in re.split(r"[.!?]+", clean_text) if part.strip()])) if clean_text else 0
    digit_count = sum(char.isdigit() for char in clean_text)
    upper_count = sum(char.isupper() for char in clean_text)
    ascii_count = sum(ord(char) < 128 for char in clean_text)
    punctuation_count = sum(char in ".,;:!?-()\"'" for char in clean_text)

    token_counter = collections.Counter(lower_words)
    token_probs = np.array(list(token_counter.values()), dtype=float)
    if token_probs.size > 0:
        token_probs = token_probs / token_probs.sum()
        token_entropy = float(-(token_probs * np.log(token_probs)).sum())
    else:
        token_entropy = 0.0

    return {
        "prompt_text": clean_text,
        "prompt_char_count": char_count,
        "prompt_word_count": word_count,
        "prompt_sentence_count": sentence_count,
        "prompt_unique_word_count": len(unique_words),
        "prompt_type_token_ratio": float(len(unique_words) / word_count) if word_count > 0 else 0.0,
        "prompt_avg_word_length": float(np.mean([len(word) for word in words])) if word_count > 0 else 0.0,
        "prompt_token_entropy": token_entropy,
        "prompt_digit_share": float(digit_count / char_count) if char_count > 0 else 0.0,
        "prompt_upper_share": float(upper_count / char_count) if char_count > 0 else 0.0,
        "prompt_non_ascii_share": float((char_count - ascii_count) / char_count) if char_count > 0 else 0.0,
        "prompt_punctuation_share": float(punctuation_count / char_count) if char_count > 0 else 0.0,
        "prompt_comma_count": clean_text.count(","),
        "prompt_colon_count": clean_text.count(":"),
        "prompt_semicolon_count": clean_text.count(";"),
        "prompt_quote_count": clean_text.count("\"") + clean_text.count("'"),
        "has_generate_prefix": int(lower_text.startswith("generate ")),
        "has_featuring": int(" featuring " in " %s " % lower_text),
        "has_with": int(" with " in " %s " % lower_text),
        "has_vocals": int(" vocal" in lower_text),
        "has_male": int(" male " in " %s " % lower_text),
        "has_female": int(" female " in " %s " % lower_text)
    }


def summarize_prompt_metrics(unique_prompts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        unique_prompts.groupby("strategy", as_index=False)
        .agg(
            prompt_count=("prompt_key", "size"),
            mean_prompt_char_count=("prompt_char_count", "mean"),
            mean_prompt_word_count=("prompt_word_count", "mean"),
            mean_prompt_sentence_count=("prompt_sentence_count", "mean"),
            mean_prompt_type_token_ratio=("prompt_type_token_ratio", "mean"),
            mean_prompt_avg_word_length=("prompt_avg_word_length", "mean"),
            mean_prompt_token_entropy=("prompt_token_entropy", "mean"),
            share_generate_prefix=("has_generate_prefix", "mean"),
            share_featuring=("has_featuring", "mean"),
            share_with=("has_with", "mean"),
            share_vocals=("has_vocals", "mean"),
            share_male=("has_male", "mean"),
            share_female=("has_female", "mean")
        )
    )

    duplicate_rates = (
        unique_prompts.groupby(["strategy", "prompt_text"], as_index=False)
        .size()
        .groupby("strategy", as_index=False)
        .agg(
            duplicate_prompt_share=("size", lambda values: float(np.mean(values > 1)))
        )
    )

    return summary.merge(duplicate_rates, on="strategy", how="left")


def load_or_compute_prompt_embeddings(
    unique_prompts: pd.DataFrame,
    text_embedding_backend: str,
    gemini_model: str,
    gemini_batch_size: int,
    embedding_cache_npy: str,
    embedding_cache_csv: str
) -> np.ndarray:
    ordered_prompts = unique_prompts.copy()
    ordered_prompts["original_position"] = np.arange(len(ordered_prompts))
    ordered_prompts = ordered_prompts.sort_values("prompt_key", kind="stable").reset_index(drop=True).copy()
    ordered_prompts["local_cache_path"] = ordered_prompts["prompt_path"].map(build_prompt_cache_path)

    local_cache_hits = ordered_prompts["local_cache_path"].map(os.path.exists)
    embedding_rows = [None] * len(ordered_prompts)

    if local_cache_hits.all():
        for row_index, cache_path in enumerate(ordered_prompts["local_cache_path"].tolist()):
            embedding_rows[row_index] = np.load(cache_path)
        embedding_matrix = np.vstack(embedding_rows)
    else:
        if text_embedding_backend == "gemini":
            missing_prompts = ordered_prompts.loc[~local_cache_hits].copy().reset_index()
            missing_embeddings = compute_gemini_text_embeddings(
                texts=missing_prompts["prompt_text"].tolist(),
                gemini_model=gemini_model,
                batch_size=gemini_batch_size
            )
        else:
            raise ValueError("Unknown text embedding backend: %s" % text_embedding_backend)

        missing_index_to_position = {
            int(original_index): position
            for position, original_index in enumerate(missing_prompts["index"].tolist())
        }

        for row_index, cache_path in enumerate(ordered_prompts["local_cache_path"].tolist()):
            if local_cache_hits.iloc[row_index]:
                embedding_rows[row_index] = np.load(cache_path)
                continue

            embedding_vector = missing_embeddings[missing_index_to_position[row_index]]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embedding_vector)
            embedding_rows[row_index] = embedding_vector

        embedding_matrix = np.vstack(embedding_rows)

    os.makedirs(os.path.dirname(embedding_cache_npy), exist_ok=True)
    np.save(embedding_cache_npy, embedding_matrix)
    write_csv(
        ordered_prompts.loc[:, ["prompt_key", "local_cache_path"]].copy(),
        embedding_cache_csv
    )

    restore_order = ordered_prompts.sort_values("original_position", kind="stable")
    return embedding_matrix[restore_order.index.to_numpy()]


def compute_gemini_text_embeddings(
    texts: list[str],
    gemini_model: str,
    batch_size: int
) -> np.ndarray:
    import litellm

    embedding_batches = []

    for start_index in range(0, len(texts), batch_size):
        text_batch = texts[start_index : start_index + batch_size]
        response = litellm.embedding(
            model=gemini_model,
            input=text_batch
        )
        batch_embeddings = np.array(
            [item["embedding"] for item in response["data"]],
            dtype=np.float32
        )
        embedding_batches.append(batch_embeddings)

    return np.concatenate(embedding_batches, axis=0)


def build_prompt_cache_path(prompt_path: str) -> str:
    prompt_dir = os.path.dirname(prompt_path)
    return os.path.join(prompt_dir, ".cache", "text_embeddings.npy")


def build_prompt_cluster_assignments(
    unique_prompts: pd.DataFrame,
    embedding_matrix: np.ndarray,
    cluster_ks: list[int],
    text_embedding_backend: str
) -> pd.DataFrame:
    assignment_tables = []

    for cluster_k in cluster_ks:
        cluster_model = KMeans(
            n_clusters=min(cluster_k, embedding_matrix.shape[0]),
            random_state=20260418 + int(cluster_k),
            n_init=10
        )
        cluster_ids = cluster_model.fit_predict(embedding_matrix)

        assignment_table = unique_prompts.loc[
            :,
            ["strategy", "prompt_id", "prompt_key", "prompt_path"]
        ].copy()
        assignment_table["text_embedding_backend"] = text_embedding_backend
        assignment_table["cluster_k"] = int(cluster_k)
        assignment_table["cluster_id"] = cluster_ids.astype(int)
        assignment_tables.append(assignment_table)

    return pd.concat(assignment_tables, axis=0, ignore_index=True)


def summarize_prompt_cluster_assignments(cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for cluster_k, cluster_table in cluster_assignments.groupby("cluster_k", sort=True):
        for strategy, strategy_table in cluster_table.groupby("strategy", sort=True):
            cluster_probs = strategy_table["cluster_id"].value_counts(normalize=True, sort=True).to_numpy(dtype=float)
            cluster_entropy = compute_entropy(cluster_probs)

            summary_rows.append(
                {
                    "strategy": strategy,
                    "cluster_k": int(cluster_k),
                    "prompt_count": int(len(strategy_table)),
                    "occupied_cluster_count": int(len(cluster_probs)),
                    "cluster_entropy": float(cluster_entropy),
                    "effective_cluster_count": float(math.exp(cluster_entropy)),
                    "top_1_mass": float(cluster_probs[0]),
                    "top_3_mass": float(cluster_probs[: min(3, len(cluster_probs))].sum())
                }
            )

    return pd.DataFrame(summary_rows).sort_values(["strategy", "cluster_k"], kind="stable").reset_index(drop=True)


def summarize_prompt_neighbors(
    unique_prompts: pd.DataFrame,
    embedding_matrix: np.ndarray,
    neighbor_k: int,
    text_embedding_backend: str
) -> pd.DataFrame:
    neighbor_model = NearestNeighbors(
        n_neighbors=min(neighbor_k + 1, embedding_matrix.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    neighbor_model.fit(embedding_matrix)
    neighbor_indices = neighbor_model.kneighbors(
        embedding_matrix,
        return_distance=False
    )[:, 1:]

    row_metrics = []

    for row_index, neighbor_idx in enumerate(neighbor_indices):
        row = unique_prompts.iloc[row_index]
        neighbor_table = unique_prompts.iloc[neighbor_idx]
        strategy_probs = neighbor_table["strategy"].value_counts(normalize=True, sort=False)
        strategy_entropy = compute_entropy(strategy_probs.to_numpy(dtype=float))

        row_metrics.append(
            {
                "strategy": row["strategy"],
                "same_strategy_share": float((neighbor_table["strategy"] == row["strategy"]).mean()),
                "neighbor_strategy_entropy": float(strategy_entropy),
                "effective_neighbor_strategy_count": float(np.exp(strategy_entropy))
            }
        )

    summary = (
        pd.DataFrame(row_metrics)
        .groupby("strategy", as_index=False)
        .mean(numeric_only=True)
    )
    counts = unique_prompts.groupby("strategy", as_index=False).size().rename(columns={"size": "prompt_count"})
    summary = counts.merge(summary, on="strategy", how="left")
    summary["text_embedding_backend"] = text_embedding_backend

    return summary.loc[
        :,
        [
            "strategy",
            "prompt_count",
            "text_embedding_backend",
            "same_strategy_share",
            "neighbor_strategy_entropy",
            "effective_neighbor_strategy_count"
        ]
    ].sort_values(["strategy"], kind="stable").reset_index(drop=True)


def compute_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


if __name__ == "__main__":
    main()
