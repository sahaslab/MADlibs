import os
import re
import math
import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path


def load_song_manifest(
    mir_csv: str,
    gpu_csv: str,
    include_strategies: list[str] | None = None
) -> pd.DataFrame:
    mir_table = pd.read_csv(mir_csv)
    gpu_table = pd.read_csv(gpu_csv)

    if include_strategies is not None:
        mir_table = mir_table.loc[mir_table["strategy"].isin(include_strategies)].copy()
        gpu_table = gpu_table.loc[gpu_table["strategy"].isin(include_strategies)].copy()

    gpu_subset = gpu_table.loc[
        :,
        [
            "join_key",
            "tempo",
            "time_signature",
            "n_instruments",
            "canonical_song_key",
            "mean_similarity"
        ]
    ].copy()

    manifest = mir_table.merge(
        gpu_subset,
        on="join_key",
        how="left"
    )

    manifest["audio_filename"] = manifest["audio_filename"].astype(str)
    manifest["prompt_path"] = manifest["audio_filename"].map(infer_prompt_path)
    manifest["prompt_key"] = manifest.apply(build_prompt_key, axis=1)
    manifest["source_key"] = manifest["model"].astype(str)
    manifest["cell_key"] = manifest["strategy"].astype(str) + "::" + manifest["model"].astype(str)
    manifest = manifest.sort_values(
        ["strategy", "model", "prompt_id", "line_number", "join_key"],
        kind="stable"
    ).reset_index(drop=True)

    return manifest


def infer_prompt_path(audio_filename: str) -> str:
    return str(Path(audio_filename).resolve().parent / "prompt.txt")


def build_prompt_key(row: pd.Series) -> str:
    strategy = str(row["strategy"])
    prompt_id = str(row["prompt_id"])
    return "%s__%s" % (strategy, prompt_id)


def attach_prompt_text(
    manifest: pd.DataFrame,
    max_workers: int
) -> pd.DataFrame:
    prompt_paths = sorted(manifest["prompt_path"].dropna().unique().tolist())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        prompt_pairs = list(executor.map(read_prompt_file_pair, prompt_paths))

    prompt_table = pd.DataFrame(
        prompt_pairs,
        columns=["prompt_path", "prompt_text"]
    )

    merged = manifest.merge(
        prompt_table,
        on="prompt_path",
        how="left"
    )
    merged["prompt_text"] = merged["prompt_text"].fillna("")

    return merged


def read_prompt_file_pair(prompt_path: str) -> tuple[str, str]:
    with open(prompt_path, "r", encoding="utf-8") as file_obj:
        prompt_text = file_obj.read().strip()
    return prompt_path, prompt_text


def select_unique_prompts(manifest: pd.DataFrame) -> pd.DataFrame:
    unique_prompts = manifest.loc[
        :,
        ["strategy", "prompt_id", "prompt_key", "prompt_path", "prompt_text"]
    ].drop_duplicates(
        subset=["prompt_key"],
        keep="first"
    ).reset_index(drop=True)

    return unique_prompts


def cap_rows_per_group(
    table: pd.DataFrame,
    group_columns: list[str],
    max_rows_per_group: int
) -> pd.DataFrame:
    if max_rows_per_group <= 0:
        return table.copy()

    capped = (
        table.sort_values(group_columns + ["join_key"], kind="stable")
        .groupby(group_columns, sort=False, group_keys=False)
        .head(max_rows_per_group)
        .reset_index(drop=True)
    )

    return capped


def l2_normalize_rows(embedding_matrix: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return embedding_matrix / row_norms


def summarize_cluster_assignments(
    assignment_table: pd.DataFrame,
    reference_mask: pd.Series | None,
    group_columns: list[str]
) -> pd.DataFrame:
    summary_rows = []

    for cluster_k, cluster_table in assignment_table.groupby("cluster_k", sort=True):
        if reference_mask is None:
            reference_clusters = None
        else:
            reference_clusters = set(
                cluster_table.loc[reference_mask.loc[cluster_table.index], "cluster_id"].tolist()
            )

        for group_key, group_table in cluster_table.groupby(group_columns, sort=True):
            cluster_counts = group_table["cluster_id"].value_counts(normalize=True, sort=True)
            cluster_probs = cluster_counts.to_numpy(dtype=float)
            cluster_entropy = compute_entropy(cluster_probs)

            if reference_clusters is None or len(reference_clusters) == 0:
                human_cluster_coverage = np.nan
            else:
                occupied_clusters = set(group_table["cluster_id"].tolist())
                human_cluster_coverage = float(
                    len(reference_clusters & occupied_clusters) / len(reference_clusters)
                )

            top_1_mass = float(cluster_probs[0])
            top_3_mass = float(cluster_probs[: min(3, len(cluster_probs))].sum())
            occupied_cluster_count = int(len(cluster_counts))
            effective_cluster_count = float(math.exp(cluster_entropy))

            row = {
                "cluster_k": int(cluster_k),
                "rows": int(len(group_table)),
                "occupied_cluster_count": occupied_cluster_count,
                "cluster_entropy": float(cluster_entropy),
                "effective_cluster_count": effective_cluster_count,
                "human_cluster_coverage": human_cluster_coverage,
                "top_1_mass": top_1_mass,
                "top_3_mass": top_3_mass
            }

            if isinstance(group_key, tuple):
                for column_name, column_value in zip(group_columns, group_key):
                    row[column_name] = column_value
            else:
                row[group_columns[0]] = group_key

            summary_rows.append(row)

    summary_table = pd.DataFrame(summary_rows)
    ordered_columns = group_columns + [
        "cluster_k",
        "rows",
        "occupied_cluster_count",
        "cluster_entropy",
        "effective_cluster_count",
        "human_cluster_coverage",
        "top_1_mass",
        "top_3_mass"
    ]

    return summary_table.loc[:, ordered_columns].sort_values(
        group_columns + ["cluster_k"],
        kind="stable"
    ).reset_index(drop=True)


def compute_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


def write_csv(table: pd.DataFrame, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    table.to_csv(output_path, index=False)


def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())
