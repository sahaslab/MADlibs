import os
import argparse
import gc
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from population_analysis_common import l2_normalize_rows, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mir_csv", default="analysis_data/mir_features.csv")
    parser.add_argument("--prompt_metrics_csv", default="analysis_data/prompt_metrics_unique_prompts.csv")
    parser.add_argument("--output_dir", default="analysis_data")
    parser.add_argument("--analysis_prefix", default="")
    parser.add_argument("--include_strategies", nargs="*", default=None)
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument(
        "--cluster_ks",
        nargs="*",
        type=int,
        default=[
            8,
            12,
            16,
            24
        ]
    )
    parser.add_argument("--max_rows_per_cell", type=int, default=2500)
    parser.add_argument("--skip_audio", action="store_true")
    parser.add_argument("--skip_prompt", action="store_true")
    parser.add_argument("--audio_embedding_manifest_csv", default="analysis_data/audio_clap_embedding_manifest_laion.csv")
    parser.add_argument("--audio_embedding_npy", default="analysis_data/audio_clap_embeddings_laion.npy")
    parser.add_argument("--audio_embedding_backend", default="clap_audio_laion")
    parser.add_argument("--prompt_embedding_manifest_csv", default="analysis_data/prompt_gemini_embedding_manifest.csv")
    parser.add_argument("--prompt_embedding_npy", default="analysis_data/prompt_gemini_embeddings.npy")
    parser.add_argument("--prompt_embedding_backend", default="gemini_text_embedding_004")
    args = parser.parse_args()

    output_prefix = ""
    if args.analysis_prefix != "":
        output_prefix = "%s_" % args.analysis_prefix

    if not args.skip_audio:
        audio_table, audio_embedding_matrix = load_audio_embedding_inputs(
            audio_embedding_manifest_csv=args.audio_embedding_manifest_csv,
            audio_embedding_npy=args.audio_embedding_npy,
            mir_csv=args.mir_csv,
            include_strategies=args.include_strategies,
            max_rows_per_cell=args.max_rows_per_cell
        )
        write_audio_cluster_outputs(
            audio_table=audio_table,
            embedding_matrix=audio_embedding_matrix,
            output_dir=args.output_dir,
            output_prefix=output_prefix,
            cluster_ks=args.cluster_ks,
            neighbor_k=args.neighbor_k,
            embedding_backend=args.audio_embedding_backend
        )
        del audio_table
        del audio_embedding_matrix
        gc.collect()

    if not args.skip_prompt:
        prompt_table, prompt_embedding_matrix = load_prompt_embedding_inputs(
            prompt_metrics_csv=args.prompt_metrics_csv,
            prompt_embedding_manifest_csv=args.prompt_embedding_manifest_csv,
            prompt_embedding_npy=args.prompt_embedding_npy,
            include_strategies=args.include_strategies
        )
        write_prompt_cluster_outputs(
            prompt_table=prompt_table,
            embedding_matrix=prompt_embedding_matrix,
            output_dir=args.output_dir,
            output_prefix=output_prefix,
            cluster_ks=args.cluster_ks,
            neighbor_k=args.neighbor_k,
            embedding_backend=args.prompt_embedding_backend
        )


def load_audio_embedding_inputs(
    audio_embedding_manifest_csv: str,
    audio_embedding_npy: str,
    mir_csv: str,
    include_strategies: list[str] | None,
    max_rows_per_cell: int
) -> tuple[pd.DataFrame, np.ndarray]:
    audio_manifest = pd.read_csv(audio_embedding_manifest_csv)
    embedding_matrix = np.load(audio_embedding_npy)
    mir_table = pd.read_csv(mir_csv, low_memory=False)

    if len(audio_manifest) != embedding_matrix.shape[0]:
        raise ValueError(
            "Audio embedding rows (%s) do not match manifest rows (%s)" % (
                embedding_matrix.shape[0],
                len(audio_manifest)
            )
        )

    audio_metadata = mir_table.loc[
        :,
        [
            "join_key",
            "strategy",
            "model",
            "prompt_id",
            "line_number",
            "audio_filename"
        ]
    ].drop_duplicates(subset=["join_key"], keep="first").copy()

    ordered_table = audio_manifest.loc[:, ["join_key"]].merge(
        audio_metadata,
        on="join_key",
        how="left",
        validate="one_to_one"
    )

    if ordered_table["strategy"].isna().any():
        missing_count = int(ordered_table["strategy"].isna().sum())
        raise ValueError("Audio embedding metadata missing for %s rows" % missing_count)

    keep_mask = np.isfinite(embedding_matrix).all(axis=1)
    if include_strategies is not None:
        keep_mask = keep_mask & ordered_table["strategy"].isin(include_strategies).to_numpy()

    ordered_table = ordered_table.loc[keep_mask].reset_index(drop=True).copy()
    embedding_matrix = embedding_matrix[keep_mask]
    ordered_table["strategy"] = ordered_table["strategy"].astype(str)
    ordered_table["model"] = ordered_table["model"].astype(str)
    ordered_table["prompt_id"] = ordered_table["prompt_id"].astype(str)

    capped_indices = select_capped_indices(
        table=ordered_table,
        group_columns=[
            "strategy",
            "model"
        ],
        max_rows_per_group=max_rows_per_cell
    )

    ordered_table = ordered_table.iloc[capped_indices].reset_index(drop=True).copy()
    embedding_matrix = embedding_matrix[capped_indices]

    return ordered_table, l2_normalize_rows(embedding_matrix)


def load_prompt_embedding_inputs(
    prompt_metrics_csv: str,
    prompt_embedding_manifest_csv: str,
    prompt_embedding_npy: str,
    include_strategies: list[str] | None
) -> tuple[pd.DataFrame, np.ndarray]:
    prompt_table = pd.read_csv(prompt_metrics_csv)
    prompt_manifest = pd.read_csv(prompt_embedding_manifest_csv)
    embedding_matrix = np.load(prompt_embedding_npy)

    if len(prompt_manifest) != embedding_matrix.shape[0]:
        raise ValueError(
            "Prompt embedding rows (%s) do not match manifest rows (%s)" % (
                embedding_matrix.shape[0],
                len(prompt_manifest)
            )
        )

    ordered_table = prompt_manifest.loc[:, ["prompt_key"]].merge(
        prompt_table,
        on="prompt_key",
        how="left",
        validate="one_to_one"
    )

    if ordered_table["strategy"].isna().any():
        missing_count = int(ordered_table["strategy"].isna().sum())
        raise ValueError("Prompt embedding metadata missing for %s rows" % missing_count)

    keep_mask = np.isfinite(embedding_matrix).all(axis=1)
    if include_strategies is not None:
        keep_mask = keep_mask & ordered_table["strategy"].isin(include_strategies).to_numpy()

    ordered_table = ordered_table.loc[keep_mask].reset_index(drop=True).copy()
    embedding_matrix = embedding_matrix[keep_mask]
    ordered_table["strategy"] = ordered_table["strategy"].astype(str)
    ordered_table["prompt_id"] = ordered_table["prompt_id"].astype(str)

    return ordered_table, l2_normalize_rows(embedding_matrix)


def select_capped_indices(
    table: pd.DataFrame,
    group_columns: list[str],
    max_rows_per_group: int
) -> np.ndarray:
    if max_rows_per_group <= 0:
        return np.arange(len(table))

    working = table.copy()
    working["original_position"] = np.arange(len(working))
    capped = (
        working.sort_values(group_columns + ["join_key"], kind="stable")
        .groupby(group_columns, sort=False, group_keys=False)
        .head(max_rows_per_group)
        .sort_values("original_position", kind="stable")
    )

    return capped["original_position"].to_numpy(dtype=int)


def write_audio_cluster_outputs(
    audio_table: pd.DataFrame,
    embedding_matrix: np.ndarray,
    output_dir: str,
    output_prefix: str,
    cluster_ks: list[int],
    neighbor_k: int,
    embedding_backend: str
) -> None:
    manifest_output = audio_table.loc[
        :,
        [
            "join_key",
            "strategy",
            "model",
            "prompt_id",
            "line_number",
            "audio_filename"
        ]
    ].copy()
    manifest_output["embedding_backend"] = embedding_backend

    cluster_assignments = build_cluster_assignments(
        metadata_table=audio_table,
        embedding_matrix=embedding_matrix,
        id_columns=[
            "join_key",
            "strategy",
            "model",
            "prompt_id",
            "line_number",
            "audio_filename"
        ],
        cluster_ks=cluster_ks,
        embedding_backend=embedding_backend
    )
    cluster_summary = summarize_cluster_assignments(
        assignment_table=cluster_assignments,
        group_columns=[
            "strategy",
            "model"
        ],
        reference_column="model",
        reference_value="real"
    )
    cluster_summary["audio_count"] = cluster_summary["rows"]
    cluster_summary_median = summarize_cluster_medians(
        cluster_summary=cluster_summary,
        group_columns=[
            "strategy",
            "model"
        ]
    )
    cluster_summary_median["audio_count"] = cluster_summary_median["rows"]

    neighbor_indices, prototype_count = build_prototype_neighbor_indices(
        embedding_matrix=embedding_matrix,
        neighbor_k=neighbor_k,
        random_state=20260418
    )
    neighbor_summary = summarize_audio_neighbors(
        audio_table=audio_table,
        neighbor_indices=neighbor_indices,
        embedding_backend=embedding_backend,
        prototype_count=prototype_count
    )
    strategy_mixing_summary = summarize_neighbor_mixing(
        metadata_table=audio_table,
        neighbor_indices=neighbor_indices,
        group_columns=[
            "strategy",
            "model"
        ],
        category_column="strategy",
        neighbor_column="neighbor_strategy"
    )
    model_mixing_summary = summarize_neighbor_mixing(
        metadata_table=audio_table,
        neighbor_indices=neighbor_indices,
        group_columns=[
            "strategy",
            "model"
        ],
        category_column="model",
        neighbor_column="neighbor_model"
    )

    write_csv(
        manifest_output,
        os.path.join(output_dir, "%saudio_embedding_manifest.csv" % output_prefix)
    )
    write_csv(
        cluster_assignments,
        os.path.join(output_dir, "%saudio_embedding_cluster_assignments.csv" % output_prefix)
    )
    write_csv(
        cluster_summary,
        os.path.join(output_dir, "%saudio_embedding_cluster_summary.csv" % output_prefix)
    )
    write_csv(
        cluster_summary_median,
        os.path.join(output_dir, "%saudio_embedding_cluster_summary_median.csv" % output_prefix)
    )
    write_csv(
        neighbor_summary,
        os.path.join(output_dir, "%saudio_embedding_neighbor_summary.csv" % output_prefix)
    )
    write_csv(
        strategy_mixing_summary,
        os.path.join(output_dir, "%saudio_embedding_neighbor_strategy_mixing_summary.csv" % output_prefix)
    )
    write_csv(
        model_mixing_summary,
        os.path.join(output_dir, "%saudio_embedding_neighbor_model_mixing_summary.csv" % output_prefix)
    )


def write_prompt_cluster_outputs(
    prompt_table: pd.DataFrame,
    embedding_matrix: np.ndarray,
    output_dir: str,
    output_prefix: str,
    cluster_ks: list[int],
    neighbor_k: int,
    embedding_backend: str
) -> None:
    cluster_assignments = build_cluster_assignments(
        metadata_table=prompt_table,
        embedding_matrix=embedding_matrix,
        id_columns=[
            "strategy",
            "prompt_id",
            "prompt_key",
            "prompt_path"
        ],
        cluster_ks=cluster_ks,
        embedding_backend=embedding_backend
    )
    cluster_summary = summarize_cluster_assignments(
        assignment_table=cluster_assignments,
        group_columns=[
            "strategy"
        ],
        reference_column=None,
        reference_value=None
    )
    cluster_summary["prompt_count"] = cluster_summary["rows"]
    cluster_summary_median = summarize_cluster_medians(
        cluster_summary=cluster_summary,
        group_columns=[
            "strategy"
        ]
    )
    cluster_summary_median["prompt_count"] = cluster_summary_median["rows"]

    neighbor_indices, prototype_count = build_prototype_neighbor_indices(
        embedding_matrix=embedding_matrix,
        neighbor_k=neighbor_k,
        random_state=20260418
    )
    neighbor_summary = summarize_prompt_neighbors(
        prompt_table=prompt_table,
        neighbor_indices=neighbor_indices,
        embedding_backend=embedding_backend,
        prototype_count=prototype_count
    )
    mixing_summary = summarize_neighbor_mixing(
        metadata_table=prompt_table,
        neighbor_indices=neighbor_indices,
        group_columns=[
            "strategy"
        ],
        category_column="strategy",
        neighbor_column="neighbor_strategy"
    )

    write_csv(
        cluster_assignments,
        os.path.join(output_dir, "%sprompt_embedding_cluster_assignments.csv" % output_prefix)
    )
    write_csv(
        cluster_summary,
        os.path.join(output_dir, "%sprompt_embedding_cluster_summary.csv" % output_prefix)
    )
    write_csv(
        cluster_summary_median,
        os.path.join(output_dir, "%sprompt_embedding_cluster_summary_median.csv" % output_prefix)
    )
    write_csv(
        neighbor_summary,
        os.path.join(output_dir, "%sprompt_embedding_neighbor_summary.csv" % output_prefix)
    )
    write_csv(
        mixing_summary,
        os.path.join(output_dir, "%sprompt_neighbor_mixing_summary.csv" % output_prefix)
    )


def build_cluster_assignments(
    metadata_table: pd.DataFrame,
    embedding_matrix: np.ndarray,
    id_columns: list[str],
    cluster_ks: list[int],
    embedding_backend: str
) -> pd.DataFrame:
    assignment_tables = []

    for cluster_k in cluster_ks:
        cluster_model = KMeans(
            n_clusters=min(int(cluster_k), embedding_matrix.shape[0]),
            random_state=20260418 + int(cluster_k),
            n_init=10
        )
        cluster_ids = cluster_model.fit_predict(embedding_matrix)

        assignment_table = metadata_table.loc[:, id_columns].copy()
        assignment_table["embedding_backend"] = embedding_backend
        assignment_table["cluster_k"] = int(cluster_k)
        assignment_table["cluster_id"] = cluster_ids.astype(int)
        assignment_tables.append(assignment_table)

    return pd.concat(assignment_tables, axis=0, ignore_index=True)


def summarize_cluster_assignments(
    assignment_table: pd.DataFrame,
    group_columns: list[str],
    reference_column: str | None,
    reference_value: str | None
) -> pd.DataFrame:
    summary_rows = []

    for cluster_k, cluster_table in assignment_table.groupby("cluster_k", sort=True):
        if reference_column is None:
            reference_clusters = None
        else:
            reference_clusters = set(
                cluster_table.loc[cluster_table[reference_column] == reference_value, "cluster_id"].tolist()
            )

        for group_key, group_table in cluster_table.groupby(group_columns, sort=True):
            cluster_probs = group_table["cluster_id"].value_counts(normalize=True, sort=True).to_numpy(dtype=float)
            cluster_entropy = compute_entropy(cluster_probs)

            if reference_clusters is None or len(reference_clusters) == 0:
                human_cluster_coverage = np.nan
            else:
                occupied_clusters = set(group_table["cluster_id"].tolist())
                human_cluster_coverage = float(len(reference_clusters & occupied_clusters) / len(reference_clusters))

            row = {
                "cluster_k": int(cluster_k),
                "rows": int(len(group_table)),
                "occupied_cluster_count": int(len(cluster_probs)),
                "cluster_entropy": float(cluster_entropy),
                "effective_cluster_count": float(math.exp(cluster_entropy)),
                "human_cluster_coverage": human_cluster_coverage,
                "top_1_mass": float(cluster_probs[0]),
                "top_3_mass": float(cluster_probs[: min(3, len(cluster_probs))].sum())
            }

            if isinstance(group_key, tuple):
                for column_name, column_value in zip(group_columns, group_key):
                    row[column_name] = column_value
            else:
                row[group_columns[0]] = group_key

            summary_rows.append(row)

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

    return pd.DataFrame(summary_rows).loc[:, ordered_columns].sort_values(
        group_columns + ["cluster_k"],
        kind="stable"
    ).reset_index(drop=True)


def summarize_cluster_medians(
    cluster_summary: pd.DataFrame,
    group_columns: list[str]
) -> pd.DataFrame:
    median_columns = [
        "rows",
        "occupied_cluster_count",
        "cluster_entropy",
        "effective_cluster_count",
        "human_cluster_coverage",
        "top_1_mass",
        "top_3_mass"
    ]

    median_summary = (
        cluster_summary.groupby(group_columns, as_index=False)
        .agg(
            cluster_k_count=("cluster_k", "nunique"),
            **{
                column_name: (column_name, "median")
                for column_name in median_columns
            }
        )
    )

    return median_summary.sort_values(group_columns, kind="stable").reset_index(drop=True)


def build_prototype_neighbor_indices(
    embedding_matrix: np.ndarray,
    neighbor_k: int,
    random_state: int
) -> tuple[list[np.ndarray], int]:
    row_count = embedding_matrix.shape[0]
    neighbor_indices = [
        np.array([], dtype=int)
        for _ in range(row_count)
    ]

    if row_count < 2:
        return neighbor_indices, row_count

    prototype_count = min(max(32, int(math.floor(math.sqrt(row_count)))), 256, row_count)
    prototype_model = KMeans(
        n_clusters=prototype_count,
        random_state=random_state,
        n_init=1,
        max_iter=50
    )
    prototype_assignments = prototype_model.fit_predict(embedding_matrix)

    for prototype_id in sorted(np.unique(prototype_assignments).tolist()):
        current_indices = np.where(prototype_assignments == prototype_id)[0]
        if current_indices.size < 2:
            continue

        current_neighbor_k = min(neighbor_k + 1, current_indices.size)
        neighbor_model = NearestNeighbors(
            n_neighbors=current_neighbor_k,
            metric="euclidean",
            algorithm="brute",
            n_jobs=1
        )
        current_matrix = embedding_matrix[current_indices]
        neighbor_model.fit(current_matrix)
        current_neighbor_positions = neighbor_model.kneighbors(
            current_matrix,
            return_distance=False
        )

        for local_row_index in range(current_indices.size):
            row_index = current_indices[local_row_index]
            row_neighbors = current_indices[current_neighbor_positions[local_row_index]]
            row_neighbors = row_neighbors[row_neighbors != row_index]
            neighbor_indices[row_index] = row_neighbors[:neighbor_k].astype(int)

    return neighbor_indices, prototype_count


def summarize_audio_neighbors(
    audio_table: pd.DataFrame,
    neighbor_indices: list[np.ndarray],
    embedding_backend: str,
    prototype_count: int
) -> pd.DataFrame:
    row_metrics = []

    for row_index, current_neighbor_indices in enumerate(neighbor_indices):
        if len(current_neighbor_indices) == 0:
            continue

        row = audio_table.iloc[row_index]
        neighbor_table = audio_table.iloc[current_neighbor_indices]
        source_probs = neighbor_table["model"].value_counts(normalize=True, sort=False).to_numpy(dtype=float)
        strategy_probs = neighbor_table["strategy"].value_counts(normalize=True, sort=False).to_numpy(dtype=float)
        source_entropy = compute_entropy(source_probs)
        strategy_entropy = compute_entropy(strategy_probs)
        same_source_share = float((neighbor_table["model"] == row["model"]).mean())
        same_strategy_share = float((neighbor_table["strategy"] == row["strategy"]).mean())
        reference_share = float((neighbor_table["model"] == "real").mean())

        row_metrics.append(
            {
                "strategy": row["strategy"],
                "model": row["model"],
                "same_source_share": same_source_share,
                "same_strategy_share": same_strategy_share,
                "reference_share": reference_share,
                "neighbor_source_entropy": float(source_entropy),
                "effective_neighbor_source_count": float(math.exp(source_entropy)),
                "same_strategy_audio_neighbor_share": same_strategy_share,
                "same_model_audio_neighbor_share": same_source_share,
                "same_cell_audio_neighbor_share": float(
                    ((neighbor_table["strategy"] == row["strategy"]) & (neighbor_table["model"] == row["model"])).mean()
                ),
                "real_audio_neighbor_share": reference_share,
                "audio_neighbor_strategy_entropy": float(strategy_entropy),
                "effective_audio_neighbor_strategy_count": float(math.exp(strategy_entropy)),
                "audio_neighbor_model_entropy": float(source_entropy),
                "effective_audio_neighbor_model_count": float(math.exp(source_entropy))
            }
        )

    summary = (
        pd.DataFrame(row_metrics)
        .groupby([
            "strategy",
            "model"
        ], as_index=False)
        .mean(numeric_only=True)
    )
    counts = pd.DataFrame(row_metrics).groupby([
        "strategy",
        "model"
    ], as_index=False).size().rename(columns={"size": "rows"})
    summary = counts.merge(
        summary,
        on=[
            "strategy",
            "model"
        ],
        how="left"
    )
    summary["embedding_backend"] = embedding_backend
    summary["prototype_cluster_count"] = int(prototype_count)

    return summary.loc[
        :,
        [
            "strategy",
            "model",
            "rows",
            "embedding_backend",
            "prototype_cluster_count",
            "same_source_share",
            "same_strategy_share",
            "reference_share",
            "neighbor_source_entropy",
            "effective_neighbor_source_count",
            "same_strategy_audio_neighbor_share",
            "same_model_audio_neighbor_share",
            "same_cell_audio_neighbor_share",
            "real_audio_neighbor_share",
            "audio_neighbor_strategy_entropy",
            "effective_audio_neighbor_strategy_count",
            "audio_neighbor_model_entropy",
            "effective_audio_neighbor_model_count"
        ]
    ].sort_values([
        "strategy",
        "model"
    ], kind="stable").reset_index(drop=True)


def summarize_prompt_neighbors(
    prompt_table: pd.DataFrame,
    neighbor_indices: list[np.ndarray],
    embedding_backend: str,
    prototype_count: int
) -> pd.DataFrame:
    row_metrics = []

    for row_index, current_neighbor_indices in enumerate(neighbor_indices):
        if len(current_neighbor_indices) == 0:
            continue

        row = prompt_table.iloc[row_index]
        neighbor_table = prompt_table.iloc[current_neighbor_indices]
        strategy_probs = neighbor_table["strategy"].value_counts(normalize=True, sort=False).to_numpy(dtype=float)
        strategy_entropy = compute_entropy(strategy_probs)

        row_metrics.append(
            {
                "strategy": row["strategy"],
                "same_strategy_share": float((neighbor_table["strategy"] == row["strategy"]).mean()),
                "neighbor_strategy_entropy": float(strategy_entropy),
                "effective_neighbor_strategy_count": float(math.exp(strategy_entropy))
            }
        )

    summary = (
        pd.DataFrame(row_metrics)
        .groupby("strategy", as_index=False)
        .mean(numeric_only=True)
    )
    counts = pd.DataFrame(row_metrics).groupby("strategy", as_index=False).size().rename(columns={"size": "prompt_count"})
    summary = counts.merge(summary, on="strategy", how="left")
    summary["text_embedding_backend"] = embedding_backend
    summary["prototype_cluster_count"] = int(prototype_count)

    return summary.loc[
        :,
        [
            "strategy",
            "prompt_count",
            "text_embedding_backend",
            "prototype_cluster_count",
            "same_strategy_share",
            "neighbor_strategy_entropy",
            "effective_neighbor_strategy_count"
        ]
    ].sort_values("strategy", kind="stable").reset_index(drop=True)


def summarize_neighbor_mixing(
    metadata_table: pd.DataFrame,
    neighbor_indices: list[np.ndarray],
    group_columns: list[str],
    category_column: str,
    neighbor_column: str
) -> pd.DataFrame:
    categories = sorted(metadata_table[category_column].astype(str).unique().tolist())
    category_to_index = {
        category: index
        for index, category in enumerate(categories)
    }
    mixing_rows = []

    for row_index, current_neighbor_indices in enumerate(neighbor_indices):
        if len(current_neighbor_indices) == 0:
            continue

        row = metadata_table.iloc[row_index]
        neighbor_values = metadata_table.iloc[current_neighbor_indices][category_column].astype(str).to_numpy()
        category_counts = np.zeros(len(categories), dtype=float)

        for neighbor_value in neighbor_values:
            category_counts[category_to_index[str(neighbor_value)]] += 1.0

        category_shares = category_counts / float(len(current_neighbor_indices))
        base_row = {
            column_name: row[column_name]
            for column_name in group_columns
        }
        for category_index, neighbor_share in enumerate(category_shares.tolist()):
            output_row = base_row.copy()
            output_row[neighbor_column] = categories[category_index]
            output_row["neighbor_share"] = float(neighbor_share)
            mixing_rows.append(output_row)

    summary = (
        pd.DataFrame(mixing_rows)
        .groupby(group_columns + [neighbor_column], as_index=False)
        .agg(mean_neighbor_share=("neighbor_share", "mean"))
        .sort_values(group_columns + [neighbor_column], kind="stable")
        .reset_index(drop=True)
    )

    return summary


def compute_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


if __name__ == "__main__":
    main()
