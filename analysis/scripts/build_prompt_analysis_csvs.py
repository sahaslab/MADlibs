import os
import argparse
import logging
import numpy as np
import pandas as pd
from build_embedding_cluster_csvs import (
    build_cluster_assignments,
    build_prototype_neighbor_indices,
    summarize_audio_neighbors,
    summarize_cluster_assignments,
    summarize_cluster_medians,
    summarize_neighbor_mixing,
    summarize_prompt_neighbors
)


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="analysis_data")
    parser.add_argument("--prompt_metrics_csv", default="analysis_data/prompt_metrics_unique_prompts.csv")
    parser.add_argument("--embedding_manifest_csv", default="analysis_data/prompt_gemini_embedding_manifest.csv")
    parser.add_argument("--embedding_npy", default="analysis_data/prompt_gemini_embeddings.npy")
    parser.add_argument("--audio_embedding_manifest_csv", default="analysis_data/audio_clap_embedding_manifest_laion.csv")
    parser.add_argument("--audio_embedding_npy", default="analysis_data/audio_clap_embeddings_laion.npy")
    parser.add_argument("--mir_csv", default="analysis_data/mir_features.csv")
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
    parser.add_argument("--output_dir", default="analysis_data")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )

    logger.info("Loading prompt embeddings")
    prompt_table, embedding_matrix = load_prompt_embedding_inputs(
        prompt_metrics_csv=args.prompt_metrics_csv,
        embedding_manifest_csv=args.embedding_manifest_csv,
        embedding_npy=args.embedding_npy
    )
    prompt_neighbor_indices, prompt_prototype_count = build_prototype_neighbor_indices(
        embedding_matrix=embedding_matrix,
        neighbor_k=args.neighbor_k,
        random_state=20260418
    )
    neighbor_summary = summarize_prompt_neighbors(
        prompt_table=prompt_table,
        neighbor_indices=prompt_neighbor_indices,
        embedding_backend="gemini_text_embedding_004",
        prototype_count=prompt_prototype_count
    )

    logger.info("Building prompt embedding summaries")
    prompt_diversity_summary, pairwise_similarity_summary = build_strategy_embedding_summaries(
        prompt_table=prompt_table,
        embedding_matrix=embedding_matrix,
        neighbor_summary=neighbor_summary
    )
    neighbor_mixing_summary = summarize_neighbor_mixing(
        metadata_table=prompt_table,
        neighbor_indices=prompt_neighbor_indices,
        group_columns=["strategy"],
        category_column="strategy",
        neighbor_column="neighbor_strategy"
    )
    prompt_cluster_assignments = build_cluster_assignments(
        metadata_table=prompt_table,
        embedding_matrix=embedding_matrix,
        id_columns=[
            "strategy",
            "prompt_id",
            "prompt_key",
            "local_cache_path"
        ],
        cluster_ks=args.cluster_ks,
        embedding_backend="gemini_text_embedding_004"
    )
    prompt_cluster_summary = summarize_cluster_assignments(
        assignment_table=prompt_cluster_assignments,
        group_columns=["strategy"],
        reference_column=None,
        reference_value=None
    )
    prompt_cluster_summary["prompt_count"] = prompt_cluster_summary["rows"]
    prompt_cluster_summary_median = summarize_cluster_medians(
        cluster_summary=prompt_cluster_summary,
        group_columns=["strategy"]
    )
    prompt_cluster_summary_median["prompt_count"] = prompt_cluster_summary_median["rows"]
    logger.info("Loading audio embeddings")
    audio_table, audio_embedding_matrix = load_audio_embedding_inputs(
        audio_embedding_manifest_csv=args.audio_embedding_manifest_csv,
        audio_embedding_npy=args.audio_embedding_npy,
        mir_csv=args.mir_csv,
        prompt_strategies=prompt_table["strategy"].unique().tolist()
    )
    logger.info("Building audio embedding summaries")
    (
        audio_diversity_summary,
        audio_strategy_pairwise_summary,
        audio_model_pairwise_summary
    ) = build_audio_embedding_summaries(
        audio_table=audio_table,
        embedding_matrix=audio_embedding_matrix
    )
    audio_neighbor_indices, audio_prototype_count = build_prototype_neighbor_indices(
        embedding_matrix=audio_embedding_matrix,
        neighbor_k=args.neighbor_k,
        random_state=20260418
    )
    audio_neighbor_summary = summarize_audio_neighbors(
        audio_table=audio_table,
        neighbor_indices=audio_neighbor_indices,
        embedding_backend="clap_audio_laion",
        prototype_count=audio_prototype_count
    )
    audio_neighbor_strategy_mixing_summary = summarize_neighbor_mixing(
        metadata_table=audio_table,
        neighbor_indices=audio_neighbor_indices,
        group_columns=["strategy", "model"],
        category_column="strategy",
        neighbor_column="neighbor_strategy"
    )
    audio_neighbor_model_mixing_summary = summarize_neighbor_mixing(
        metadata_table=audio_table,
        neighbor_indices=audio_neighbor_indices,
        group_columns=["strategy", "model"],
        category_column="model",
        neighbor_column="neighbor_model"
    )
    audio_cluster_assignments = build_cluster_assignments(
        metadata_table=audio_table,
        embedding_matrix=audio_embedding_matrix,
        id_columns=[
            "join_key",
            "strategy",
            "model",
            "prompt_id",
            "line_number",
            "audio_filename"
        ],
        cluster_ks=args.cluster_ks,
        embedding_backend="clap_audio_laion"
    )
    audio_cluster_summary = summarize_cluster_assignments(
        assignment_table=audio_cluster_assignments,
        group_columns=["strategy", "model"],
        reference_column="model",
        reference_value="real"
    )
    audio_cluster_summary["audio_count"] = audio_cluster_summary["rows"]
    audio_cluster_summary_median = summarize_cluster_medians(
        cluster_summary=audio_cluster_summary,
        group_columns=["strategy", "model"]
    )
    audio_cluster_summary_median["audio_count"] = audio_cluster_summary_median["rows"]
    audio_diversity_summary = audio_diversity_summary.merge(
        audio_neighbor_summary,
        on=["strategy", "model"],
        how="left",
        validate="one_to_one"
    )

    logger.info("Writing prompt-analysis CSVs")
    write_csv(
        prompt_diversity_summary,
        os.path.join(args.output_dir, "prompt_strategy_diversity_summary.csv")
    )
    write_csv(
        pairwise_similarity_summary,
        os.path.join(args.output_dir, "prompt_strategy_pairwise_similarity.csv")
    )
    write_csv(
        neighbor_mixing_summary,
        os.path.join(args.output_dir, "prompt_neighbor_mixing_summary.csv")
    )
    write_csv(
        neighbor_summary,
        os.path.join(args.output_dir, "prompt_embedding_neighbor_summary.csv")
    )
    write_csv(
        prompt_cluster_assignments,
        os.path.join(args.output_dir, "prompt_embedding_cluster_assignments.csv")
    )
    write_csv(
        prompt_cluster_summary,
        os.path.join(args.output_dir, "prompt_embedding_cluster_summary.csv")
    )
    write_csv(
        prompt_cluster_summary_median,
        os.path.join(args.output_dir, "prompt_embedding_cluster_summary_median.csv")
    )
    write_csv(
        audio_diversity_summary,
        os.path.join(args.output_dir, "audio_embedding_diversity_summary.csv")
    )
    write_csv(
        audio_strategy_pairwise_summary,
        os.path.join(args.output_dir, "audio_embedding_strategy_pairwise_similarity.csv")
    )
    write_csv(
        audio_model_pairwise_summary,
        os.path.join(args.output_dir, "audio_embedding_model_pairwise_similarity.csv")
    )
    write_csv(
        audio_neighbor_strategy_mixing_summary,
        os.path.join(args.output_dir, "audio_embedding_neighbor_strategy_mixing_summary.csv")
    )
    write_csv(
        audio_neighbor_model_mixing_summary,
        os.path.join(args.output_dir, "audio_embedding_neighbor_model_mixing_summary.csv")
    )
    write_csv(
        audio_neighbor_summary,
        os.path.join(args.output_dir, "audio_embedding_neighbor_summary.csv")
    )
    write_csv(
        audio_cluster_assignments,
        os.path.join(args.output_dir, "audio_embedding_cluster_assignments.csv")
    )
    write_csv(
        audio_cluster_summary,
        os.path.join(args.output_dir, "audio_embedding_cluster_summary.csv")
    )
    write_csv(
        audio_cluster_summary_median,
        os.path.join(args.output_dir, "audio_embedding_cluster_summary_median.csv")
    )
    logger.info("Finished prompt-analysis CSVs")


def load_prompt_embedding_inputs(
    prompt_metrics_csv: str,
    embedding_manifest_csv: str,
    embedding_npy: str
) -> tuple[pd.DataFrame, np.ndarray]:
    prompt_table = pd.read_csv(prompt_metrics_csv)
    embedding_manifest = pd.read_csv(embedding_manifest_csv)
    embedding_matrix = np.load(embedding_npy)

    prompt_table["prompt_text"] = prompt_table["prompt_text"].fillna("")
    ordered_table = embedding_manifest.merge(
        prompt_table,
        on="prompt_key",
        how="left",
        validate="one_to_one"
    )

    if len(ordered_table) != embedding_matrix.shape[0]:
        raise ValueError(
            "Embedding rows (%s) do not match prompt rows (%s)" % (
                embedding_matrix.shape[0],
                len(ordered_table)
            )
        )

    ordered_table["strategy"] = ordered_table["strategy"].astype(str)
    ordered_table["prompt_id"] = ordered_table["prompt_id"].astype(str)

    return ordered_table, l2_normalize_rows(embedding_matrix)


def load_audio_embedding_inputs(
    audio_embedding_manifest_csv: str,
    audio_embedding_npy: str,
    mir_csv: str,
    prompt_strategies: list[str]
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
    ].copy()
    ordered_table = audio_manifest.merge(
        audio_metadata,
        on="join_key",
        how="left",
        validate="one_to_one"
    )

    finite_embedding_mask = np.isfinite(embedding_matrix).all(axis=1)
    strategy_mask = ordered_table["strategy"].isin(prompt_strategies).to_numpy()
    keep_mask = finite_embedding_mask & strategy_mask

    ordered_table = ordered_table.loc[keep_mask].reset_index(drop=True).copy()
    embedding_matrix = embedding_matrix[keep_mask]
    ordered_table["strategy"] = ordered_table["strategy"].astype(str)
    ordered_table["model"] = ordered_table["model"].astype(str)
    ordered_table["prompt_id"] = ordered_table["prompt_id"].astype(str)

    return ordered_table, l2_normalize_rows(embedding_matrix)


def build_strategy_embedding_summaries(
    prompt_table: pd.DataFrame,
    embedding_matrix: np.ndarray,
    neighbor_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = sorted(prompt_table["strategy"].unique().tolist())
    strategy_stats = []
    pairwise_rows = []
    strategy_vectors = {}
    strategy_sum_vectors = {}
    strategy_centroids = {}

    for strategy in strategies:
        strategy_mask = prompt_table["strategy"] == strategy
        strategy_rows = prompt_table.loc[strategy_mask].reset_index(drop=True)
        strategy_embeddings = embedding_matrix[strategy_mask.to_numpy()]
        strategy_vectors[strategy] = strategy_embeddings
        strategy_sum_vectors[strategy] = strategy_embeddings.sum(axis=0)
        strategy_centroids[strategy] = normalize_vector(strategy_sum_vectors[strategy])

    neighbor_lookup = build_neighbor_lookup(neighbor_summary)

    for strategy in strategies:
        strategy_rows = prompt_table.loc[prompt_table["strategy"] == strategy].reset_index(drop=True)
        strategy_embeddings = strategy_vectors[strategy]
        prompt_count = int(len(strategy_rows))
        sum_vector = strategy_sum_vectors[strategy]
        centroid_vector = strategy_centroids[strategy]
        within_mean_cosine = compute_within_mean_cosine(sum_vector, prompt_count)
        within_distance_interval = compute_within_cosine_distance_analytic_interval(
            strategy_embeddings
        )
        centroid_similarities = strategy_embeddings @ centroid_vector
        centroid_mean_cosine = float(np.mean(centroid_similarities))

        prompt_text_counts = strategy_rows["prompt_text"].value_counts(dropna=False)
        distinct_text_count = int(len(prompt_text_counts))
        distinct_text_share = float(distinct_text_count / prompt_count)
        duplicate_pair_share = compute_duplicate_pair_share(prompt_text_counts)
        effective_text_count = compute_effective_text_count(prompt_text_counts)
        nonempty_prompt_share = float(np.mean(strategy_rows["prompt_text"].str.len().to_numpy() > 0))
        unique_text_indices = strategy_rows.drop_duplicates(
            subset=["prompt_text"],
            keep="first"
        ).index.to_numpy()
        unique_text_embeddings = strategy_embeddings[unique_text_indices]
        effective_rank_centered = compute_roy_vetterli_effective_rank(
            strategy_embeddings,
            center=True
        )
        effective_rank_centered_unique_text = compute_roy_vetterli_effective_rank(
            unique_text_embeddings,
            center=True
        )
        effective_rank_raw = compute_roy_vetterli_effective_rank(
            strategy_embeddings,
            center=False
        )
        effective_rank_raw_unique_text = compute_roy_vetterli_effective_rank(
            unique_text_embeddings,
            center=False
        )

        row = {
            "strategy": strategy,
            "prompt_count": prompt_count,
            "distinct_prompt_text_count": distinct_text_count,
            "nonempty_prompt_share": nonempty_prompt_share,
            "distinct_prompt_text_share": distinct_text_share,
            "duplicate_prompt_pair_share": duplicate_pair_share,
            "effective_prompt_text_count": effective_text_count,
            "mean_prompt_word_count": float(strategy_rows["prompt_word_count"].mean()),
            "mean_prompt_token_entropy": float(strategy_rows["prompt_token_entropy"].mean()),
            "effective_rank_centered": effective_rank_centered,
            "effective_rank_centered_unique_text": effective_rank_centered_unique_text,
            "effective_rank_raw": effective_rank_raw,
            "effective_rank_raw_unique_text": effective_rank_raw_unique_text,
            "within_mean_cosine": within_mean_cosine,
            "within_mean_cosine_distance": float(1.0 - within_mean_cosine),
            "within_mean_cosine_distance_lci": within_distance_interval["lci"],
            "within_mean_cosine_distance_uci": within_distance_interval["uci"],
            "centroid_mean_cosine": centroid_mean_cosine,
            "centroid_mean_cosine_distance": float(1.0 - centroid_mean_cosine),
            "same_strategy_neighbor_share": neighbor_lookup[strategy]["same_strategy_share"],
            "neighbor_strategy_entropy": neighbor_lookup[strategy]["neighbor_strategy_entropy"],
            "effective_neighbor_strategy_count": neighbor_lookup[strategy]["effective_neighbor_strategy_count"]
        }
        strategy_stats.append(row)

    for left_index, left_strategy in enumerate(strategies):
        for right_strategy in strategies[left_index + 1 : ]:
            left_count = strategy_vectors[left_strategy].shape[0]
            right_count = strategy_vectors[right_strategy].shape[0]
            between_mean_cosine = float(
                strategy_sum_vectors[left_strategy] @ strategy_sum_vectors[right_strategy]
            ) / float(left_count * right_count)
            centroid_cosine = float(strategy_centroids[left_strategy] @ strategy_centroids[right_strategy])
            left_within = next(
                row["within_mean_cosine"]
                for row in strategy_stats
                if row["strategy"] == left_strategy
            )
            right_within = next(
                row["within_mean_cosine"]
                for row in strategy_stats
                if row["strategy"] == right_strategy
            )
            pairwise_rows.append({
                "strategy_1": left_strategy,
                "strategy_2": right_strategy,
                "prompt_count_1": left_count,
                "prompt_count_2": right_count,
                "between_mean_cosine": clamp_unit_interval_value(between_mean_cosine),
                "centroid_cosine": clamp_unit_interval_value(centroid_cosine),
                "within_mean_cosine_1": left_within,
                "within_mean_cosine_2": right_within,
                "separation_gap_1": float(left_within - between_mean_cosine),
                "separation_gap_2": float(right_within - between_mean_cosine),
                "internal_similarity_advantage": float(min(left_within, right_within) - between_mean_cosine)
            })

    diversity_summary = pd.DataFrame(strategy_stats).sort_values(
        ["within_mean_cosine_distance", "distinct_prompt_text_share"],
        ascending=[False, False],
        kind="stable"
    ).reset_index(drop=True)
    pairwise_summary = pd.DataFrame(pairwise_rows).sort_values(
        ["internal_similarity_advantage", "between_mean_cosine"],
        ascending=[False, False],
        kind="stable"
    ).reset_index(drop=True)

    return diversity_summary, pairwise_summary


def build_audio_embedding_summaries(
    audio_table: pd.DataFrame,
    embedding_matrix: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_keys = sorted(
        audio_table.loc[:, ["model", "strategy"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    cell_vectors = {}
    cell_sum_vectors = {}
    cell_centroids = {}

    for model, strategy in cell_keys:
        cell_mask = (audio_table["model"] == model) & (audio_table["strategy"] == strategy)
        cell_embeddings = embedding_matrix[cell_mask.to_numpy()]
        cell_vectors[(model, strategy)] = cell_embeddings
        cell_sum_vectors[(model, strategy)] = cell_embeddings.sum(axis=0)
        cell_centroids[(model, strategy)] = normalize_vector(cell_sum_vectors[(model, strategy)])

    diversity_rows = []
    strategy_pairwise_rows = []
    model_pairwise_rows = []

    for model, strategy in cell_keys:
        cell_embeddings = cell_vectors[(model, strategy)]
        audio_count = int(cell_embeddings.shape[0])
        effective_rank_centered = compute_roy_vetterli_effective_rank(
            cell_embeddings,
            center=True
        )
        effective_rank_raw = compute_roy_vetterli_effective_rank(
            cell_embeddings,
            center=False
        )
        within_mean_cosine = compute_within_mean_cosine(
            cell_sum_vectors[(model, strategy)],
            audio_count
        )
        within_distance_interval = compute_within_cosine_distance_analytic_interval(
            cell_embeddings
        )
        centroid_similarities = cell_embeddings @ cell_centroids[(model, strategy)]
        centroid_mean_cosine = float(np.mean(centroid_similarities))

        diversity_rows.append({
            "strategy": strategy,
            "model": model,
            "audio_count": audio_count,
            "effective_rank_centered": effective_rank_centered,
            "effective_rank_raw": effective_rank_raw,
            "within_mean_cosine": within_mean_cosine,
            "within_mean_cosine_distance": float(1.0 - within_mean_cosine),
            "within_mean_cosine_distance_lci": within_distance_interval["lci"],
            "within_mean_cosine_distance_uci": within_distance_interval["uci"],
            "centroid_mean_cosine": centroid_mean_cosine,
            "centroid_mean_cosine_distance": float(1.0 - centroid_mean_cosine)
        })

    for model in sorted(audio_table["model"].unique().tolist()):
        model_keys = [
            cell_key
            for cell_key in cell_keys
            if cell_key[0] == model
        ]
        strategy_pairwise_rows.extend(build_pairwise_embedding_rows(
            group_name="model",
            group_value=model,
            pair_name="strategy",
            keys=model_keys,
            label_index=1,
            vectors=cell_vectors,
            sum_vectors=cell_sum_vectors,
            centroids=cell_centroids,
            diversity_rows=diversity_rows
        ))

    for strategy in sorted(audio_table["strategy"].unique().tolist()):
        strategy_keys = [
            cell_key
            for cell_key in cell_keys
            if cell_key[1] == strategy
        ]
        model_pairwise_rows.extend(build_pairwise_embedding_rows(
            group_name="strategy",
            group_value=strategy,
            pair_name="model",
            keys=strategy_keys,
            label_index=0,
            vectors=cell_vectors,
            sum_vectors=cell_sum_vectors,
            centroids=cell_centroids,
            diversity_rows=diversity_rows
        ))

    return (
        pd.DataFrame(diversity_rows).sort_values(
            ["model", "effective_rank_centered", "strategy"],
            kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(strategy_pairwise_rows).sort_values(
            ["model", "internal_similarity_advantage", "between_mean_cosine"],
            ascending=[True, False, False],
            kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(model_pairwise_rows).sort_values(
            ["strategy", "internal_similarity_advantage", "between_mean_cosine"],
            ascending=[True, False, False],
            kind="stable"
        ).reset_index(drop=True)
    )


def build_pairwise_embedding_rows(
    group_name: str,
    group_value: str,
    pair_name: str,
    keys: list[tuple[str, str]],
    label_index: int,
    vectors: dict[tuple[str, str], np.ndarray],
    sum_vectors: dict[tuple[str, str], np.ndarray],
    centroids: dict[tuple[str, str], np.ndarray],
    diversity_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    within_lookup = {
        (str(row["model"]), str(row["strategy"])): float(row["within_mean_cosine"])
        for row in diversity_rows
    }

    for left_index, left_key in enumerate(keys):
        for right_key in keys[left_index + 1 : ]:
            left_count = vectors[left_key].shape[0]
            right_count = vectors[right_key].shape[0]
            between_mean_cosine = float(sum_vectors[left_key] @ sum_vectors[right_key]) / float(
                left_count * right_count
            )
            centroid_cosine = float(centroids[left_key] @ centroids[right_key])
            left_within = within_lookup[left_key]
            right_within = within_lookup[right_key]
            rows.append({
                group_name: group_value,
                "%s_1" % pair_name: left_key[label_index],
                "%s_2" % pair_name: right_key[label_index],
                "%s_count_1" % pair_name: left_count,
                "%s_count_2" % pair_name: right_count,
                "between_mean_cosine": clamp_unit_interval_value(between_mean_cosine),
                "centroid_cosine": clamp_unit_interval_value(centroid_cosine),
                "within_mean_cosine_1": left_within,
                "within_mean_cosine_2": right_within,
                "separation_gap_1": float(left_within - between_mean_cosine),
                "separation_gap_2": float(right_within - between_mean_cosine),
                "internal_similarity_advantage": float(min(left_within, right_within) - between_mean_cosine)
            })

    return rows


def build_neighbor_lookup(
    neighbor_summary: pd.DataFrame
) -> dict[str, dict[str, float]]:
    return {
        str(row["strategy"]): {
            "same_strategy_share": float(row["same_strategy_share"]),
            "neighbor_strategy_entropy": float(row["neighbor_strategy_entropy"]),
            "effective_neighbor_strategy_count": float(row["effective_neighbor_strategy_count"])
        }
        for _, row in neighbor_summary.iterrows()
    }


def compute_duplicate_pair_share(
    prompt_text_counts: pd.Series
) -> float:
    counts = prompt_text_counts.to_numpy(dtype=float)
    prompt_count = float(counts.sum())
    if prompt_count <= 1.0:
        return 0.0
    duplicate_pairs = float(np.sum(counts * (counts - 1.0)))
    return duplicate_pairs / float(prompt_count * (prompt_count - 1.0))


def compute_effective_text_count(
    prompt_text_counts: pd.Series
) -> float:
    counts = prompt_text_counts.to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    entropy = compute_entropy(probabilities)
    return float(np.exp(entropy))


def compute_entropy(
    probabilities: np.ndarray
) -> float:
    nonzero_probabilities = probabilities[probabilities > 0]
    return -float(np.sum(nonzero_probabilities * np.log(nonzero_probabilities)))


def compute_roy_vetterli_effective_rank(
    matrix: np.ndarray,
    center: bool
) -> float:
    working_matrix = matrix.astype(float, copy=True)
    if center:
        working_matrix = working_matrix - working_matrix.mean(axis=0, keepdims=True)

    if working_matrix.shape[0] >= working_matrix.shape[1]:
        spectrum_matrix = working_matrix.T @ working_matrix
    else:
        spectrum_matrix = working_matrix @ working_matrix.T

    singular_values = np.sqrt(
        np.maximum(
            np.linalg.eigvalsh(spectrum_matrix),
            0.0
        )
    )
    max_singular_value = float(np.max(singular_values))
    rank_tolerance = float(
        max(working_matrix.shape) * np.finfo(float).eps * max_singular_value
    )
    nonzero_singular_values = singular_values[singular_values > rank_tolerance]
    singular_value_sum = float(nonzero_singular_values.sum())
    if singular_value_sum == 0.0:
        return 0.0

    probabilities = nonzero_singular_values / singular_value_sum
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def compute_within_cosine_distance_analytic_interval(
    matrix: np.ndarray
) -> dict[str, float]:
    # Analytic 95% CI for the within-group mean cosine distance on L2-normalized rows.
    # The mean pairwise cosine similarity over the n*(n-1) ordered off-diagonal pairs
    # is a degree-2 U-statistic with kernel h(x, y) = <x, y>. Hoeffding's ANOVA
    # decomposition gives Var(U_n) ~ 4 * zeta_1 / n for large n, where
    # zeta_1 = Var_X(g(X)) and g(x) = E_Y[<x, Y>]. We estimate g at each row by its
    # leave-one-out mean similarity to the rest: g_hat(x_i) = (<x_i, S> - 1)/(n - 1)
    # where S = sum_j x_j and <x_i, x_i> = 1 since rows are unit-normalized.
    # The CI for the mean cosine *distance* = 1 - theta has the same half-width
    # since it's an affine transform of the similarity. For very small n (<~20)
    # this Wald interval will understate uncertainty.
    row_count = matrix.shape[0]
    if row_count <= 1:
        return {
            "lci": np.nan,
            "uci": np.nan
        }

    sum_vector = matrix.sum(axis=0)
    leave_one_out_mean = (matrix @ sum_vector - 1.0) / float(row_count - 1)
    g_variance = float(np.var(leave_one_out_mean, ddof=1))
    u_statistic_variance = 4.0 * g_variance / float(row_count)
    # 1.959963984540054 is the standard-normal 97.5th percentile (two-sided 95%).
    half_width = 1.959963984540054 * float(np.sqrt(u_statistic_variance))

    sum_norm_sq = float(sum_vector @ sum_vector)
    mean_cosine = (sum_norm_sq - row_count) / float(row_count * (row_count - 1))
    mean_distance = float(1.0 - clamp_unit_interval_value(mean_cosine))
    return {
        "lci": mean_distance - half_width,
        "uci": mean_distance + half_width
    }


def compute_within_mean_cosine(
    sum_vector: np.ndarray,
    prompt_count: int
) -> float:
    if prompt_count <= 1:
        return np.nan
    sum_norm_sq = float(sum_vector @ sum_vector)
    within_mean_cosine = (sum_norm_sq - prompt_count) / float(prompt_count * (prompt_count - 1))
    return clamp_unit_interval_value(within_mean_cosine)


def l2_normalize_rows(
    embedding_matrix: np.ndarray
) -> np.ndarray:
    row_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return embedding_matrix / row_norms


def normalize_vector(
    vector: np.ndarray
) -> np.ndarray:
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0:
        return vector.copy()
    return vector / vector_norm


def clamp_unit_interval_value(
    value: float
) -> float:
    return float(min(1.0, max(-1.0, value)))


def write_csv(
    table: pd.DataFrame,
    output_path: str
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    table.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
