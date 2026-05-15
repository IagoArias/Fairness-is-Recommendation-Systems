from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity

from src.eval.metrics import awpd, laplace_log_p, novelty_score, relevance_score


def evaluate_run_awpd_observations(
    ratings_long: pd.DataFrame,
    ratings_by_model: Dict[str, List[pd.DataFrame]],
    user_clusters: np.ndarray,
    item_clusters: np.ndarray,
    affinity: np.ndarray,
    num_users: int = None,
    popularity: np.ndarray = None,
) -> pd.DataFrame:
    """
    Calls awpd() for each (model, run) and returns one row per (Model, Run, UserID).

    Columns: Model, Run, UserID, AWPD, UPD, Misalignment.
    """
    num_users = len(user_clusters)
    num_items = len(item_clusters)

    if popularity is None:
        n_users_train = ratings_long["UserID"].nunique() or 1
        item_counts = ratings_long.groupby("MovieID")["UserID"].nunique()
        popularity = np.array(
            [item_counts.get(i, 0) / n_users_train for i in range(num_items)],
            dtype=float,
        )

    # UPD measures deviation from the user's *original* taste profile, so history
    # is fixed at run 0 (the initial training data) for the entire simulation.
    history_run0_map = ratings_long.groupby("UserID")["MovieID"].apply(list).to_dict()
    history_run0 = [history_run0_map.get(u, []) for u in range(num_users)]

    rows = []
    for model_name, model_runs in ratings_by_model.items():
        previous_df = ratings_long
        for run_idx, current_df in enumerate(model_runs):
            new_rows = current_df.iloc[len(previous_df):]

            rec_map = new_rows.groupby("UserID")["MovieID"].apply(list).to_dict()
            recommendations = [rec_map.get(u, []) for u in range(num_users)]

            result = awpd(
                user_clusters=user_clusters,
                item_clusters=item_clusters,
                affinity=affinity,
                popularity=popularity,
                history=history_run0,
                recommendations=recommendations,
            )

            for u, has_recs in enumerate(recommendations):
                if not has_recs:
                    continue
                rows.append({
                    "Model":        model_name,
                    "Run":          run_idx,
                    "UserID":       u,
                    "AWPD":         float(result["per_user_awpd"][u]),
                    "UPD":          float(result["upd"][u]),
                    "Misalignment": float(result["misalignment"][u]),
                })

            previous_df = current_df

    return pd.DataFrame(rows)


def _precompute_item_similarity(ratings_long: pd.DataFrame, num_users: int, num_items: int) -> np.ndarray:
    """Cosine similarity between item vectors built from ratings_long."""
    R = np.zeros((num_users, num_items), dtype=float)
    for row in ratings_long.itertuples(index=False):
        R[row.UserID, row.MovieID] = row.Rating
    return cosine_similarity(R.T)  # (I, I)


def evaluate_run_quality_observations(
    ratings_long: pd.DataFrame,
    ratings_by_model: Dict[str, List[pd.DataFrame]],
    num_users: int,
    num_items: int,
) -> pd.DataFrame:
    """
    Computes Novelty, Relevance, and Diversity for each (model, run) per user.

    Novelty(u)   = mean(-log2(P[j])) over recommended items j  (higher = more novel)
    Relevance(u) = mean actual rating over recommended items j
    Diversity(u) = mean(1 - sim(i, j)) over all pairs in recommended items
                   (item similarity precomputed once from initial ratings_long)

    Columns: Model, Run, UserID, Novelty, Relevance, Diversity.
    """
    n_train_users = ratings_long["UserID"].nunique() or 1
    item_counts_series = ratings_long.groupby("MovieID")["UserID"].nunique()
    item_counts = np.array([item_counts_series.get(i, 0) for i in range(num_items)], dtype=float)
    log_P = laplace_log_p(item_counts, n_train_users)  # precomputed once

    S = _precompute_item_similarity(ratings_long, num_users, num_items)  # precomputed once

    rows = []
    for model_name, model_runs in ratings_by_model.items():
        previous_df = ratings_long
        for run_idx, current_df in enumerate(model_runs):
            new_rows = current_df.iloc[len(previous_df):]

            for u, grp in new_rows.groupby("UserID"):
                items = grp["MovieID"].to_numpy()

                novelty = novelty_score(items, log_P)
                relevance = relevance_score(grp["Rating"].to_numpy())

                if items.size >= 2:
                    sim_matrix = S[np.ix_(items, items)]
                    i_idx, j_idx = np.triu_indices(items.size, k=1)
                    diversity = float((1.0 - sim_matrix[i_idx, j_idx]).mean())
                else:
                    diversity = 0.0

                rows.append({
                    "Model":     model_name,
                    "Run":       run_idx,
                    "UserID":    u,
                    "Novelty":   novelty,
                    "Relevance": relevance,
                    "Diversity": diversity,
                })

            previous_df = current_df

    return pd.DataFrame(rows)
