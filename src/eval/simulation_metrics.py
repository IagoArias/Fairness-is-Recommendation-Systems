from __future__ import annotations

from math import sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.eval.metrics import awpd, dcg


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

    Columns: Model, Run, UserID, AWPD, PopDeviation, Misalignment.
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

    rows = []
    for model_name, model_runs in ratings_by_model.items():
        previous_df = ratings_long
        for run_idx, current_df in enumerate(model_runs):
            new_rows = current_df.iloc[len(previous_df):]

            history_map = previous_df.groupby("UserID")["MovieID"].apply(list).to_dict()
            history = [history_map.get(u, []) for u in range(num_users)]

            rec_map = new_rows.groupby("UserID")["MovieID"].apply(list).to_dict()
            recommendations = [rec_map.get(u, []) for u in range(num_users)]

            result = awpd(
                user_clusters=user_clusters,
                item_clusters=item_clusters,
                affinity=affinity,
                popularity=popularity,
                history=history,
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
                    "PopDeviation": float(result["pop_deviation"][u]),
                    "Misalignment": float(result["misalignment"][u]),
                })

            previous_df = current_df

    return pd.DataFrame(rows)
