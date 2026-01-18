import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity

from src.eval.metrics import evaluate_all_metrics


def compute_similarity_matrix(R_train: np.ndarray, user_means: np.ndarray) -> np.ndarray:
    """User-user Pearson similarity using sparse matrices."""
    U, I = R_train.shape

    rows, cols = np.where(~np.isnan(R_train))
    vals = R_train[rows, cols].astype(np.float32)

    data = vals - user_means[rows].astype(np.float32)

    C = sp.csr_matrix((data, (rows, cols)), shape=(U, I))
    M = sp.csr_matrix((np.ones_like(data, dtype=np.float32), (rows, cols)), shape=(U, I))

    N = (C @ C.T).toarray().astype(np.float32)
    S = (C.multiply(C) @ M.T).toarray().astype(np.float32)

    denom = np.sqrt(S * S.T)
    sim = np.divide(N, denom, out=np.zeros_like(N), where=denom > 0)
    np.fill_diagonal(sim, 0.0)
    return sim


def predict_user_ratings(u: int, R_train: np.ndarray, similarity_matrix: np.ndarray, user_means: np.ndarray, k, rated_by_item):
    """
    Predict missing ratings for user u using user-based CF with Pearson similarity.
    rated_by_item[j]: np.array of users that rated item j in train.
    """
    num_items = R_train.shape[1]
    preds = np.copy(R_train[u])
    sims_u = similarity_matrix[u]
    missing_items = np.where(np.isnan(R_train[u]))[0]
    for j in missing_items:
        neighbors = rated_by_item[j]
        if neighbors.size == 0:
            preds[j] = user_means[u]
            continue

        neighbors = neighbors[neighbors != u]
        if neighbors.size == 0:
            preds[j] = user_means[u]
            continue

        sims = sims_u[neighbors]

        if k is not None and neighbors.size > k:
            top_idx = np.argpartition(np.abs(sims), -k)[-k:]
            neighbors = neighbors[top_idx]
            sims = sims[top_idx]

        num = np.sum(sims * (R_train[neighbors, j] - user_means[neighbors]))
        denom = np.sum(np.abs(sims))

        preds[j] = user_means[u] if denom == 0 else user_means[u] + num / denom

    return preds


def predict_all_users_user_based(R_train: np.ndarray, k=None, n_jobs: int = -1) -> np.ndarray:
    """Compute user-based CF predictions for all users."""
    num_users, num_items = R_train.shape
    global_mean_value = np.nanmean(R_train) if np.any(~np.isnan(R_train)) else 0.0
    user_means = np.array(
        [np.nanmean(R_train[u]) if np.any(~np.isnan(R_train[u])) else global_mean_value for u in range(num_users)]
    )

    sim_matrix = compute_similarity_matrix(R_train, user_means)
    rated_by_item = [np.where(~np.isnan(R_train[:, j]))[0] for j in range(num_items)]

    R_pred_rows = Parallel(n_jobs=n_jobs)(
        delayed(predict_user_ratings)(u, R_train, sim_matrix, user_means, k, rated_by_item) for u in range(num_users)
    )
    return np.vstack(R_pred_rows)


def compute_item_similarity_cosine(R_train: np.ndarray) -> np.ndarray:
    """Item-item cosine similarity on NaN-filled user-item matrix."""
    R_filled = np.nan_to_num(R_train, nan=0.0)
    return cosine_similarity(R_filled.T)


def predict_all_users_item_based(R_train: np.ndarray, item_sim: np.ndarray, k=None) -> np.ndarray:
    """Compute item-based CF predictions for all users."""
    num_users, num_items = R_train.shape
    out = np.copy(R_train)
    for u in range(num_users):
        rated = ~np.isnan(R_train[u])
        rated_idx = np.where(rated)[0]
        rated_r = R_train[u, rated_idx]
        for j in range(num_items):
            if np.isnan(R_train[u, j]):
                sims = item_sim[j, rated_idx]
                if rated_idx.size == 0:
                    out[u, j] = np.nanmean(R_train[:, j]) if np.any(~np.isnan(R_train[:, j])) else 0.0
                    continue
                if k is not None and rated_idx.size > k:
                    top_idx = np.argpartition(np.abs(sims), -k)[-k:]
                    sims_k = sims[top_idx]
                    rr = rated_r[top_idx]
                else:
                    sims_k = sims
                    rr = rated_r
                denom = np.sum(np.abs(sims_k))
                out[u, j] = (np.sum(sims_k * rr) / denom) if denom > 0 else (np.nanmean(R_train[:, j]) if np.any(~np.isnan(R_train[:, j])) else 0.0)
    return out


def evaluate_user_based_cf(
    folds,
    k_values=None,
    topn_nov_rel: int = 20,
    topn_div: int = 5,
    k_ndcg: int = 20,
    n_jobs: int = -1,
):
    """Run user-based CF across folds.
    Returns:
      metrics_df: one row per (fold, k)
      avg_df: averaged across folds per k (and Model/Similarity)
    """
    k_values = k_values or [50, 150, 500, None]
    all_folds_metrics = []

    for fold_num, (train_df, test_df) in enumerate(folds, start=1):
        R_train = train_df.values
        R_test = test_df.values
        num_users, num_items = R_train.shape

        global_mean_value = np.nanmean(R_train) if np.any(~np.isnan(R_train)) else 0.0
        user_means = np.array(
            [
                np.nanmean(R_train[u]) if np.any(~np.isnan(R_train[u])) else global_mean_value
                for u in range(num_users)
            ]
        )

        sim_matrix = compute_similarity_matrix(R_train, user_means)
        rated_by_item = [np.where(~np.isnan(R_train[:, j]))[0] for j in range(num_items)]

        for k in k_values:
            R_pred_rows = Parallel(n_jobs=n_jobs)(
                delayed(predict_user_ratings)(
                    u, R_train, sim_matrix, user_means, k, rated_by_item
                )
                for u in range(num_users)
            )
            R_pred = np.vstack(R_pred_rows)

            metrics = evaluate_all_metrics(
                R_train,
                R_test,
                R_pred,
                train_df,
                topn_nov_rel=topn_nov_rel,
                topn_div=topn_div,
                k_ndcg=k_ndcg,
            )

            metrics.update(
                {
                    "Fold": fold_num,
                    "k_neighbors": "all" if k is None else k,
                    "Model": "Memory-based CF",
                    "Similarity": "Pearson (user-based)",
                }
            )
            all_folds_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_folds_metrics)

    group_cols = ["Model", "Similarity", "k_neighbors"]
    num_cols = metrics_df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "Fold"]

    avg_df = (
        metrics_df
        .groupby(group_cols, as_index=False)[num_cols]
        .mean()
    )

    return metrics_df, avg_df


def evaluate_item_based_cf(
    folds,
    k_values=None,
    topn_nov_rel: int = 20,
    topn_div: int = 5,
    k_ndcg: int = 20,
):
    """Run item-based CF across folds.
    Returns:
      metrics_df: one row per (fold, k)
      avg_df: averaged across folds per k (and Model/Similarity)
    """
    k_values = k_values or [50, 150, 500, None]
    all_folds_metrics_item = []

    for fold_num, (train_df, test_df) in enumerate(folds, start=1):
        R_train = train_df.values
        R_test = test_df.values

        item_sim = compute_item_similarity_cosine(R_train)

        for k in k_values:
            R_pred = predict_all_users_item_based(R_train, item_sim, k)
            metrics = evaluate_all_metrics(
                R_train,
                R_test,
                R_pred,
                train_df,
                topn_nov_rel=topn_nov_rel,
                topn_div=topn_div,
                k_ndcg=k_ndcg,
            )

            metrics.update(
                {
                    "Fold": fold_num,
                    "k_neighbors": "all" if k is None else k,
                    "Model": "Memory-based CF",
                    "Similarity": "Cosine (item-based)",
                }
            )
            all_folds_metrics_item.append(metrics)

    metrics_df = pd.DataFrame(all_folds_metrics_item)

    # Average across folds per k (and keep metadata columns)
    group_cols = ["Model", "Similarity", "k_neighbors"]
    num_cols = metrics_df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "Fold"]  # don't average Fold

    avg_df = (
        metrics_df
        .groupby(group_cols, as_index=False)[num_cols]
        .mean()
    )

    return metrics_df, avg_df

