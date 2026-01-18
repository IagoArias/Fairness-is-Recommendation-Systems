from math import sqrt
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baselines import popularity_baseline


def get_test_indices(R_test: np.ndarray):
    """Positions where we have test ratings."""
    return list(zip(*np.where(~np.isnan(R_test))))


def dcg(relevances):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def eval_rmse_mae(R_test: np.ndarray, R_pred: np.ndarray) -> Tuple[float, float]:
    test_idx = get_test_indices(R_test)
    y_true = [R_test[u, j] for (u, j) in test_idx]
    y_pred = [R_pred[u, j] for (u, j) in test_idx]
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def eval_novelty_relevance(R_train: np.ndarray, R_test: np.ndarray, R_pred: np.ndarray, top_n: int = 20):
    """
    Novelty: avg(1 / (pop_j + 1)) over top-N recommended unknown items.
    Relevance: mean true rating (if any) on those top-N per user, then avg over users.
    """
    num_users, _ = R_train.shape

    novelty_u = []
    relevance_u = []

    for u in range(num_users):
        known = ~np.isnan(R_train[u])
        unknown_items = np.where(~known)[0]
        if unknown_items.size == 0:
            continue

        k = min(top_n, unknown_items.size)
        top_items = unknown_items[np.argsort(R_pred[u, unknown_items])[::-1][:k]]

        # Novelty
        item_pop = np.sum(~np.isnan(R_train), axis=0)
        P = (item_pop + 1.0) / (num_users + 2.0)  # Laplace smoothing
        novelty_u.append(np.mean(-np.log2(P[top_items])))

        # Relevance on test ratings (if exist)
        rel = [R_test[u, j] for j in top_items if not np.isnan(R_test[u, j])]
        if rel:
            relevance_u.append(np.mean(rel))

    novelty = float(np.mean(novelty_u)) if novelty_u else np.nan
    relevance = float(np.mean(relevance_u)) if relevance_u else np.nan
    return novelty, relevance


def eval_serendipity(R_train, R_test, R_pred, train_df, top_n: int = 20, tau: float = 3.5):
    """
    Ser(u) = (1/|R_u|) * sum_{i in R_u} (1 - base01(u,i)) * I[r_ui >= tau]
    R_u: top-N items unknown in train for u with rating in test.
    base01: popularity baseline rescaled to [0,1] (higher = more expected).
    """
    base15 = popularity_baseline(train_df).values.astype(float)

    # rescale to [0,1]
    base01 = (base15 - 1.0) / 4.0
    base01 = np.clip(base01, 0.0, 1.0)

    R_train = np.asarray(R_train, dtype=float)
    R_test = np.asarray(R_test, dtype=float)
    R_pred = np.asarray(R_pred, dtype=float)
    m, _ = R_train.shape
    unknown_mask = np.isnan(R_train)

    user_vals = []
    for u in range(m):
        cand = np.where(unknown_mask[u])[0]
        if cand.size == 0:
            continue

        k = min(top_n, cand.size)
        top = cand[np.argsort(R_pred[u, cand])[::-1][:k]]  # Top_u

        has_gt = ~np.isnan(R_test[u, top])  # only with ground truth in test
        Ru = top[has_gt]
        if Ru.size == 0:
            continue

        rel = (R_test[u, Ru] >= tau).astype(float)
        ser_u = float(np.mean((1.0 - base01[u, Ru]) * rel))
        user_vals.append(ser_u)

    return float(np.mean(user_vals)) if user_vals else np.nan


def compute_item_similarity(R_train: np.ndarray):
    """Cosine similarity between item vectors (using train, NaNs -> 0)."""
    R_filled = np.nan_to_num(R_train, nan=0.0)
    return cosine_similarity(R_filled.T)


def eval_diversity(R_train: np.ndarray, R_pred: np.ndarray, item_similarity, top_n: int = 5):
    """
    Diversity(u) = average_{i<j in Top_u} (1 - Sim(i,j)),
    with Top_u being top-N items among those unknown in train.
    Users with <2 items contribute 0. Returns mean over users.
    """
    if top_n <= 1:
        return 0.0

    R_train = np.asarray(R_train, dtype=float)
    R_pred = np.asarray(R_pred, dtype=float)
    S = np.asarray(item_similarity, dtype=float)

    num_users, _ = R_pred.shape
    div_scores = []

    R_pred_safe = np.where(np.isnan(R_pred), -np.inf, R_pred)

    for u in range(num_users):
        unknown = np.where(np.isnan(R_train[u]))[0]
        k = min(top_n, unknown.size)

        if k < 2:
            div_scores.append(0.0)
            continue

        top_items = unknown[np.argsort(R_pred_safe[u, unknown])[-k:]]

        s = 0.0
        cnt = 0
        for a in range(k):
            for b in range(a + 1, k):
                s += 1.0 - S[top_items[a], top_items[b]]
                cnt += 1

        div_scores.append(s / cnt)

    return float(np.mean(div_scores)) if div_scores else np.nan


def eval_ndcg(R_train: np.ndarray, R_test: np.ndarray, R_pred: np.ndarray, k: int = 20):
    """
    nDCG@k over unknown-in-train items.
    Utility: test rating if available, else 0.
    """
    num_users, _ = R_train.shape
    ndcgs = []

    for u in range(num_users):
        unknown_items = np.where(np.isnan(R_train[u]))[0]
        if unknown_items.size == 0:
            continue

        k_u = min(k, unknown_items.size)
        top_items = unknown_items[np.argsort(R_pred[u, unknown_items])[::-1][:k_u]]

        rel_pred = [R_test[u, j] if not np.isnan(R_test[u, j]) else 0.0 for j in top_items]
        dcg_u = dcg(rel_pred)

        true_utils = [R_test[u, j] if not np.isnan(R_test[u, j]) else 0.0 for j in unknown_items]
        ideal_rel = sorted(true_utils, reverse=True)[:k_u]
        idcg_u = dcg(ideal_rel)

        if idcg_u > 0:
            ndcgs.append(dcg_u / idcg_u)

    return float(np.mean(ndcgs)) if ndcgs else np.nan


def evaluate_all_metrics(
    R_train: np.ndarray,
    R_test: np.ndarray,
    R_pred: np.ndarray,
    train_df,
    topn_nov_rel: int = 20,
    topn_div: int = 5,
    k_ndcg: int = 20,
) -> Dict[str, float]:
    rmse, mae = eval_rmse_mae(R_test, R_pred)
    novelty, relevance = eval_novelty_relevance(R_train, R_test, R_pred, top_n=topn_nov_rel)
    serendipity = eval_serendipity(R_train, R_test, R_pred, train_df, top_n=topn_nov_rel)
    item_sim = compute_item_similarity(R_train)
    diversity = eval_diversity(R_train, R_pred, item_sim, top_n=topn_div)
    ndcg = eval_ndcg(R_train, R_test, R_pred, k=k_ndcg)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Novelty": novelty,
        "Relevance": relevance,
        "Serendipity": serendipity,
        "Diversity": diversity,
        "nDCG@20": ndcg,
    }
