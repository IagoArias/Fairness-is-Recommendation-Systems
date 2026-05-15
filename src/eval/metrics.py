from math import sqrt
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baselines import popularity_baseline


def get_test_indices(R_test: np.ndarray):
    """Positions where we have test ratings."""
    return list(zip(*np.where(~np.isnan(R_test))))


def dcg(relevances):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def laplace_log_p(item_counts: np.ndarray, num_users: int) -> np.ndarray:
    """Laplace-smoothed log2 popularity probability for each item."""
    P = (item_counts + 1.0) / (num_users + 2.0)
    return np.log2(np.clip(P, 1e-10, 1.0))


def novelty_score(item_indices: np.ndarray, log_P: np.ndarray) -> float:
    """Mean self-information over a set of items given precomputed log2(P)."""
    return float(-log_P[item_indices].mean())


def relevance_score(ratings: np.ndarray) -> float:
    """Mean rating over a set of items."""
    return float(ratings.mean())


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
    item_pop = np.sum(~np.isnan(R_train), axis=0)
    log_P = laplace_log_p(item_pop, num_users)

    novelty_u = []
    relevance_u = []

    for u in range(num_users):
        unknown_items = np.where(np.isnan(R_train[u]))[0]
        if unknown_items.size == 0:
            continue

        k = min(top_n, unknown_items.size)
        top_items = unknown_items[np.argsort(R_pred[u, unknown_items])[::-1][:k]]

        novelty_u.append(novelty_score(top_items, log_P))

        test_ratings = R_test[u, top_items]
        rated = test_ratings[~np.isnan(test_ratings)]
        if rated.size > 0:
            relevance_u.append(relevance_score(rated))

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


def _popularity_groups(
    item_counts: np.ndarray,
    h_frac: float = 0.2,
    t_frac: float = 0.2,
) -> np.ndarray:
    """
    Assign each item to a popularity group following Abdollahpouri et al. (2021).

    Returns an int8 array of length I where:
      2 = Head  — fewest most-popular items whose interactions sum to h_frac of total
      1 = Mid   — everything in between
      0 = Tail  — least-popular items whose interactions sum to t_frac of total
    """
    total = item_counts.sum()
    if total == 0:
        return np.ones(len(item_counts), dtype=np.int8)  # all Mid

    sorted_desc = np.argsort(item_counts)[::-1]
    cumsum_desc = np.cumsum(item_counts[sorted_desc])
    head_cut = int(np.searchsorted(cumsum_desc, h_frac * total, side="left"))

    sorted_asc = sorted_desc[::-1]
    cumsum_asc = np.cumsum(item_counts[sorted_asc])
    tail_cut = int(np.searchsorted(cumsum_asc, t_frac * total, side="left"))

    groups = np.ones(len(item_counts), dtype=np.int8)   # Mid = 1
    groups[sorted_desc[: head_cut + 1]] = 2             # Head = 2
    groups[sorted_asc[: tail_cut + 1]] = 0              # Tail = 0
    return groups


def awpd(
    user_clusters: np.ndarray,        # (U,)     user -> cluster index
    item_clusters: np.ndarray,        # (I,)     item -> cluster index
    affinity: np.ndarray,            
    popularity: np.ndarray,           # (I,)     item popularity, normalized [0,1]
    history: list[list[int]],         # (U,)     item indices per user (ragged)
    recommendations: list[list[int]], # (U,)     item indices per user
    rmse_per_user: np.ndarray = None  # (U,)     optional, for diagnostic plotting
) -> dict:
    U = len(history)

    affinity_normalized = 1 / (1 + np.exp(-affinity))
    all_items = np.concatenate([np.array(h) for h in history])
    item_counts       = np.bincount(all_items, minlength=len(popularity))
    empirical_popularity = item_counts / item_counts.max()  # [0,1], preserves Zipf shape

    # H/M/T groups for UPD (Abdollahpouri et al., 2021)
    item_groups = _popularity_groups(item_counts)

    upd_scores    = np.empty(U)
    misalignments = np.empty(U)
    scores        = np.empty(U)

    for u in range(U):
        uc   = user_clusters[u]
        hist = np.array(history[u])
        recs = np.array(recommendations[u])

        # UPD: JS divergence between H/M/T distribution in history vs. recommendations
        p = np.array([np.sum(item_groups[hist] == g) for g in range(3)], dtype=float)
        q = np.array([np.sum(item_groups[recs] == g) for g in range(3)], dtype=float)
        p /= p.sum() if p.sum() > 0 else 1.0
        q /= q.sum() if q.sum() > 0 else 1.0
        upd_scores[u] = float(jensenshannon(p, q))

        # Cluster misalignment: sigmoid handles the [-∞, +∞] → (0,1) mapping
        aff              = affinity_normalized[uc, item_clusters[recs]]
        misalignments[u] = (1 - aff).mean()

        # AWPD: Euclidean distance in (UPD, misalignment) space
        scores[u] = np.sqrt(upd_scores[u]**2 + misalignments[u]**2)

    # Aggregate by cluster
    cluster_ids = np.unique(user_clusters)
    by_cluster = {}
    for c in cluster_ids:
        mask = user_clusters == c
        entry = {
            "awpd":         float(scores[mask].mean()),
            "upd":          float(upd_scores[mask].mean()),
            "misalignment": float(misalignments[mask].mean()),
            "n_users":      int(mask.sum()),
        }
        if rmse_per_user is not None:
            entry["rmse"] = float(rmse_per_user[mask].mean())
        by_cluster[int(c)] = entry

    result = {
        "awpd":          float(scores.mean()),
        "per_user_awpd": scores,
        "upd":           upd_scores,
        "misalignment":  misalignments,
        "by_cluster":    by_cluster,
    }

    if rmse_per_user is not None:
        result["rmse"] = rmse_per_user

    return result