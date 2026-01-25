import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm


def generate_mask_zipf_global(
    U: int,
    I: int,
    density: float = 0.01,
    alpha_user: float = 1.1,
    alpha_item: float = 1.2,
    seed: int = 0,
    oversample: float = 1.5,
    min_user_ratings: int | None = None,
    min_item_ratings: int | None = None,
    n_user_clusters: int | None = None,
    n_item_clusters: int | None = None,
    beta: float = 0.0,
    affinity: np.ndarray | None = None,
    return_clusters: bool = False,
):
    if U <= 0 or I <= 0:
        return [] if not return_clusters else ([], None, None, None)

    if min_user_ratings is not None and min_user_ratings < 0:
        raise ValueError("min_user_ratings must be >= 0 or None.")
    if min_item_ratings is not None and min_item_ratings < 0:
        raise ValueError("min_item_ratings must be >= 0 or None.")
    if min_user_ratings is not None and min_user_ratings > I:
        raise ValueError("min_user_ratings cannot exceed I (unique items per user).")
    if min_item_ratings is not None and min_item_ratings > U:
        raise ValueError("min_item_ratings cannot exceed U (unique users per item).")

    rng = np.random.default_rng(seed)
    M = int(round(density * U * I))
    M = max(0, min(M, U * I))
    if M == 0 and (min_user_ratings is None and min_item_ratings is None):
        return [] if not return_clusters else ([], None, None, None)

    ru = np.arange(1, U + 1, dtype=np.float64)
    pu = ru ** (-alpha_user)
    pu /= pu.sum()
    pu = pu[rng.permutation(U)]

    ri = np.arange(1, I + 1, dtype=np.float64)
    pi = ri ** (-alpha_item)
    pi /= pi.sum()
    pi = pi[rng.permutation(I)]

    use_clusters = (
        n_user_clusters is not None
        and n_item_clusters is not None
        and n_user_clusters >= 2
        and n_item_clusters >= 2
        and float(beta) != 0.0
    )

    user_cluster = None
    item_cluster = None
    A = None
    pi_by_g = None

    if use_clusters:
        G = int(n_user_clusters)
        H = int(n_item_clusters)

        user_cluster = rng.integers(0, G, size=U, dtype=np.int64)
        item_cluster = rng.integers(0, H, size=I, dtype=np.int64)

        if affinity is None:
            A = rng.normal(0.0, 1.0, size=(G, H))
            if G == H:
                A = A + 0.5 * np.eye(G)
        else:
            A = np.asarray(affinity, dtype=np.float64)
            if A.shape != (G, H):
                raise ValueError(f"affinity must have shape ({G},{H})")

        pi_by_g = np.empty((G, I), dtype=np.float64)
        ic = item_cluster
        for g in range(G):
            logits = float(beta) * A[g, ic]
            logits = logits - logits.max()
            pg = pi * np.exp(logits)
            pg = pg / pg.sum()
            pi_by_g[g] = pg

    keys = np.empty(0, dtype=np.int64)

    while keys.size < M:
        need = M - keys.size
        batch = int(np.ceil(need * oversample)) + 32

        u = rng.choice(U, size=batch, replace=True, p=pu).astype(np.int64)

        if not use_clusters:
            i = rng.choice(I, size=batch, replace=True, p=pi).astype(np.int64)
        else:
            i = np.empty(batch, dtype=np.int64)
            gu = user_cluster[u]
            for g in range(pi_by_g.shape[0]):
                idx = np.where(gu == g)[0]
                if idx.size:
                    i[idx] = rng.choice(I, size=idx.size, replace=True, p=pi_by_g[g]).astype(
                        np.int64
                    )

        k = u * np.int64(I) + i
        keys = np.unique(np.concatenate([keys, k]))

    if keys.size > M:
        keys = rng.choice(keys, size=M, replace=False)
    keys = keys[:M]

    if min_user_ratings is not None or min_item_ratings is not None:
        key_set = set(keys.tolist())

        u0 = (keys // I).astype(np.int64)
        i0 = (keys % I).astype(np.int64)
        user_deg = np.bincount(u0, minlength=U).astype(np.int64)
        item_deg = np.bincount(i0, minlength=I).astype(np.int64)

        def add_pair(uu: int, ii: int):
            kk = int(uu) * int(I) + int(ii)
            if kk in key_set:
                return False
            key_set.add(kk)
            user_deg[uu] += 1
            item_deg[ii] += 1
            return True

        if min_user_ratings is not None:
            for uu in np.where(user_deg < min_user_ratings)[0]:
                need_u = int(min_user_ratings - user_deg[uu])
                tries = 0
                while need_u > 0 and len(key_set) < U * I and tries < 10_000:
                    batch = int(np.ceil(need_u * oversample)) + 32

                    if not use_clusters:
                        p_items = pi
                    else:
                        g = int(user_cluster[uu])
                        p_items = pi_by_g[g]

                    cand_items = rng.choice(I, size=batch, replace=True, p=p_items).astype(
                        np.int64
                    )
                    cand_items = np.unique(cand_items)

                    added = 0
                    for ii in cand_items:
                        if add_pair(int(uu), int(ii)):
                            added += 1
                            need_u -= 1
                            if need_u == 0:
                                break
                    tries = tries + 1 if added == 0 else 0

        if min_item_ratings is not None:
            for ii in np.where(item_deg < min_item_ratings)[0]:
                need_i = int(min_item_ratings - item_deg[ii])
                tries = 0
                while need_i > 0 and len(key_set) < U * I and tries < 10_000:
                    batch = int(np.ceil(need_i * oversample)) + 32
                    cand_users = rng.choice(U, size=batch, replace=True, p=pu).astype(np.int64)
                    cand_users = np.unique(cand_users)

                    added = 0
                    for uu in cand_users:
                        if add_pair(int(uu), int(ii)):
                            added += 1
                            need_i -= 1
                            if need_i == 0:
                                break
                    tries = tries + 1 if added == 0 else 0

        keys = np.fromiter(key_set, dtype=np.int64)

    u = (keys // I).astype(int)
    i = (keys % I).astype(int)
    pairs = list(zip(u.tolist(), i.tolist()))

    if return_clusters:
        return pairs, user_cluster, item_cluster, A
    return pairs


def scores_to_ratings_fixed(scores: np.ndarray, tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    if tau.shape != (4,) or not np.all(np.diff(tau) > 0):
        raise ValueError("tau must be shape (4,) strictly increasing.")
    return (np.digitize(scores, tau, right=True) + 1).astype(np.int64)


def fill_ratings_clusters(
    U: int,
    I: int,
    pairs: list[tuple[int, int]],
    user_cluster: np.ndarray,
    item_cluster: np.ndarray,
    tau: np.ndarray,
    seed: int = 0,
    mu: float = 0.0,
    sigma_bu: float = 0.35,
    sigma_bi: float = 0.25,
    sigma_eps: float = 0.6,
    gamma: float = 0.6,
    A: np.ndarray | None = None,
    dtype=np.float32,
):
    rng = np.random.default_rng(seed)
    R = np.full((U, I), np.nan, dtype=dtype)
    if len(pairs) == 0:
        return R

    pairs_arr = np.asarray(pairs, dtype=np.int64)
    u = pairs_arr[:, 0]
    i = pairs_arr[:, 1]

    g = user_cluster[u]
    h = item_cluster[i]
    G = int(user_cluster.max() + 1)
    H = int(item_cluster.max() + 1)

    if A is None:
        A = rng.normal(0.0, 1.0, size=(G, H))
    else:
        A = np.asarray(A, dtype=np.float64)
        if A.shape != (G, H):
            raise ValueError(f"A must have shape ({G},{H})")

    A = (A - A.mean()) / (A.std() + 1e-12)

    b_u = rng.normal(0.0, sigma_bu, size=U)
    b_i = rng.normal(0.0, sigma_bi, size=I)
    eps = rng.normal(0.0, sigma_eps, size=len(pairs))

    scores = mu + b_u[u] + b_i[i] + gamma * A[g, h] + eps
    ratings = scores_to_ratings_fixed(scores, tau)

    R[u, i] = ratings.astype(dtype)
    return R


def pearson_centered(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0:
        return np.nan
    return float((x * y).sum() / denom)


def sample_user_user_pearson_centered(R, n_pairs=50_000, min_common=5, seed=0):
    if isinstance(R, pd.DataFrame):
        R = R.to_numpy()
    rng = np.random.default_rng(seed)
    U, I = R.shape
    deg = np.sum(~np.isnan(R), axis=1)
    eligible = np.where(deg >= min_common)[0]
    if len(eligible) < 2:
        return np.array([])

    sims = []
    attempts = 0
    max_attempts = n_pairs * 30

    while len(sims) < n_pairs and attempts < max_attempts:
        u, v = rng.choice(eligible, size=2, replace=False)
        common = (~np.isnan(R[u])) & (~np.isnan(R[v]))
        if common.sum() < min_common:
            attempts += 1
            continue

        s = pearson_centered(R[u, common], R[v, common])
        if np.isfinite(s):
            sims.append(s)

        attempts += 1

    return np.array(sims)


def summarize(arr):
    if arr.size == 0:
        return {"n": 0}
    qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "pos_frac": float((arr > 0).mean()),
    }



