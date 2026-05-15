from surprise import Dataset, Reader, SVDpp, SVD, BaselineOnly, KNNBasic,NormalPredictor
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .cluster_simulation import ratings_for_new_pairs


class PopularityModel:
    """Non-personalized model: recommends items with the highest mean rating."""
    def fit(self, trainset):
        return self


def get_models():
    return {
        "BaselineOnly": BaselineOnly(verbose=False),
        
        "CF User-Pearson": KNNBasic(
            sim_options={"name": "pearson_baseline", "user_based": True},
            verbose=False,
        ),
        "SVD": SVD(random_state=42, verbose=False),
        "SVDpp": SVDpp(random_state=42, verbose=False),
        "Popularity": PopularityModel(),
        "Random": NormalPredictor(),
    }


def _as_inner_item_array(items):
    if not items:
        return np.array([], dtype=int)
    return np.asarray(sorted(set(int(i) for i in items)), dtype=int)


def _topk_from_scores(scores, k):
    finite_idx = np.flatnonzero(np.isfinite(scores))
    if finite_idx.size == 0:
        return np.array([], dtype=int)
    k_eff = min(k, finite_idx.size)
    if k_eff == finite_idx.size:
        idx = finite_idx
    else:
        idx = finite_idx[np.argpartition(scores[finite_idx], -k_eff)[-k_eff:]]
    return idx[np.argsort(scores[idx])[::-1]]


def topk_baseline(
    algo,
    trainset,
    k=5,
    exclude_seen=True,
    blocked_items_by_user=None,
    return_scores=False,
):
    topk = {}
    topk_scores = {}
    mu = trainset.global_mean
    bi = algo.bi  # (n_items,)
    for u in range(trainset.n_users):
        seen = {i for (i, _) in trainset.ur[u]}
        scores = mu + algo.bu[u] + bi.copy()
        if exclude_seen and seen:
            scores[list(seen)] = -np.inf
        blocked = _as_inner_item_array((blocked_items_by_user or {}).get(u))
        if blocked.size:
            scores[blocked] = -np.inf
        idx = _topk_from_scores(scores, k)
        topk[u] = idx
        topk_scores[u] = scores[idx]
    if return_scores:
        return topk, topk_scores
    return topk

def topk_svd(
    algo,
    trainset,
    k=5,
    exclude_seen=True,
    blocked_items_by_user=None,
    return_scores=False,
):
    topk = {}
    topk_scores = {}
    mu = trainset.global_mean
    bi = algo.bi              # (n_items,)
    qi = algo.qi              # (n_items, n_factors)

    for u in range(trainset.n_users):
        seen = {i for (i, _) in trainset.ur[u]}
        pu = algo.pu[u]       # (n_factors,)

        scores = mu + algo.bu[u] + bi + qi @ pu  # (n_items,)
        if exclude_seen and seen:
            scores[list(seen)] = -np.inf
        blocked = _as_inner_item_array((blocked_items_by_user or {}).get(u))
        if blocked.size:
            scores[blocked] = -np.inf
        idx = _topk_from_scores(scores, k)
        topk[u] = idx
        topk_scores[u] = scores[idx]
    if return_scores:
        return topk, topk_scores
    return topk

def topk_svdpp(
    algo,
    trainset,
    k=5,
    exclude_seen=True,
    blocked_items_by_user=None,
    return_scores=False,
):
    topk = {}
    topk_scores = {}
    mu = trainset.global_mean
    bi = algo.bi
    qi = algo.qi
    yj = algo.yj  # (n_items, n_factors)  (en Surprise suele llamarse yj)

    for u in range(trainset.n_users):
        Iu = [i for (i, _) in trainset.ur[u]]
        seen = set(Iu)

        if len(Iu) > 0:
            z = algo.pu[u] + yj[Iu].sum(axis=0) / np.sqrt(len(Iu))
        else:
            z = algo.pu[u]

        scores = mu + algo.bu[u] + bi + qi @ z
        if exclude_seen and seen:
            scores[list(seen)] = -np.inf
        blocked = _as_inner_item_array((blocked_items_by_user or {}).get(u))
        if blocked.size:
            scores[blocked] = -np.inf
        idx = _topk_from_scores(scores, k)
        topk[u] = idx
        topk_scores[u] = scores[idx]
    if return_scores:
        return topk, topk_scores
    return topk


def topk_knn_vectorized(
    algo,
    trainset,
    k=5,
    exclude_seen=True,
    blocked_items_by_user=None,
    return_scores=False,
):
    """
    User-based CF with Pearson-baseline similarity.
    Scores item i for user u as the weighted average of all neighbors who rated i,
    weighted by their Pearson similarity to u.
    """
    U, I = trainset.n_users, trainset.n_items

    # build sparse ratings matrix R (U x I)
    rows, cols, data = [], [], []
    for u in range(U):
        for (i, r) in trainset.ur[u]:
            rows.append(u); cols.append(i); data.append(r)
    R = csr_matrix((data, (rows, cols)), shape=(U, I), dtype=np.float32).toarray()

    R_binary = (R != 0).astype(np.float32)
    sim = algo.sim  # (U, U) — full similarity matrix, no sparsification

    mu = trainset.global_mean
    bu = algo.bu  # (U,) — already computed by pearson_baseline
    bi = algo.bi  # (I,) — already computed by pearson_baseline
    B = mu + bu[:, None] + bi[None, :]       # (U, I) baseline matrix
    R_centered = np.where(R != 0, R - B, 0)  # subtract bias only where rated

    numerator   = sim @ R_centered           # (U, I)
    denominator = np.abs(sim) @ R_binary     # (U, I)
    denominator[denominator == 0] = 1.0
    scores_matrix = B + numerator / denominator  # (U, I)

    topk = {}
    topk_scores = {}
    for u in range(U):
        scores = scores_matrix[u].copy()
        if exclude_seen:
            seen = [i for (i, _) in trainset.ur[u]]
            if seen:
                scores[seen] = -np.inf
        blocked = _as_inner_item_array((blocked_items_by_user or {}).get(u))
        if blocked.size:
            scores[blocked] = -np.inf
        idx = _topk_from_scores(scores, k)
        topk[u] = idx
        topk_scores[u] = scores[idx]

    if return_scores:
        return topk, topk_scores
    return topk

def topk_popularity(
    algo,
    trainset,
    k=5,
    exclude_seen=True,
    blocked_items_by_user=None,
    return_scores=False,
):
    # Compute item mean ratings once
    item_sum = np.zeros(trainset.n_items, dtype=np.float64)
    item_count = np.zeros(trainset.n_items, dtype=np.int32)
    for u in range(trainset.n_users):
        for (i, r) in trainset.ur[u]:
            item_sum[i] += r
            item_count[i] += 1
    item_mean = np.where(item_count > 0, item_sum / item_count, 0.0)

    # Global ranking (highest mean first) — computed once, reused for all users
    global_ranking = np.argsort(item_mean)[::-1]

    topk = {}
    topk_scores = {}
    for u in range(trainset.n_users):
        seen = {i for (i, _) in trainset.ur[u]} if exclude_seen else set()
        blocked = set(_as_inner_item_array((blocked_items_by_user or {}).get(u)).tolist())
        excluded = seen | blocked

        chosen = []
        for i in global_ranking:
            if i not in excluded:
                chosen.append(i)
            if len(chosen) == k:
                break

        chosen = np.array(chosen, dtype=int)
        topk[u] = chosen
        topk_scores[u] = item_mean[chosen]

    if return_scores:
        return topk, topk_scores
    return topk


def topk_normalpredictor_uniform(
    algo,
    trainset,
    k=5,
    seed=0,
    exclude_seen=True,
    blocked_items_by_user=None,
    return_scores=False,
):
    """
    Devuelve top-k (inner item ids) por usuario para NormalPredictor.
    Como NormalPredictor no depende de u ni i, el top-k es equivalente a
    samplear k ítems no vistos uniformemente (sin replacement).
    """
    rng = np.random.default_rng(seed)
    n_items = trainset.n_items
    all_items = np.arange(n_items)

    topk = {}
    topk_scores = {}
    for u in range(trainset.n_users):
        blocked = _as_inner_item_array((blocked_items_by_user or {}).get(u))
        if exclude_seen:
            seen = {i for (i, _) in trainset.ur[u]}
            excluded = np.union1d(np.fromiter(seen, dtype=int), blocked)
            candidates = np.setdiff1d(all_items, excluded, assume_unique=False)
        elif blocked.size:
            candidates = np.setdiff1d(all_items, blocked, assume_unique=False)
        else:
            candidates = all_items

        if candidates.size <= k:
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=k, replace=False)

        topk[u] = chosen
        if return_scores:
            raw_uid = trainset.to_raw_uid(u)
            topk_scores[u] = np.asarray(
                [
                    float(algo.predict(raw_uid, trainset.to_raw_iid(int(i_inner))).est)
                    for i_inner in chosen
                ],
                dtype=float,
            )

    if return_scores:
        return topk, topk_scores
    return topk


def topk_to_dataframe(trainset, topk, topk_scores):
    rows = []
    for u_inner, i_inners in topk.items():
        u_raw = int(trainset.to_raw_uid(u_inner))
        scores = np.asarray(topk_scores.get(u_inner, []), dtype=float)
        for rank, i_inner in enumerate(i_inners, start=1):
            i_raw = int(trainset.to_raw_iid(int(i_inner)))
            rows.append(
                {
                    "UserID": u_raw,
                    "MovieID": i_raw,
                    "PredictedRating": float(scores[rank - 1]),
                    "Rank": rank,
                }
            )
    if not rows:
        return pd.DataFrame(
            {
                "UserID": pd.Series(dtype=int),
                "MovieID": pd.Series(dtype=int),
                "PredictedRating": pd.Series(dtype=float),
                "Rank": pd.Series(dtype=int),
            }
        )
    return pd.DataFrame(rows, columns=["UserID", "MovieID", "PredictedRating", "Rank"])





def run_topk_loop_with_state(
    ratings_long,
    reader,
    U,
    I,
    uc,
    ic,
    state,
    A,
    n_runs=5,
    k=4,
    cooldown_runs=None,
    include_base_in_cooldown=True,
):
    """
    1. Trains each algorityhm
    2. Each algorith generate top k predictions
    3. A synthetic value is assigned to that prediction
    4. Does this n_runs

    If cooldown_runs is a positive integer, a recommended (UserID, MovieID)
    pair can only be shown again after cooldown_runs iterations.

    Returns ratings_by_model, topk_by_model, data_train_by_model, rmse_df.
    rmse_df has one row per (Model, Run) with the RMSE between predicted and actual ratings.
    """

    ratings_by_model = {}
    topk_by_model = {}
    data_train_by_model = {}
    rmse_rows = []

    for name, algo in get_models().items():
        ratings_long_current = ratings_long.copy()
        increments = [ratings_long.copy()]
        topk_runs = []
        use_cooldown = isinstance(cooldown_runs, int) and cooldown_runs > 0
        exclude_seen = not use_cooldown
        last_shown_run = {}

        if use_cooldown and include_base_in_cooldown:
            base_pairs = ratings_long[["UserID", "MovieID"]].drop_duplicates()
            for row in base_pairs.itertuples(index=False):
                last_shown_run[(int(row.UserID), int(row.MovieID))] = 0

        for run in range(n_runs):
            data_current = Dataset.load_from_df(
                ratings_long_current[["UserID", "MovieID", "Rating"]],
                reader,
            )
            trainset = data_current.build_full_trainset()

            algo.fit(trainset)
            blocked_items_by_user = None
            if use_cooldown:
                blocked_items_by_user = {}
                raw_item_to_inner = trainset._raw2inner_id_items
                recent_items_by_user = {}
                for (u_raw, i_raw), last_run in last_shown_run.items():
                    if (run - last_run) < cooldown_runs:
                        recent_items_by_user.setdefault(u_raw, []).append(i_raw)
                for u_inner in range(trainset.n_users):
                    u_raw = int(trainset.to_raw_uid(u_inner))
                    recent_raw_items = recent_items_by_user.get(u_raw, [])
                    blocked = [
                        int(raw_item_to_inner[i_raw])
                        for i_raw in recent_raw_items
                        if i_raw in raw_item_to_inner
                    ]
                    if blocked:
                        blocked_items_by_user[u_inner] = blocked

            if name == "BaselineOnly":
                topk, topk_scores = topk_baseline(
                    algo,
                    trainset,
                    k=k,
                    exclude_seen=exclude_seen,
                    blocked_items_by_user=blocked_items_by_user,
                    return_scores=True,
                )
            elif name == "SVD":
                topk, topk_scores = topk_svd(
                    algo,
                    trainset,
                    k=k,
                    exclude_seen=exclude_seen,
                    blocked_items_by_user=blocked_items_by_user,
                    return_scores=True,
                )
            elif name == "SVDpp":
                topk, topk_scores = topk_svdpp(
                    algo,
                    trainset,
                    k=k,
                    exclude_seen=exclude_seen,
                    blocked_items_by_user=blocked_items_by_user,
                    return_scores=True,
                )
            elif name == "CF User-Pearson":
                topk, topk_scores = topk_knn_vectorized(
                    algo,
                    trainset,
                    k=k,
                    exclude_seen=exclude_seen,
                    blocked_items_by_user=blocked_items_by_user,
                    return_scores=True,
                )
            elif name == "Popularity":
                topk, topk_scores = topk_popularity(
                    algo,
                    trainset,
                    k=k,
                    exclude_seen=exclude_seen,
                    blocked_items_by_user=blocked_items_by_user,
                    return_scores=True,
                )
            elif name == "Random":
                topk, topk_scores = topk_normalpredictor_uniform(
                    algo,
                    trainset,
                    k=k,
                    seed=42 + run,
                    exclude_seen=exclude_seen,
                    blocked_items_by_user=blocked_items_by_user,
                    return_scores=True,
                )
            else:
                raise ValueError(f"Unknown model: {name}")

            topk_runs.append(topk)
            prediction_df = topk_to_dataframe(trainset, topk, topk_scores)

            rows = []
            for u_inner, i_inners in topk.items():
                u_raw = int(trainset.to_raw_uid(u_inner))
                for i_inner in i_inners:
                    i_raw = int(trainset.to_raw_iid(int(i_inner)))
                    rows.append((u_raw, i_raw))

            if use_cooldown:
                for pair in rows:
                    last_shown_run[pair] = run

            topk_df = prediction_df[["UserID", "MovieID"]].copy()
            if rows:
                R_topk = ratings_for_new_pairs(
                    U=U,
                    I=I,
                    pairs=rows,
                    user_cluster=uc,
                    item_cluster=ic,
                    state=state,
                    A=A,
                )

                u_idx = topk_df["UserID"].to_numpy(dtype=int)
                i_idx = topk_df["MovieID"].to_numpy(dtype=int)
                topk_df["Rating"] = R_topk[u_idx, i_idx]

                merged = prediction_df[["UserID", "MovieID", "PredictedRating"]].merge(
                    topk_df[["UserID", "MovieID", "Rating"]], on=["UserID", "MovieID"]
                )
                if len(merged) > 0:
                    rmse = np.sqrt(((merged["PredictedRating"] - merged["Rating"]) ** 2).mean())
                    rmse_rows.append({"Model": name, "Run": run, "RMSE": rmse})
            else:
                topk_df["Rating"] = pd.Series(dtype=float)

            ratings_long_current = pd.concat([ratings_long_current, topk_df], ignore_index=True)
            increments.append(topk_df.copy())

        topk_by_model[name] = topk_runs
        data_train_by_model[name] = []
        # keep only deltas until all models finish
        ratings_by_model[name] = increments

    # reconstruct cumulative snapshots from deltas now that all models are done
    for name, increments in ratings_by_model.items():
        runs = []
        cumulative = increments[0]
        for delta in increments[1:]:
            cumulative = pd.concat([cumulative, delta], ignore_index=True)
            runs.append(cumulative)
        ratings_by_model[name] = runs

    rmse_df = pd.DataFrame(rmse_rows)
    return ratings_by_model, topk_by_model, data_train_by_model, rmse_df



def popularity_analysis(ratings_by_model, plot_lorenz=True, gini_only=False):
    """
    Makes all popularity analysis with plots included.
    gini_only=True: single side-by-side figure with Gini + Lorenz, colours matched.
    """
    def gini(x):
        x = np.asarray(x, dtype=float)
        x = x[x >= 0]
        if x.size == 0:
            return np.nan
        x = np.sort(x)
        n = x.size
        cumx = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    def compute_popularity_metrics_by_model(ratings_by_model):
        item_col = "MovieID"
        all_items = pd.Index([], name=item_col)
        for name, runs in ratings_by_model.items():
            if not runs:
                continue
            if item_col not in runs[0].columns:
                raise KeyError(f"'{item_col}' not found in model '{name}' run 0 dataframe.")
            all_items = all_items.union(pd.Index(runs[0][item_col].unique(), name=item_col))

        rows = []
        if len(all_items) == 0:
            return pd.DataFrame(columns=[
                "model", "run", "gini_cnt", "delta_gini_cnt",
                "top10_share", "top100_share", "total_interactions"
            ])

        first_model = next(iter(ratings_by_model))
        base_df = ratings_by_model[first_model][0]
        base_cnt = (
            base_df.groupby(item_col)
                   .size()
                   .reindex(all_items, fill_value=0)
                   .sort_values(ascending=False)
        )
        base_total = float(base_cnt.sum())
        rows.append({
            "model": "Base",
            "run": -1,
            "gini_cnt": float(gini(base_cnt.values)),
            "delta_gini_cnt": np.nan,
            "top10_share": base_cnt.head(10).sum() / base_total if base_total else np.nan,
            "top100_share": base_cnt.head(100).sum() / base_total if base_total else np.nan,
        })

        for name, runs in ratings_by_model.items():
            if not runs:
                continue
            gini0 = None
            for run_idx, df in enumerate(runs):
                if item_col not in df.columns:
                    raise KeyError(f"'{item_col}' not found in model '{name}' run {run_idx} dataframe.")
                item_cnt = (
                    df.groupby(item_col)
                      .size()
                      .reindex(all_items, fill_value=0)
                      .sort_values(ascending=False)
                )
                total = float(item_cnt.sum())
                top10_share = item_cnt.head(20).sum() / total if total else np.nan
                top100_share = item_cnt.head(100).sum() / total if total else np.nan
                g = float(gini(item_cnt.values))
                if gini0 is None:
                    gini0 = g
                rows.append({
                    "model": name,
                    "run": run_idx,
                    "gini_cnt": g,
                    "delta_gini_cnt": g - gini0,
                    "top10_share": float(top10_share) if total else np.nan,
                    "top100_share": float(top100_share) if total else np.nan,
                })

        return (
            pd.DataFrame(rows)
              .sort_values(["model", "run"])
              .reset_index(drop=True)
        )

    def lorenz_curve(x):
        x = np.asarray(x, dtype=float)
        x = x[x >= 0]
        if x.size == 0 or x.sum() == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])
        x = np.sort(x)
        cumx = np.cumsum(x)
        lorenz_y = np.insert(cumx / cumx[-1], 0, 0.0)
        lorenz_x = np.linspace(0.0, 1.0, lorenz_y.size)
        return lorenz_x, lorenz_y

    popularity_metrics = compute_popularity_metrics_by_model(ratings_by_model)

    # shared colour map — sorted model names → consistent colours across both plots
    model_names_sorted = sorted(ratings_by_model.keys())
    colors = {name: col for name, col in zip(model_names_sorted, plt.cm.tab10.colors)}

    if gini_only:
        # --- single figure: Gini (left) + Lorenz (right) ---
        fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(14, 5))

        for method, sub in popularity_metrics[popularity_metrics['model'] != 'Base'].groupby('model'):
            sub = sub.sort_values('run')
            ax_g.plot(sub['run'], sub['gini_cnt'], marker='o', label=method, color=colors[method])
        ax_g.set_title('Coeficiente de Gini por iteración')
        ax_g.set_xlabel('iteración')
        ax_g.set_ylabel('Gini')
        ax_g.grid(True, alpha=0.3)

        first_model = next(iter(ratings_by_model))
        base_cnt = ratings_by_model[first_model][0].groupby('MovieID').size().values
        lx, ly = lorenz_curve(base_cnt)
        ax_l.plot(lx, ly, label='Base', color='gray', linestyle='--', linewidth=2)
        for name, runs in ratings_by_model.items():
            if not runs:
                continue
            lx, ly = lorenz_curve(runs[-1].groupby('MovieID').size().values)
            ax_l.plot(lx, ly, label=name, color=colors[name])
        ax_l.plot([0, 1], [0, 1], linestyle=':', color='black', alpha=0.4)
        ax_l.set_title('Curva de Lorenz en la última iteración')
        ax_l.set_xlabel('Proporción de ítems')
        ax_l.set_ylabel('Proporción de interacciones')
        ax_l.set_aspect('equal', adjustable='box')
        ax_l.grid(True, alpha=0.3)

        handles, labels = ax_g.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(.81, 0.5))
        plt.tight_layout(rect=[0, 0, 0.87, 1])
        plt.show()

    else:
        # --- original three-metric plots ---
        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for method, sub in popularity_metrics.groupby('model'):
            c = colors.get(method, None)
            axes[0].plot(sub['run'], sub['gini_cnt'], marker='o', label=method, color=c)
            axes[1].plot(sub['run'], sub['top10_share'], marker='o', label=method, color=c)
            axes[2].plot(sub['run'], sub['top100_share'], marker='o', label=method, color=c)

        axes[0].set_title('Gini')
        axes[0].set_xlabel('run')
        axes[0].set_ylabel('value')
        axes[1].set_title('Top-20 share')
        axes[1].set_xlabel('run')
        axes[2].set_title('Top-100 share')
        axes[2].set_xlabel('run')

        for ax in axes:
            ax.axvline(-1, color='gray', linestyle='--', alpha=0.5)
            ax.legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(.85, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

        # --- Lorenz curve ---
        if plot_lorenz:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            first_model = next(iter(ratings_by_model))
            base_cnt = ratings_by_model[first_model][0].groupby('MovieID').size().values
            lx, ly = lorenz_curve(base_cnt)
            ax.plot(lx, ly, label='Base', color='gray', linestyle='--', linewidth=2)

            for name, runs in ratings_by_model.items():
                if not runs:
                    continue
                lx, ly = lorenz_curve(runs[-1].groupby('MovieID').size().values)
                ax.plot(lx, ly, label=f"{name} (final)", color=colors[name])

            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)
            ax.set_title('Lorenz Curve (Item Interaction Counts)')
            ax.set_xlabel('Cumulative share of items')
            ax.set_ylabel('Cumulative share of interactions')
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.show()

    return popularity_metrics
