from surprise import Dataset, Reader, SVDpp, SVD, BaselineOnly, KNNBasic,NormalPredictor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cluster_simulation import ratings_for_new_pairs

def get_models():
    return {
        "BaselineOnly": BaselineOnly(verbose=False),
        #"KNNBasic": KNNBasic(
          #  sim_options={"name": "pearson_baseline", "user_based": True},
           # verbose=False,
       # ),
        "SVD": SVD(random_state=42, verbose=False),
       # "SVDpp": SVDpp(random_state=42, verbose=False),
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


def topk_knn_candidates(
    algo,
    trainset,
    k=5,
    max_cand=2000,
    exclude_seen=True,
    blocked_items_by_user=None,
    return_scores=False,
):
    topk = {}
    topk_scores = {}
    user_based = algo.sim_options.get("user_based", True)

    for u in range(trainset.n_users):
        seen = {i for (i, _) in trainset.ur[u]}
        blocked = set((blocked_items_by_user or {}).get(u, []))
        cand = set()

        if user_based:
            neigh = algo.get_neighbors(u, k=algo.k)  # vecinos usuarios (inner ids)
            for v in neigh:
                for (i, _) in trainset.ur[v]:
                    if ((not exclude_seen) or (i not in seen)) and (i not in blocked):
                        cand.add(i)
                        if len(cand) >= max_cand:
                            break
                if len(cand) >= max_cand:
                    break
        else:
            # item-based: candidatos = vecinos de los ítems que el usuario ha visto
            for (j, _) in trainset.ur[u]:
                for i in algo.get_neighbors(j, k=algo.k):
                    if ((not exclude_seen) or (i not in seen)) and (i not in blocked):
                        cand.add(i)
                        if len(cand) >= max_cand:
                            break
                if len(cand) >= max_cand:
                    break

        if not cand:
            topk[u] = np.array([], dtype=int)
            topk_scores[u] = np.array([], dtype=float)
            continue

        cand = np.fromiter(cand, dtype=int)
        scores = np.empty(len(cand), dtype=float)
        for t, i in enumerate(cand):
            est = algo.estimate(u, i)  # usa inner ids
            if isinstance(est, tuple):
                est = est[0]
            scores[t] = est

        k_eff = min(k, len(scores))
        if k_eff == 0:
            topk[u] = np.array([], dtype=int)
            topk_scores[u] = np.array([], dtype=float)
        else:
            idx = np.argpartition(scores, -k_eff)[-k_eff:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            topk[u] = cand[idx]
            topk_scores[u] = scores[idx]
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
    return_predictions=False,
):
    """
    1. Trains each algorityhm
    2. Each algorith generate top k predictions
    3. A synthetic value is assigned to that prediction
    4. Does this n_runs

    If cooldown_runs is a positive integer, a recommended (UserID, MovieID)
    pair can only be shown again after cooldown_runs iterations.

    If return_predictions is True, also returns one dataframe per model/run
    with the selected pairs and their predicted scores at recommendation time.
    """

    ratings_by_model = {}
    topk_by_model = {}
    data_train_by_model = {}
    predictions_by_model = {}

    for name, algo in get_models().items():
        ratings_long_current = ratings_long.copy()
        ratings_long_runs = []
        data_train_runs = []
        topk_runs = []
        prediction_runs = []
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
            elif name == "KNNBasic":
                topk, topk_scores = topk_knn_candidates(
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
            prediction_runs.append(prediction_df)

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
            else:
                topk_df["Rating"] = pd.Series(dtype=float)

            ratings_long_current = pd.concat([ratings_long_current, topk_df], ignore_index=True)
            ratings_long_runs.append(ratings_long_current)
            data_train_runs.append(data_current)

        ratings_by_model[name] = ratings_long_runs
        topk_by_model[name] = topk_runs
        data_train_by_model[name] = data_train_runs
        predictions_by_model[name] = prediction_runs

    if return_predictions:
        return ratings_by_model, topk_by_model, data_train_by_model, predictions_by_model
    return ratings_by_model, topk_by_model, data_train_by_model



def popularity_analysis(ratings_by_model, plot_lorenz=True):
    """
    Makes all popularity analysis with plots included
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

    # --- metrics plots ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for method, sub in popularity_metrics.groupby('model'):
        axes[0].plot(sub['run'], sub['gini_cnt'], marker='o', label=method)
        axes[1].plot(sub['run'], sub['top10_share'], marker='o', label=method)
        axes[2].plot(sub['run'], sub['top100_share'], marker='o', label=method)

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
        base_df = ratings_by_model[first_model][0]
        base_cnt = base_df.groupby('MovieID').size().values
        lx, ly = lorenz_curve(base_cnt)
        ax.plot(lx, ly, label='Base', linewidth=2)

        for name, runs in ratings_by_model.items():
            if not runs:
                continue
            last_df = runs[-1]
            lx, ly = lorenz_curve(last_df.groupby('MovieID').size().values)
            ax.plot(lx, ly, label=f"{name} (final)")

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)
        ax.set_title('Lorenz Curve (Item Interaction Counts)')
        ax.set_xlabel('Cumulative share of items')
        ax.set_ylabel('Cumulative share of interactions')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    return popularity_metrics
