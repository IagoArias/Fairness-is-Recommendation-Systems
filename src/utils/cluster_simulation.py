import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


def default_values(
    U=None,
    I=None,
    density=None,
    alpha_user=None,
    alpha_item=None,
    seed=None,
    min_user_ratings=None,
    min_item_ratings=None,
    n_user_clusters=None,
    n_item_clusters=None,
    beta=None,
    return_clusters=None,
    tau=None,
    mu=None,
    sigma_bu=None,
    sigma_bi=None,
    sigma_eps=None,
    gamma=None,
):
    '''
    Returns default values for the parameters of the simulation functions. 
    If any parameter is provided, it will override the default value.     
    '''

    U = 1000 if U is None else U
    I = 2000 if I is None else I
    density = 0.06 if density is None else density
    alpha_user = 0.9 if alpha_user is None else alpha_user
    alpha_item = 0.9 if alpha_item is None else alpha_item
    seed = 42 if seed is None else seed
    min_user_ratings = 20 if min_user_ratings is None else min_user_ratings
    min_item_ratings = 5 if min_item_ratings is None else min_item_ratings
    n_user_clusters = 10 if n_user_clusters is None else n_user_clusters
    n_item_clusters = 10 if n_item_clusters is None else n_item_clusters
    beta = 1.0 if beta is None else beta
    return_clusters = True if return_clusters is None else return_clusters
    tau = np.array([-1.2, -0.4, 0.4, 1.2]) if tau is None else tau
    mu = 0.4 if mu is None else mu
    sigma_bu = 0.5 if sigma_bu is None else sigma_bu
    sigma_bi = 0.5 if sigma_bi is None else sigma_bi
    sigma_eps = 0.4 if sigma_eps is None else sigma_eps
    gamma = 0.4 if gamma is None else gamma
    return (
        U,
        I,
        density,
        alpha_user,
        alpha_item,
        seed,
        min_user_ratings,
        min_item_ratings,
        n_user_clusters,
        n_item_clusters,
        beta,
        return_clusters,
        tau,
        mu,
        sigma_bu,
        sigma_bi,
        sigma_eps,
        gamma,
    )

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
            logits = logits - logits.max() # preguntar para qué es esto
            pg = pi * np.exp(logits)
            pg = pg / pg.sum()
            pi_by_g[g] = pg

    # Step 1: sample user degrees independently from item popularity.
    user_deg_target = rng.multinomial(M, pu).astype(np.int64)

    # Enforce the structural cap: each user can connect to at most I unique items.
    overflow = 0
    over_mask = user_deg_target > I
    if np.any(over_mask):
        overflow = int((user_deg_target[over_mask] - I).sum())
        user_deg_target = np.minimum(user_deg_target, I)

    # Redistribute overflow to users with free capacity, preserving total edges when possible.
    if overflow > 0:
        free_cap = (I - user_deg_target).astype(np.int64)
        free_total = int(free_cap.sum())
        if free_total > 0:
            q = np.where(free_cap > 0, pu * free_cap, 0.0)
            q_sum = q.sum()
            if q_sum > 0:
                add = rng.multinomial(min(overflow, free_total), q / q_sum).astype(np.int64)
                user_deg_target = np.minimum(I, user_deg_target + add)

    # Step 2: for each user, draw exactly d_u unique items with weighted probabilities.
    pair_u = []
    pair_i = []
    for uu in range(U):
        k = int(user_deg_target[uu])
        if k <= 0:
            continue

        if not use_clusters:
            p_items = pi
        else:
            g = int(user_cluster[uu])
            p_items = pi_by_g[g]

        if k >= I:
            chosen_items = np.arange(I, dtype=np.int64)
        else:
            chosen_items = rng.choice(I, size=k, replace=False, p=p_items).astype(np.int64)

        pair_u.append(np.full(chosen_items.size, uu, dtype=np.int64))
        pair_i.append(chosen_items)

    if pair_u:
        u = np.concatenate(pair_u)
        i = np.concatenate(pair_i)
        keys = u * np.int64(I) + i
    else:
        keys = np.empty(0, dtype=np.int64)

    if min_user_ratings is not None or min_item_ratings is not None:
        key_set = set(keys.tolist())

        u0 = (keys // I).astype(np.int64)
        i0 = (keys % I).astype(np.int64)
        user_deg = np.bincount(u0, minlength=U).astype(np.int64)
        item_deg = np.bincount(i0, minlength=I).astype(np.int64)
        all_items = np.arange(I, dtype=np.int64)
        all_users = np.arange(U, dtype=np.int64)
        user_seen_items = [set() for _ in range(U)]
        item_seen_users = [set() for _ in range(I)]
        for uu0, ii0 in zip(u0.tolist(), i0.tolist()):
            user_seen_items[int(uu0)].add(int(ii0))
            item_seen_users[int(ii0)].add(int(uu0))

        def add_pair(uu: int, ii: int):
            kk = int(uu) * int(I) + int(ii)
            if kk in key_set:
                return False
            key_set.add(kk)
            user_deg[uu] += 1
            item_deg[ii] += 1
            user_seen_items[uu].add(ii)
            item_seen_users[ii].add(uu)
            return True

        if min_user_ratings is not None:
            for uu in np.where(user_deg < min_user_ratings)[0]:
                need_u = int(min_user_ratings - user_deg[uu])
                if need_u <= 0:
                    continue
                seen = user_seen_items[int(uu)]
                if len(seen) >= I:
                    continue

                if not use_clusters:
                    p_items = pi
                else:
                    g = int(user_cluster[uu])
                    p_items = pi_by_g[g]

                unseen_items = np.setdiff1d(
                    all_items, np.fromiter(seen, dtype=np.int64), assume_unique=False
                )
                if unseen_items.size == 0:
                    continue
                k = int(min(need_u, unseen_items.size))
                p_unseen = np.asarray(p_items[unseen_items], dtype=np.float64)
                p_sum = float(p_unseen.sum())
                if p_sum > 0.0 and np.isfinite(p_sum):
                    p_unseen /= p_sum
                    chosen_items = rng.choice(unseen_items, size=k, replace=False, p=p_unseen)
                else:
                    chosen_items = rng.choice(unseen_items, size=k, replace=False)

                for ii in chosen_items:
                    add_pair(int(uu), int(ii))

        if min_item_ratings is not None:
            for ii in np.where(item_deg < min_item_ratings)[0]:
                need_i = int(min_item_ratings - item_deg[ii])
                if need_i <= 0:
                    continue
                seen = item_seen_users[int(ii)]
                if len(seen) >= U:
                    continue

                unseen_users = np.setdiff1d(
                    all_users, np.fromiter(seen, dtype=np.int64), assume_unique=False
                )
                if unseen_users.size == 0:
                    continue
                k = int(min(need_i, unseen_users.size))
                p_unseen = np.asarray(pu[unseen_users], dtype=np.float64)
                p_sum = float(p_unseen.sum())
                if p_sum > 0.0 and np.isfinite(p_sum):
                    p_unseen /= p_sum
                    chosen_users = rng.choice(unseen_users, size=k, replace=False, p=p_unseen)
                else:
                    chosen_users = rng.choice(unseen_users, size=k, replace=False)

                for uu in chosen_users:
                    add_pair(int(uu), int(ii))

        keys = np.fromiter(key_set, dtype=np.int64)

    u = (keys // I).astype(int)
    i = (keys % I).astype(int)
    pairs = list(zip(u.tolist(), i.tolist()))

    if return_clusters:
        return pairs, user_cluster, item_cluster, A, pi
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
    return_state: bool = False,
):
    rng = np.random.default_rng(seed)
    R = np.full((U, I), np.nan, dtype=dtype)
    if len(pairs) == 0:
        if return_state:
            state = {
                "A": None,
                "b_u": None,
                "b_i": None,
                "mu": float(mu),
                "sigma_bu": float(sigma_bu),
                "sigma_bi": float(sigma_bi),
                "sigma_eps": float(sigma_eps),
                "gamma": float(gamma),
                "tau": np.asarray(tau, dtype=np.float64),
            }
            return R, state
        return R

    pairs_arr = np.asarray(pairs, dtype=np.int64)
    u = pairs_arr[:, 0]
    i = pairs_arr[:, 1]

    use_clusters = user_cluster is not None and item_cluster is not None
    if use_clusters:
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
    else:
        A = None

    b_u = rng.normal(0.0, sigma_bu, size=U)
    b_i = rng.normal(0.0, sigma_bi, size=I)
    eps = rng.normal(0.0, sigma_eps, size=len(pairs))

    if use_clusters:
        scores = mu + b_u[u] + b_i[i] + gamma * A[g, h] + eps
    else:
        scores = mu + b_u[u] + b_i[i] + eps
    ratings = scores_to_ratings_fixed(scores, tau)

    R[u, i] = ratings.astype(dtype)
    if return_state:
        state = {
            "A": A,
            "b_u": b_u,
            "b_i": b_i,
            "mu": float(mu),
            "sigma_bu": float(sigma_bu),
            "sigma_bi": float(sigma_bi),
            "sigma_eps": float(sigma_eps),
            "gamma": float(gamma),
            "tau": np.asarray(tau, dtype=np.float64),
        }
        return R, state
    return R


def ratings_for_new_pairs(
    U: int,
    I: int,
    pairs: list[tuple[int, int]],
    user_cluster: np.ndarray,
    item_cluster: np.ndarray,
    tau: np.ndarray | None = None,
    seed: int = 0,
    mu: float = 0.0,
    sigma_bu: float = 0.35,
    sigma_bi: float = 0.25,
    sigma_eps: float = 0.6,
    gamma: float = 0.6,
    A: np.ndarray | None = None,
    b_u: np.ndarray | None = None,
    b_i: np.ndarray | None = None,
    state: dict | None = None,
    dtype=np.float32,
    return_state: bool = False,
):
    rng = np.random.default_rng(seed)
    R = np.full((U, I), np.nan, dtype=dtype)
    if len(pairs) == 0:
        if return_state:
            return R, state
        return R

    pairs_arr = np.asarray(pairs, dtype=np.int64)
    u = pairs_arr[:, 0]
    i = pairs_arr[:, 1]
    use_clusters = user_cluster is not None and item_cluster is not None

    if state is not None:
        A = state.get("A", A)
        b_u = state.get("b_u", b_u)
        b_i = state.get("b_i", b_i)
        mu = state.get("mu", mu)
        sigma_bu = state.get("sigma_bu", sigma_bu)
        sigma_bi = state.get("sigma_bi", sigma_bi)
        sigma_eps = state.get("sigma_eps", sigma_eps)
        gamma = state.get("gamma", gamma)
        tau = state.get("tau", tau)

    if use_clusters:
        g = user_cluster[u]
        h = item_cluster[i]
        G = int(user_cluster.max() + 1)
        H = int(item_cluster.max() + 1)
        if A is None:
            A = rng.normal(0.0, 1.0, size=(G, H))
            A = (A - A.mean()) / (A.std() + 1e-12)
        else:
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (G, H):
                raise ValueError(f"A must have shape ({G},{H})")
    else:
        A = None

    if b_u is None:
        b_u = rng.normal(0.0, sigma_bu, size=U)
    if b_i is None:
        b_i = rng.normal(0.0, sigma_bi, size=I)
    eps = rng.normal(0.0, sigma_eps, size=len(pairs))

    if tau is None:
        raise ValueError("tau must be provided when state is None.")

    if use_clusters:
        scores = mu + b_u[u] + b_i[i] + gamma * A[g, h] + eps
    else:
        scores = mu + b_u[u] + b_i[i] + eps
    ratings = scores_to_ratings_fixed(scores, tau)

    R[u, i] = ratings.astype(dtype)
    if return_state:
        state = {
            "A": A,
            "b_u": b_u,
            "b_i": b_i,
            "mu": float(mu),
            "sigma_bu": float(sigma_bu),
            "sigma_bi": float(sigma_bi),
            "sigma_eps": float(sigma_eps),
            "gamma": float(gamma),
            "tau": np.asarray(tau, dtype=np.float64),
        }
        return R, state
    return R


def generate_simulation(return_state: bool = False, **overrides):
    
    '''
    Generates a user-item rating matrix based on a cluster-based simulation. The parameters
    can be tweaked by passing them as keyword arguments.   
    '''
    affinity = overrides.pop("affinity", None)
    U, I, d, au, ai, seed, min_user_ratings, min_item_ratings, n_user_clusters, n_item_clusters, beta, return_clusters, tau, mu, sigma_bu, sigma_bi, sigma_eps, gamma = default_values(
        **overrides
    )
    pairs, user_cluster, item_cluster, A, pi = generate_mask_zipf_global(
        U=U,
        I=I,
        density=d,
        alpha_user=au,
        alpha_item=ai,
        seed=seed,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        affinity=affinity,
        beta=beta,
        return_clusters=True,
    )
    result = fill_ratings_clusters(
        U=U,
        I=I,
        pairs=pairs,
        user_cluster=user_cluster,
        item_cluster=item_cluster,
        tau=tau,
        seed=seed + 1,
        mu=mu,
        sigma_bu=sigma_bu,
        sigma_bi=sigma_bi,
        sigma_eps=sigma_eps,
        gamma=gamma,
        A=A,
        return_state=return_state,
    )
    if return_state:
        R, state = result
        # Persist clusters used to generate the simulation.
        state["user_cluster"] = user_cluster
        state["item_cluster"] = item_cluster
        state["pi"] = pi
        return R, state
    return result


def plot_simulation_distributions_and_heatmap(
    R_syn: np.ndarray | pd.DataFrame,
    heatmap_users: int = 3000,
    heatmap_items: int = 1682,
    heatmap_title: str = "",
):
    """
    Plot rating distribution, ratings-per-user/item histograms, and a 5-color heatmap.
    """
    ratings_df = pd.DataFrame(R_syn)
    density = ratings_df.notna().sum().sum() / ratings_df.size
    print(f"Density: {density}")
    vals = ratings_df.values
    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        raise ValueError("R_syn contains no observed ratings (all values are NaN).")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    rating_counts = ratings_df.stack().astype(int).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    axes[0].bar(rating_counts.index, rating_counts.values, color="steelblue", width=0.8)
    axes[0].set_title("Rating value distribution")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks([1, 2, 3, 4, 5])

    user_counts = mask.sum(axis=1)
    sns.histplot(user_counts, bins=30, ax=axes[1], color="teal")
    axes[1].set_title("Ratings per user")
    axes[1].set_xlabel("# ratings")
    axes[1].set_ylabel("Users")
    axes[1].set_yscale("log")

    item_counts = mask.sum(axis=0)
    sns.histplot(item_counts, bins=30, ax=axes[2], color="darkorange")
    axes[2].set_title("Ratings per item")
    axes[2].set_xlabel("# ratings")
    axes[2].set_ylabel("Items")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.show()

    subset = ratings_df.iloc[:heatmap_users, :heatmap_items]
    subset_vals = subset.to_numpy()
    subset_masked = np.ma.masked_invalid(subset_vals)

    colors = ["#b41f1f", "#e15f1d", "#ffcc00", "#6eff0e", "#2787d6"]
    cmap = ListedColormap(colors).with_extremes(bad="white")
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig2, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        subset_masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(heatmap_title)
    ax.set_xlabel("Item index")
    ax.set_ylabel("User index")
    fig2.colorbar(im, ax=ax, label="Rating value", ticks=[1, 2, 3, 4, 5])
    plt.show()

    return fig, fig2


def plot_cluster_ordered_heatmap(
    R_syn: np.ndarray | pd.DataFrame,
    user_cluster: np.ndarray,
    item_cluster: np.ndarray,
    A: np.ndarray | None = None,
    heatmap_users: int | None = 3000,
    heatmap_items: int | None = 1682,
    heatmap_title: str = "Rating heatmap ordered by user/item cluster",
):
    """
    Plot a 5-color heatmap like plot_simulation_distributions_and_heatmap, ordering
    users and items by their cluster labels and adding cluster color strips:
    left strip for users and top strip for items.
    """
    ratings_df = pd.DataFrame(R_syn)
    n_users, n_items = ratings_df.shape

    user_cluster = np.asarray(user_cluster)
    item_cluster = np.asarray(item_cluster)
    if user_cluster.shape[0] != n_users:
        raise ValueError("user_cluster length must match number of users (rows in R_syn).")
    if item_cluster.shape[0] != n_items:
        raise ValueError("item_cluster length must match number of items (cols in R_syn).")

    # Stable sort keeps original order inside each cluster.
    user_order = np.argsort(user_cluster, kind="stable")
    item_order = np.argsort(item_cluster, kind="stable")

    ordered = ratings_df.iloc[user_order, item_order]
    if heatmap_users is not None:
        ordered = ordered.iloc[:heatmap_users, :]
    if heatmap_items is not None:
        ordered = ordered.iloc[:, :heatmap_items]

    shown_users = ordered.shape[0]
    shown_items = ordered.shape[1]
    user_cluster_ord = user_cluster[user_order][:shown_users]
    item_cluster_ord = item_cluster[item_order][:shown_items]

    def _cluster_layout(cluster_ord: np.ndarray):
        labels, counts = np.unique(cluster_ord, return_counts=True)
        ends = np.cumsum(counts)
        starts = ends - counts
        centers = starts + (counts - 1) / 2.0
        return labels, starts, counts, centers

    user_labels, user_starts, user_counts, user_centers = _cluster_layout(user_cluster_ord)
    item_labels, item_starts, item_counts, item_centers = _cluster_layout(item_cluster_ord)

    ordered_vals = ordered.to_numpy()
    if np.isnan(ordered_vals).all():
        raise ValueError("R_syn contains no observed ratings (all values are NaN).")
    ordered_masked = np.ma.masked_invalid(ordered_vals)

    colors = ["#b41f1f", "#e15f1d", "#ffcc00", "#6eff0e", "#2787d6"]
    cmap = ListedColormap(colors).with_extremes(bad="white")
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=[0.05, 0.91, 0.04],
        height_ratios=[0.08, 0.86, 0.06],
        wspace=0.02,
        hspace=0.03,
    )

    ax_corner = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_corner_right_top = fig.add_subplot(gs[0, 2])
    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)
    cax_rating = fig.add_subplot(gs[1, 2])
    ax_corner_bottom_left = fig.add_subplot(gs[2, 0])
    cax_affinity = fig.add_subplot(gs[2, 1])
    ax_corner_bottom_right = fig.add_subplot(gs[2, 2])

    # Top strip: keep only labels (no color fill).
    ax_top.set_facecolor("white")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_ylim(0, 1)

    # Left strip: keep only labels (no color fill).
    ax_left.set_facecolor("white")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_xlim(0, 1)

    for spine in ax_top.spines.values():
        spine.set_visible(False)
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    im = ax_main.imshow(
        ordered_masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax_main.set_title(heatmap_title)
    ax_main.set_xlabel("Item clusters")
    ax_main.set_ylabel("User clusters")

    # Keep main heatmap clean; cluster ids are drawn inside the color strips.
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    top_text_transform = blended_transform_factory(ax_top.transData, ax_top.transAxes)
    left_text_transform = blended_transform_factory(ax_left.transAxes, ax_left.transData)

    for xc, c in zip(item_centers, item_labels):
        ax_top.text(
            xc,
            0.1,
            str(int(c)),
            transform=top_text_transform,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            clip_on=False,
        )
    for yc, c in zip(user_centers, user_labels):
        ax_left.text(
            0.8,
            yc,
            str(int(c)),
            transform=left_text_transform,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            clip_on=False,
        )

    if A is not None:
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("A must be 2D with shape (n_user_clusters, n_item_clusters).")
        max_user_label = int(user_labels.max()) if user_labels.size > 0 else -1
        max_item_label = int(item_labels.max()) if item_labels.size > 0 else -1
        if A.shape[0] <= max_user_label or A.shape[1] <= max_item_label:
            raise ValueError("A shape must cover all shown user/item cluster labels.")

        aff_abs = float(np.nanmax(np.abs(A))) if A.size > 0 else 1.0
        aff_abs = max(aff_abs, 1e-12)
        # Use a custom diverging map (purple -> white -> teal) to avoid overlap
        # with the rating palette colors.
        aff_cmap = LinearSegmentedColormap.from_list(
            "affinity_purple_teal",
            ["#5b2a86", "#f7f7f7", "#008b8b"],
        )
        aff_norm = plt.Normalize(vmin=-aff_abs, vmax=aff_abs)

        for ug, us, uc in zip(user_labels, user_starts, user_counts):
            for ih, is_, ic in zip(item_labels, item_starts, item_counts):
                edge_color = aff_cmap(aff_norm(A[int(ug), int(ih)]))
                rect = Rectangle(
                    (float(is_) - 0.5, float(us) - 0.5),
                    float(ic),
                    float(uc),
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=1.0,
                    alpha=0.95,
                )
                ax_main.add_patch(rect)

        # Affinity legend for boundary colors at the bottom (dedicated axis).
        aff_sm = plt.cm.ScalarMappable(cmap=aff_cmap, norm=aff_norm)
        aff_sm.set_array([])
        fig.colorbar(
            aff_sm,
            cax=cax_affinity,
            orientation="horizontal",
            label="Affinity (boundary color)",
        )
    else:
        cax_affinity.axis("off")

    # Keep strips perfectly aligned with the main heatmap.
    ax_top.set_xlim(ax_main.get_xlim())
    ax_left.set_ylim(ax_main.get_ylim())

    fig.colorbar(im, cax=cax_rating, label="Rating value", ticks=[1, 2, 3, 4, 5])

    ax_corner.axis("off")
    ax_corner_right_top.axis("off")
    ax_corner_bottom_left.axis("off")
    ax_corner_bottom_right.axis("off")
    plt.show()

    return fig, user_order, item_order






