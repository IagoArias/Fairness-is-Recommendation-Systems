import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
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


@st.cache_data(show_spinner=False)
def build_ratings(
    U: int,
    I: int,
    density: float,
    alpha_user: float,
    alpha_item: float,
    seed: int,
    min_user_ratings: int | None,
    min_item_ratings: int | None,
    n_user_clusters: int,
    n_item_clusters: int,
    beta: float,
    tau: tuple[float, float, float, float],
    mu: float,
    sigma_bu: float,
    sigma_bi: float,
    sigma_eps: float,
    gamma: float,
):
    pairs, uc, ic, A = generate_mask_zipf_global(
        U,
        I,
        density=density,
        alpha_user=alpha_user,
        alpha_item=alpha_item,
        seed=seed,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        beta=beta,
        return_clusters=True,
    )
    R_syn = fill_ratings_clusters(
        U=U,
        I=I,
        pairs=pairs,
        user_cluster=uc,
        item_cluster=ic,
        tau=np.array(tau),
        seed=seed,
        mu=mu,
        sigma_bu=sigma_bu,
        sigma_bi=sigma_bi,
        sigma_eps=sigma_eps,
        gamma=gamma,
        A=A,
    )
    return R_syn


st.set_page_config(page_title="Cluster Simulation", layout="wide")
st.title("Cluster Simulation: Distributions and Heatmap")

with st.sidebar:
    st.header("Simulation Parameters")
    U = st.number_input("Users (U)", min_value=10, max_value=5000, value=1000, step=50)
    I = st.number_input("Items (I)", min_value=10, max_value=5000, value=2000, step=50)
    density = st.slider("Density", min_value=0.001, max_value=0.2, value=0.063, step=0.001)
    alpha_user = st.slider("Alpha user", min_value=0.0, max_value=2.0, value=0.9, step=0.05)
    alpha_item = st.slider("Alpha item", min_value=0.0, max_value=2.0, value=0.9, step=0.05)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
    min_user_ratings = st.number_input(
        "Min ratings per user",
        min_value=0,
        max_value=500,
        value=20,
        step=1,
    )
    min_item_ratings = st.number_input(
        "Min ratings per item",
        min_value=0,
        max_value=500,
        value=5,
        step=1,
    )
    st.divider()
    st.subheader("Clusters")
    n_user_clusters = st.number_input(
        "User clusters",
        min_value=2,
        max_value=50,
        value=12,
        step=1,
    )
    n_item_clusters = st.number_input(
        "Item clusters",
        min_value=2,
        max_value=50,
        value=18,
        step=1,
    )
    beta = st.slider("Beta (taste strength)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    st.divider()
    st.subheader("Rating Model")
    mu = st.slider("mu", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    sigma_bu = st.slider("sigma_bu", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    sigma_bi = st.slider("sigma_bi", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    sigma_eps = st.slider("sigma_eps", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    gamma = st.slider("gamma", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    st.subheader("Thresholds")
    t1 = st.slider("t1", min_value=-2.0, max_value=2.0, value=-1.2, step=0.1)
    t2 = st.slider("t2", min_value=-2.0, max_value=2.0, value=-0.4, step=0.1)
    t3 = st.slider("t3", min_value=-2.0, max_value=2.0, value=0.4, step=0.1)
    t4 = st.slider("t4", min_value=-2.0, max_value=2.0, value=1.2, step=0.1)

    st.divider()
    st.subheader("Heatmap")
    heatmap_users = st.number_input(
        "Users shown", min_value=10, max_value=500, value=100, step=10
    )
    heatmap_items = st.number_input(
        "Items shown", min_value=10, max_value=500, value=100, step=10
    )

tau = (t1, t2, t3, t4)
if not (t1 < t2 < t3 < t4):
    st.error("Thresholds must be strictly increasing.")
    st.stop()

R_syn = build_ratings(
    U=U,
    I=I,
    density=density,
    alpha_user=alpha_user,
    alpha_item=alpha_item,
    seed=seed,
    min_user_ratings=min_user_ratings if min_user_ratings > 0 else None,
    min_item_ratings=min_item_ratings if min_item_ratings > 0 else None,
    n_user_clusters=n_user_clusters,
    n_item_clusters=n_item_clusters,
    beta=beta,
    tau=tau,
    mu=mu,
    sigma_bu=sigma_bu,
    sigma_bi=sigma_bi,
    sigma_eps=sigma_eps,
    gamma=gamma,
)

ratings_df = pd.DataFrame(R_syn)
vals = ratings_df.values
mask = ~np.isnan(vals)

if mask.sum() == 0:
    st.warning("No ratings generated with the current settings.")
    st.stop()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.histplot(ratings_df.stack(), bins=np.arange(0.5, 5.6, 0.5), ax=axes[0], color="steelblue")
axes[0].set_title("Rating value distribution")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")

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
st.pyplot(fig, clear_figure=True)

st.subheader("Rating heatmap (subset)")
subset = ratings_df.iloc[:heatmap_users, :heatmap_items]
subset_vals = subset.to_numpy()
subset_masked = np.ma.masked_invalid(subset_vals)

colors = ["#b41f1f", "#e15f1d", "#ffcc00", "#6eff0e", "#2787d6"]
cmap = ListedColormap(colors).with_extremes(bad="white")
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = BoundaryNorm(bounds, cmap.N)

fig2, ax2 = plt.subplots(figsize=(8, 6))
im = ax2.imshow(
    subset_masked,
    aspect="auto",
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
)
ax2.set_title(f"Rating heatmap ({heatmap_users} users x {heatmap_items} items)")
ax2.set_xlabel("Item index")
ax2.set_ylabel("User index")
fig2.colorbar(im, ax=ax2, label="Rating value", ticks=[1, 2, 3, 4, 5])
st.pyplot(fig2, clear_figure=True)
