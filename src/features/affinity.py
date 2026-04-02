import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def affinity_matrix(
        n_user_clusters: int, 
        n_item_clusters: int, 
        kind: str = "normal",
        seed: int | None = None) -> np.ndarray:
    """
    Build an affinity matrix for user and item clusters.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, .8, size=(n_user_clusters, n_item_clusters))
    if kind == "outliers":
        outlier_share = 0.2
        n_out_user = max(1, int(round(outlier_share * n_user_clusters)))
        n_out_item = max(1, int(round(outlier_share * n_item_clusters)))

        outlier_users = np.arange(n_user_clusters - n_out_user, n_user_clusters)
        outlier_items = np.arange(n_item_clusters - n_out_item, n_item_clusters)

        normal_users = np.setdiff1d(np.arange(n_user_clusters), outlier_users)
        normal_items = np.setdiff1d(np.arange(n_item_clusters), outlier_items)

        # Outlier user clusters dislike normal item clusters and strongly prefer outlier item clusters.
        if normal_items.size > 0:
            A[np.ix_(outlier_users, normal_items)] -= 2
        if outlier_items.size > 0:
            A[np.ix_(outlier_users, outlier_items)] += 1.5

        # Symmetric effect for outlier item clusters versus normal user clusters.
        if normal_users.size > 0:
            A[np.ix_(normal_users, outlier_items)] -= 2.0
        if outlier_users.size > 0:
            A[np.ix_(outlier_users, outlier_items)] += 1.5
    elif kind == "two_types":
        user_split = n_user_clusters // 2
        item_split = n_item_clusters // 2

        user_types = np.zeros(n_user_clusters, dtype=int)
        item_types = np.zeros(n_item_clusters, dtype=int)
        user_types[user_split:] = 1
        item_types[item_split:] = 1

        same_type = user_types[:, None] == item_types[None, :]
        A[same_type] += 1.2
        A[~same_type] -= 1.2
    elif kind != "normal":
        raise ValueError(f"Unknown affinity matrix kind: {kind}")

    return A


def plot_affinity_heatmap(
        A: np.ndarray,
        title: str = "Affinity Matrix",
        cmap: str = "coolwarm",
        annotate: bool = True,
        vmin: float | None = None,
        vmax: float | None = None):
    """
    Plot a heatmap for an affinity matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if vmin is None:
        vmin = float(np.min(A))
    if vmax is None:
        vmax = float(np.max(A))

    if vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        A,
        ax=ax,
        cmap=cmap,
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        annot=annotate,
        fmt=".2f",
        cbar_kws={"label": "Affinity"},
    )
    ax.set_title(title)
    ax.set_xlabel("Item Clusters")
    ax.set_ylabel("User Clusters")
    plt.tight_layout()
    plt.show()
    return fig, ax

    
