import numpy as np
import pandas as pd

from src.eval.metrics import evaluate_all_metrics


def mf_fit_biased(
    R_train,
    n_factors: int = 40,
    n_epochs: int = 20,
    lr: float = 0.01,
    reg: float = 0.05,
    random_state: int = 42,
    verbose: bool = False,
):
    """
    Biased MF: r_hat = mu + b_u + b_i + p_u^T q_i
    R_train: numpy matrix (users x items) with NaNs for unknown cells.
    """
    rng = np.random.default_rng(random_state)

    R = R_train.astype(float)
    num_users, num_items = R.shape

    mask = ~np.isnan(R)
    u_idx, i_idx = np.where(mask)
    ratings = R[mask]
    if ratings.size == 0:
        raise ValueError("R_train has no known ratings.")

    mu = float(np.mean(ratings))
    bu = np.zeros(num_users, dtype=float)
    bi = np.zeros(num_items, dtype=float)
    P = 0.1 * rng.standard_normal((num_users, n_factors))
    Q = 0.1 * rng.standard_normal((num_items, n_factors))

    n_obs = ratings.shape[0]
    indices = np.arange(n_obs)

    for epoch in range(n_epochs):
        rng.shuffle(indices)
        for k in indices:
            u = u_idx[k]
            i = i_idx[k]
            r_ui = ratings[k]

            pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
            err = r_ui - pred

            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

            p_u = P[u]
            q_i = Q[i]

            P[u] += lr * (err * q_i - reg * p_u)
            Q[i] += lr * (err * p_u - reg * q_i)

        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs} done")

    return mu, bu, bi, P, Q


def mf_predict_unknown(mu, bu, bi, P, Q, R_train):
    """
    Return matrix R_pred with NaN in known cells and predictions in unknown cells.
    """
    R = R_train.astype(float)
    num_users, num_items = R.shape

    full_pred = mu + bu[:, None] + bi[None, :] + P @ Q.T

    R_pred = np.full((num_users, num_items), np.nan, dtype=float)
    mask_unknown = np.isnan(R)
    R_pred[mask_unknown] = full_pred[mask_unknown]
    return R_pred


def evaluate_mf_with_metrics_on_folds(
    folds,
    n_factors: int = 40,
    n_epochs: int = 20,
    lr: float = 0.01,
    reg: float = 0.05,
    random_state: int = 42,
    topn_nov_rel: int = 20,
    topn_div: int = 5,
    k_ndcg: int = 20,
    max_folds=None,
    verbose: bool = False,
):
    rows = []

    for f, (train_df, test_df) in enumerate(folds, 1):
        if max_folds is not None and f > max_folds:
            break

        if verbose:
            print(f"\n=== Fold {f} ===")

        R_train = train_df.values.astype(float)
        R_test = test_df.values.astype(float)

        mu, bu, bi, P, Q = mf_fit_biased(
            R_train,
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr=lr,
            reg=reg,
            random_state=random_state + f,
            verbose=verbose,
        )

        R_pred = mf_predict_unknown(mu, bu, bi, P, Q, R_train)

        metrics = evaluate_all_metrics(
            R_train,
            R_test,
            R_pred,
            train_df,
            topn_nov_rel=topn_nov_rel,
            topn_div=topn_div,
            k_ndcg=k_ndcg,
        )
        metrics["fold"] = f
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    cols = ["fold"] + [c for c in metrics_df.columns if c != "fold"]
    metrics_df = metrics_df[cols]
    avg_metrics = metrics_df.drop(columns=["fold"]).mean(numeric_only=True).to_dict()
    return metrics_df, avg_metrics
