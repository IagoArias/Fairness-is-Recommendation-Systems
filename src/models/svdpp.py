import numpy as np
import pandas as pd

from src.eval.metrics import evaluate_all_metrics

def svdpp_predict_unknown(
    mu,
    bu,
    bi,
    P,
    Q,
    Y,
    user_items,
    R_train,
    use_implicit: bool = True,
    max_items_per_user=None,
    random_state: int = 42,
):
    """Predict only in unknown cells (NaN), leaving known ratings untouched."""
    R = R_train.astype(float)
    num_users, num_items = R.shape

    full_pred = np.zeros((num_users, num_items), dtype=float)

    for u in range(num_users):
        Nu = user_items[u]
        if use_implicit and len(Nu) > 0:
            if max_items_per_user is not None and len(Nu) > max_items_per_user:
                rng_u = np.random.default_rng(random_state + int(u))
                Nu_eff = rng_u.choice(Nu, size=max_items_per_user, replace=False)
            else:
                Nu_eff = Nu

            norm = 1.0 / np.sqrt(len(Nu_eff))
            y_sum = Y[Nu_eff].sum(axis=0) * norm
            user_vec = P[u] + y_sum
        else:
            user_vec = P[u]

        full_pred[u, :] = mu + bu[u] + bi + (Q @ user_vec)

    R_pred = np.full((num_users, num_items), np.nan, dtype=float)
    mask_unknown = np.isnan(R)
    R_pred[mask_unknown] = full_pred[mask_unknown]
    return R_pred


def svdpp_fit_fast(
    R_train,
    n_factors: int = 40,
    n_epochs: int = 20,
    lr: float = 0.01,
    reg: float = 0.05,
    random_state: int = 42,
    use_implicit: bool = True,
    max_items_per_user=None,
    verbose: bool = False,
):
    """
    Faster SVD++ loop:
      - Iterate by user
      - Cache y_sum = |N(u)|^-1/2 * sum Y[j] once per user
      - Aggregate updates to Y per user
    """
    rng = np.random.default_rng(random_state)

    R = R_train.astype(np.float32)
    num_users, num_items = R.shape

    mask = ~np.isnan(R)
    ratings = R[mask]
    if ratings.size == 0:
        raise ValueError("R_train has no known ratings.")

    mu = float(ratings.mean())
    bu = np.zeros(num_users, dtype=np.float32)
    bi = np.zeros(num_items, dtype=np.float32)

    P = (0.1 * rng.standard_normal((num_users, n_factors))).astype(np.float32)
    Q = (0.1 * rng.standard_normal((num_items, n_factors))).astype(np.float32)
    Y = (0.1 * rng.standard_normal((num_items, n_factors))).astype(np.float32)

    user_items_full = []
    user_ratings_full = []
    for u in range(num_users):
        items_u = np.where(mask[u])[0]
        user_items_full.append(items_u.astype(np.int32))
        user_ratings_full.append(R[u, items_u].astype(np.float32))

    for epoch in range(n_epochs):
        users = np.arange(num_users)
        rng.shuffle(users)

        se = 0.0
        n_obs = 0

        for u in users:
            items_u = user_items_full[u]
            if items_u.size == 0:
                continue

            Nu = items_u
            y_sum = 0.0
            norm = 0.0

            if use_implicit and Nu.size > 0:
                if max_items_per_user is not None and Nu.size > max_items_per_user:
                    Nu = rng.choice(Nu, size=max_items_per_user, replace=False)
                norm = (1.0 / np.sqrt(Nu.size)).astype(np.float32)
                y_sum = Y[Nu].sum(axis=0) * norm

            grad_y = np.zeros(n_factors, dtype=np.float32)

            for i, r_ui in zip(items_u, user_ratings_full[u]):
                user_vec = P[u] + (y_sum if (use_implicit and Nu.size > 0) else 0.0)

                pred = mu + bu[u] + bi[i] + np.dot(Q[i], user_vec)
                err = r_ui - pred

                se += float(err * err)
                n_obs += 1

                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])

                p_u_old = P[u].copy()
                q_i_old = Q[i].copy()

                P[u] += lr * (err * q_i_old - reg * p_u_old)
                Q[i] += lr * (err * user_vec - reg * q_i_old)

                if use_implicit and Nu.size > 0:
                    grad_y += err * q_i_old

            if use_implicit and Nu.size > 0:
                m = items_u.size
                Y[Nu] += lr * (norm * grad_y - (reg * m) * Y[Nu])

        if verbose and n_obs > 0:
            rmse = np.sqrt(se / n_obs)
            print(f"Epoch {epoch + 1}/{n_epochs} — train RMSE: {rmse:.4f}")

    return mu, bu, bi, P, Q, Y, user_items_full


def evaluate_svdpp_with_metrics_on_folds(
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
    use_implicit: bool = True,
    max_items_per_user=None,
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

        mu, bu, bi, P, Q, Y, user_items = svdpp_fit_fast(
            R_train,
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr=lr,
            reg=reg,
            random_state=random_state + f,
            use_implicit=use_implicit,
            max_items_per_user=max_items_per_user,
            verbose=verbose,
        )

        R_pred = svdpp_predict_unknown(
            mu,
            bu,
            bi,
            P,
            Q,
            Y,
            user_items,
            R_train,
            use_implicit=use_implicit,
            max_items_per_user=max_items_per_user,
        )

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
