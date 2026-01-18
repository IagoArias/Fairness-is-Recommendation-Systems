import numpy as np
import pandas as pd


def global_mean(train: pd.DataFrame) -> pd.DataFrame:
    mean = np.nanmean(train.values)
    return pd.DataFrame(mean, index=train.index, columns=train.columns)


def item_mean(train: pd.DataFrame) -> pd.DataFrame:
    """Column means; fallback to global mean if a column is entirely NaN."""
    col_means = train.mean(axis=0, skipna=True)
    global_mean_value = train.stack().mean()
    col_means = col_means.fillna(global_mean_value)
    return pd.DataFrame(
        np.tile(col_means.to_numpy(), (train.shape[0], 1)),
        index=train.index,
        columns=train.columns,
    )


def user_mean(train: pd.DataFrame) -> pd.DataFrame:
    """Row means; fallback to global mean if a row is entirely NaN."""
    row_means = train.mean(axis=1, skipna=True)
    global_mean_value = train.stack().mean()
    row_means = row_means.fillna(global_mean_value)
    return pd.DataFrame(
        np.tile(row_means.to_numpy().reshape(-1, 1), (1, train.shape[1])),
        index=train.index,
        columns=train.columns,
    )


def random_baseline(train: pd.DataFrame) -> pd.DataFrame:
    min_rating = np.nanmin(train.values)
    max_rating = np.nanmax(train.values)
    rand_preds = np.random.uniform(min_rating, max_rating, size=train.shape)
    return pd.DataFrame(rand_preds, index=train.index, columns=train.columns)


def popularity_baseline(train: pd.DataFrame) -> pd.DataFrame:
    """Score items by popularity (rating count), normalized to [1,5]."""
    item_popularity = np.sum(~np.isnan(train.values), axis=0)
    preds = np.full(train.shape, np.nan)

    for u in range(train.shape[0]):
        unseen_items = np.isnan(train.values[u])
        preds[u, unseen_items] = item_popularity[unseen_items]

    min_score = np.nanmin(preds)
    max_score = np.nanmax(preds)
    if max_score > min_score:
        preds = 1 + 4 * (preds - min_score) / (max_score - min_score)

    return pd.DataFrame(preds, index=train.index, columns=train.columns)
