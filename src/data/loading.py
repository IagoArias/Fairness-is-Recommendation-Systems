import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


DEFAULT_DATA_PATH = Path("../ml-100k")


def load_movielens_100k(data_path: Path = DEFAULT_DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the MovieLens 100k dataset from a local folder.
    Returns ratings, users, and movies DataFrames.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"The data folder {data_path} does not exist.")

    ratings_df = pd.read_csv(
        data_path / "u.data",
        sep="\t",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
    )

    users_df = pd.read_csv(
        data_path / "u.user",
        sep="|",
        names=["UserID", "Age", "Gender", "Occupation", "Zip-code"],
        engine="python",
    )

    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    movie_cols = ["MovieID", "Title", "ReleaseDate", "VideoReleaseDate", "IMDbURL"] + genre_cols
    movies_df = pd.read_csv(
        data_path / "u.item",
        sep="|",
        names=movie_cols,
        usecols=["MovieID", "Title"] + genre_cols,
        encoding="latin1",
        engine="python",
    )

    return ratings_df, users_df, movies_df


def build_ratings_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot ratings into a user-item matrix."""
    return ratings_df.pivot_table(index="UserID", columns="MovieID", values="Rating")


def build_user_folds(
    ratings_matrix: pd.DataFrame,
    n_splits: int = 5,
    test_ratings_per_user: int = 5,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create user-based folds by hiding `test_ratings_per_user` ratings per user.
    Mirrors the sampling strategy in the original notebook.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    user_ids = ratings_matrix.index.tolist()
    user_indices = np.arange(len(user_ids))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_user_idx, test_user_idx in kf.split(user_indices):
        train_users = [user_ids[i] for i in train_user_idx]
        test_users = [user_ids[i] for i in test_user_idx]

        train_matrix = ratings_matrix.copy()
        test_matrix = pd.DataFrame(np.nan, index=ratings_matrix.index, columns=ratings_matrix.columns)

        for user in test_users:
            user_ratings = ratings_matrix.loc[user].dropna()
            if len(user_ratings) >= test_ratings_per_user:
                test_items = random.sample(list(user_ratings.index), test_ratings_per_user)
                for item in test_items:
                    test_matrix.at[user, item] = ratings_matrix.at[user, item]
                    train_matrix.at[user, item] = np.nan

        folds.append((train_matrix, test_matrix))

    return folds
