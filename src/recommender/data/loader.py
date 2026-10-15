"""Load Food.com reviews and recipes and convert them to implicit feedback.

Food.com reviews ship a real ``timestamp`` column (``DateSubmitted`` in the
raw Kaggle export, normalized here to Unix seconds), which is why this
project follows the movielens-svd-ranking convention of a genuine temporal
split rather than the anime-bpr-recommender's per-user sequence proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_REVIEW_COLUMNS = {"user_id", "recipe_id", "rating", "timestamp"}
REQUIRED_RECIPE_COLUMNS = {
    "recipe_id", "name", "category", "ingredients",
    "calories", "fat_g", "sodium_mg", "protein_g", "carbohydrate_g", "sugar_g",
}
MIN_POSITIVE_RATING = 4.0


@dataclass(frozen=True)
class InteractionData:
    """Implicit-feedback interactions with contiguous internal indices.

    Attributes:
        frame: Columns ``user_index``, ``item_index``, ``rating``, ``timestamp``.
        user_ids: Original user ids, positionally aligned to ``user_index``.
        item_ids: Original recipe ids, positionally aligned to ``item_index``.
    """

    frame: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Read reviews.csv and validate its contract.

    Args:
        path: Path to a Food.com-format reviews file.

    Returns:
        The raw frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py --synthetic"
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_REVIEW_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    return frame


def load_recipe_metadata(path: str | Path) -> pd.DataFrame:
    """Read recipes.csv (id, category, ingredients, nutrition facts)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    frame = pd.read_csv(path)
    missing = REQUIRED_RECIPE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def build_implicit_feedback(
    reviews: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> InteractionData:
    """Binarize reviews, apply an iterative k-core filter and reindex.

    A rating of 4.0 or above (on the Food.com 0-5 scale) is a positive. Lower
    ratings are *dropped* rather than treated as negatives: they signal
    dislike, which is strictly more information than the "unobserved" state
    the implicit-feedback model assumes, and they already fall into the
    negative pool by virtue of not being positive.

    Args:
        reviews: Raw Food.com reviews frame.
        min_user_interactions: k-core threshold on the user side.
        min_item_interactions: k-core threshold on the item (recipe) side.

    Returns:
        Interactions with contiguous indices, sorted by timestamp.

    Raises:
        ValueError: If the thresholds eliminate every interaction.
    """
    frame = reviews.loc[reviews["rating"] >= MIN_POSITIVE_RATING].copy()
    frame = frame.drop_duplicates(subset=["user_id", "recipe_id"], keep="first")

    # Iterative: dropping sparse recipes can push a user below the user
    # threshold, so a single pass would leave the invariant violated.
    while True:
        user_counts = frame["user_id"].value_counts()
        item_counts = frame["recipe_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        keep_items = item_counts[item_counts >= min_item_interactions].index

        filtered = frame[frame["user_id"].isin(keep_users) & frame["recipe_id"].isin(keep_items)]
        if len(filtered) == len(frame):
            frame = filtered
            break
        frame = filtered
        if frame.empty:
            break

    if frame.empty:
        raise ValueError(
            "Core filtering removed every interaction; thresholds "
            f"({min_user_interactions}, {min_item_interactions}) are too strict."
        )

    user_ids = np.sort(frame["user_id"].unique())
    item_ids = np.sort(frame["recipe_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    out = pd.DataFrame(
        {
            "user_index": frame["user_id"].map(user_map).astype(np.int64),
            "item_index": frame["recipe_id"].map(item_map).astype(np.int64),
            # Retained after binarization: the rating still carries confidence
            # (a 5.0 is a stronger positive than a 4.0) even though NMF's
            # training matrix treats every surviving row as a 1.
            "rating": frame["rating"].astype(np.float64),
            "timestamp": frame["timestamp"].astype(np.int64),
        }
    )
    # Sort by time first: every downstream temporal operation assumes it.
    out = out.sort_values(["timestamp", "user_index"]).reset_index(drop=True)

    return InteractionData(frame=out, user_ids=user_ids, item_ids=item_ids)
