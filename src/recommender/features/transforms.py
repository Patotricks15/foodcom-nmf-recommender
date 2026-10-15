"""Feature transforms.

Every transform in this module takes the *training partition only* and, where
the value depends on when it is computed, an explicit ``as_of`` timestamp.
That signature is the point: a transform cannot accidentally read the future
because it is never handed the future.

Food.com reviews carry real Unix event times, so time-aware features here are
expressed in calendar days rather than in interaction-count proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400.0


def _empty(dtype: str = "float64") -> pd.Series:
    return pd.Series(dtype=dtype)


def user_activity_count(train: pd.DataFrame, log_transform: bool = True) -> pd.Series:
    """Number of training interactions per user.

    Args:
        train: Training partition.
        log_transform: Apply ``log1p``. Activity counts are heavy-tailed; the
            raw scale would let a handful of power reviewers dominate any
            model that consumes this feature linearly.

    Returns:
        Series indexed by ``user_index``.
    """
    if train.empty:
        return _empty()
    counts = train.groupby("user_index").size().astype("float64")
    return np.log1p(counts) if log_transform else counts


def user_mean_rating(train: pd.DataFrame) -> pd.Series:
    """Mean explicit rating per user over the training partition.

    Only positives survive implicit-feedback filtering, so this measures how
    generously a user rates the recipes they already liked, not their
    overall bias.
    """
    if train.empty:
        return _empty()
    return train.groupby("user_index")["rating"].mean().astype("float64")


def user_tenure_days(train: pd.DataFrame) -> pd.Series:
    """Calendar days between a user's first and last training interaction.

    Separates a user with 50 reviews over three years from a user with 50
    reviews in one weekend; the two have very different taste stability.
    """
    if train.empty:
        return _empty()
    grouped = train.groupby("user_index")["timestamp"]
    return ((grouped.max() - grouped.min()) / SECONDS_PER_DAY).astype("float64")


def user_recency_days(train: pd.DataFrame, as_of: float) -> pd.Series:
    """Calendar days from a user's last training event to ``as_of``.

    Args:
        train: Training partition.
        as_of: Point-in-time cut-off, in Unix seconds.

    Returns:
        Series indexed by ``user_index``; larger means staler.

    Raises:
        ValueError: If any training event is at or after ``as_of``, which
            would mean the caller passed a cut-off inside the training window.
    """
    if train.empty:
        return _empty()
    last = train.groupby("user_index")["timestamp"].max()
    if (last >= as_of).any():
        raise ValueError(
            "user_recency_days received training events at or after the "
            f"as_of cut-off {as_of}; the partition is not point-in-time safe."
        )
    return ((as_of - last) / SECONDS_PER_DAY).astype("float64")


def item_popularity_rank(
    train: pd.DataFrame, n_items: int, normalize: bool = True
) -> pd.Series:
    """Dense popularity rank over all recipes, 0 being the most popular.

    Recipes unseen in training share the last rank rather than being
    dropped, so the series is total over the catalogue and the serving path
    never has to handle a missing key.
    """
    counts = (
        train.groupby("item_index").size()
        if not train.empty
        else pd.Series(dtype="int64")
    )
    counts = counts.reindex(range(n_items), fill_value=0)
    ranks = counts.rank(method="dense", ascending=False).astype("float64") - 1.0
    if normalize and n_items > 1:
        ranks = ranks / float(n_items - 1)
    return ranks


def item_trailing_popularity(
    train: pd.DataFrame,
    n_items: int,
    as_of: float,
    window_days: float = 90.0,
    log_transform: bool = True,
) -> pd.Series:
    """Interaction count inside the ``window_days`` ending at ``as_of``.

    This is the feature the temporal split exists to justify. Lifetime
    popularity is dominated by whatever recipe went viral years ago; a
    trailing window tracks what is popular *now*, which is what a
    recommender serving traffic at ``as_of`` actually needs.

    Args:
        train: Training partition.
        n_items: Size of the item catalogue.
        as_of: Point-in-time cut-off, in Unix seconds.
        window_days: Width of the trailing window.
        log_transform: Apply ``log1p`` to the window count.

    Returns:
        Series indexed by ``item_index``, total over the catalogue.

    Raises:
        ValueError: If ``window_days`` is not positive.
    """
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}.")
    if train.empty:
        return pd.Series(np.zeros(n_items), index=range(n_items), dtype="float64")
    window_start = as_of - window_days * SECONDS_PER_DAY
    # Half-open window [window_start, as_of): an event at as_of is future.
    recent = train[(train["timestamp"] >= window_start) & (train["timestamp"] < as_of)]
    counts = recent.groupby("item_index").size().reindex(range(n_items), fill_value=0)
    counts = counts.astype("float64")
    return np.log1p(counts) if log_transform else counts


def item_category_tokens(catalogue: pd.DataFrame, item_ids: np.ndarray) -> pd.Series:
    """Recipe category label, reindexed to internal item indices.

    Content signal available for recipes with no interactions at all, which
    is the only thing a purely collaborative model cannot supply for a cold
    recipe.

    Args:
        catalogue: Raw ``recipes.csv`` frame.
        item_ids: Original recipe ids positionally aligned to item indices.

    Returns:
        Series indexed by ``item_index``.
    """
    lookup = catalogue.set_index("recipe_id")["category"]
    values = lookup.reindex(item_ids).fillna("").astype(str).to_numpy()
    return pd.Series(values, index=range(len(item_ids)), dtype="object")


def _nutrition_passthrough(
    catalogue: pd.DataFrame, item_ids: np.ndarray, column: str, fallback: float
) -> pd.Series:
    lookup = catalogue.set_index("recipe_id")[column]
    values = lookup.reindex(item_ids).fillna(fallback).astype("float64").to_numpy()
    return pd.Series(values, index=range(len(item_ids)), dtype="float64")


def item_calories(
    catalogue: pd.DataFrame, item_ids: np.ndarray, fallback: float = 0.0
) -> pd.Series:
    """Calories per serving, passed through from the recipe catalogue.

    Backs the nutrition-aware post-filter directly: the collaborative model
    never sees this value, so the filter is a genuinely independent
    domain-constraint layer rather than something the ranker could learn
    around.
    """
    return _nutrition_passthrough(catalogue, item_ids, "calories", fallback)


def item_sodium_mg(
    catalogue: pd.DataFrame, item_ids: np.ndarray, fallback: float = 0.0
) -> pd.Series:
    """Sodium content in milligrams, passed through from the catalogue."""
    return _nutrition_passthrough(catalogue, item_ids, "sodium_mg", fallback)
