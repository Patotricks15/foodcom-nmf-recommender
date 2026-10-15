"""Post-hoc nutrition-aware filter over ranked recipe candidates.

The NMF ranker never sees calories, sodium, or allergens: those live only in
the recipe catalogue, read here directly rather than through the
collaborative model. That separation is deliberate -- a domain constraint
("never serve a peanut recipe to a user with a peanut allergy") must hold
regardless of what the collaborative signal has learned, so it is enforced
as an explicit post-filter on the ranked candidate list rather than folded
into training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NutritionConstraints:
    """Dietary limits applied to a ranked candidate list.

    Attributes:
        max_calories: Reject recipes above this many calories per serving.
            ``None`` disables the check.
        max_sodium_mg: Reject recipes above this much sodium (mg).
            ``None`` disables the check.
        exclude_allergens: Recipes whose ingredient tokens intersect this set
            are rejected, regardless of the other limits.
    """

    max_calories: float | None = None
    max_sodium_mg: float | None = None
    exclude_allergens: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class FilterOutcome:
    """Result of applying a nutrition filter to one ranked candidate list."""

    kept_item_indices: np.ndarray
    removed_item_indices: np.ndarray
    n_candidates: int
    n_removed: int

    @property
    def fraction_removed(self) -> float:
        if self.n_candidates == 0:
            return 0.0
        return self.n_removed / self.n_candidates


def index_catalogue(catalogue: pd.DataFrame, item_ids: np.ndarray) -> pd.DataFrame:
    """Reindex the raw recipe catalogue to internal item indices.

    Args:
        catalogue: Raw ``recipes.csv`` frame.
        item_ids: Original recipe ids positionally aligned to item indices.

    Returns:
        Frame indexed ``0..n_items-1`` with ``calories``, ``sodium_mg`` and
        ``ingredients`` columns, safe to index by internal item index.
    """
    lookup = catalogue.set_index("recipe_id")
    reindexed = lookup.reindex(item_ids).reset_index(drop=True)
    reindexed["ingredients"] = reindexed["ingredients"].fillna("")
    reindexed["calories"] = reindexed["calories"].fillna(0.0)
    reindexed["sodium_mg"] = reindexed["sodium_mg"].fillna(0.0)
    return reindexed


def recipe_violates(row: pd.Series, constraints: NutritionConstraints) -> bool:
    """Whether a single recipe row violates any declared constraint."""
    if constraints.max_calories is not None and row["calories"] > constraints.max_calories:
        return True
    if constraints.max_sodium_mg is not None and row["sodium_mg"] > constraints.max_sodium_mg:
        return True
    if constraints.exclude_allergens:
        tokens = set(str(row["ingredients"]).split("|")) if row["ingredients"] else set()
        if tokens & constraints.exclude_allergens:
            return True
    return False


def apply_nutrition_filter(
    ranked_item_indices: np.ndarray,
    indexed_catalogue: pd.DataFrame,
    constraints: NutritionConstraints,
) -> FilterOutcome:
    """Remove recipes violating ``constraints`` from a ranked candidate list.

    This is a *hard* filter: violating recipes are dropped outright rather
    than down-weighted, which is the appropriate behaviour for a safety
    constraint such as an allergen. Relative order among the survivors is
    preserved, so the filter never re-ranks -- it only removes.

    Args:
        ranked_item_indices: Candidate recipe indices, best first.
        indexed_catalogue: Output of :func:`index_catalogue`.
        constraints: Limits to enforce.

    Returns:
        A :class:`FilterOutcome` with the surviving indices in their
        original relative order.
    """
    kept: list[int] = []
    removed: list[int] = []
    for item in ranked_item_indices:
        row = indexed_catalogue.iloc[int(item)]
        if recipe_violates(row, constraints):
            removed.append(int(item))
        else:
            kept.append(int(item))
    return FilterOutcome(
        kept_item_indices=np.array(kept, dtype=np.int64),
        removed_item_indices=np.array(removed, dtype=np.int64),
        n_candidates=len(ranked_item_indices),
        n_removed=len(removed),
    )
