"""Public models API."""

from __future__ import annotations

from recommender.models.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
)
from recommender.models.filters import (
    FilterOutcome,
    NutritionConstraints,
    apply_nutrition_filter,
    index_catalogue,
    recipe_violates,
)
from recommender.models.nmf import NMFRecommender

__all__ = [
    "FilterOutcome",
    "ItemKNNRecommender",
    "NMFRecommender",
    "NutritionConstraints",
    "PopularityRecommender",
    "RandomRecommender",
    "apply_nutrition_filter",
    "index_catalogue",
    "recipe_violates",
]
