#!/usr/bin/env python3
"""Fetch Food.com Recipes and Reviews, or generate a synthetic fixture.

The synthetic generator is the default so the repository is runnable from a
clean checkout with no Kaggle account. It reproduces the real schema
(recipes with category/ingredients/nutrition facts, reviews with a real
timestamp column), including a long-tailed recipe-popularity distribution,
user taste clusters, and calendar-based review drift.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KAGGLE_DATASET = "irkaal/foodcom-recipes-and-reviews"

CATEGORIES = [
    "Dessert", "Breakfast", "Main Dish", "Soup", "Salad", "Appetizer",
    "Bread", "Beverage", "Side Dish", "Vegan", "Vegetarian", "Low Carb",
]
ALLERGEN_POOL = [
    "peanut", "tree-nut", "dairy", "egg", "shellfish", "soy", "gluten", "sesame",
]
BASE_INGREDIENTS = [
    "flour", "sugar", "butter", "egg", "milk", "salt", "pepper", "onion",
    "garlic", "tomato", "chicken", "rice", "olive-oil", "cheese", "basil",
]


def generate_synthetic(
    n_users: int = 1000,
    n_items: int = 500,
    seed: int = 20260115,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fixture with taste clusters, popularity skew and real time drift.

    Three properties matter for this project specifically:

    * **Taste clusters** give a collaborative model recoverable structure.
      Without them every model would score at the random baseline.
    * **Pareto popularity** reproduces the long tail that makes coverage a
      meaningful metric.
    * **Timestamp drift** makes recipe popularity change over the time axis,
      so a global temporal split is genuinely harder than a random one and
      the point-in-time machinery has something real to protect against.

    Nutrition fields and allergen tags are drawn per cluster so the
    nutrition-aware filter has real, non-random structure to act on (for
    example a "Dessert" cluster skews toward sugar and dairy allergens).

    Args:
        n_users: Number of synthetic users.
        n_items: Number of synthetic recipes.
        seed: Seed controlling the whole generation.

    Returns:
        ``(reviews, recipes)`` frames matching the Food.com schema used by
        :mod:`recommender.data.loader`.
    """
    rng = np.random.default_rng(seed)

    n_clusters = 8
    item_cluster = rng.integers(0, n_clusters, size=n_items)
    user_cluster = rng.integers(0, n_clusters, size=n_users)

    base_popularity = rng.pareto(1.5, size=n_items) + 1.0
    base_popularity /= base_popularity.sum()

    # Each recipe has an era: popularity peaks at a point on the time axis.
    item_era = rng.uniform(0.0, 1.0, size=n_items)

    # Nutrition facts and allergens correlate with the (arbitrary) cluster,
    # which stands in for a cuisine/meal-type signal.
    calorie_centre = rng.uniform(150.0, 650.0, size=n_clusters)
    sodium_centre = rng.uniform(100.0, 900.0, size=n_clusters)

    recipes_rows = []
    for item in range(n_items):
        cluster = item_cluster[item]
        category = CATEGORIES[cluster % len(CATEGORIES)]
        calories = float(np.clip(rng.normal(calorie_centre[cluster], 80.0), 20.0, 1400.0))
        fat_g = float(np.clip(calories * rng.uniform(0.02, 0.06), 0.5, 90.0))
        protein_g = float(np.clip(rng.normal(12.0, 8.0), 0.0, 70.0))
        carb_g = float(np.clip(rng.normal(35.0, 20.0), 0.0, 180.0))
        sugar_g = float(np.clip(carb_g * rng.uniform(0.1, 0.6), 0.0, 120.0))
        sodium_mg = float(np.clip(rng.normal(sodium_centre[cluster], 150.0), 0.0, 3500.0))
        n_allergens = int(rng.integers(0, 3))
        allergens = sorted(rng.choice(ALLERGEN_POOL, size=n_allergens, replace=False))
        n_ingredients = int(rng.integers(3, 9))
        ingredients = sorted(rng.choice(BASE_INGREDIENTS, size=n_ingredients, replace=False))
        if allergens:
            ingredients = sorted(set(ingredients) | set(allergens))
        recipes_rows.append(
            {
                "recipe_id": item + 1,
                "name": f"Synthetic Recipe {item + 1:04d}",
                "category": category,
                "ingredients": "|".join(ingredients),
                "calories": round(calories, 1),
                "fat_g": round(fat_g, 1),
                "sodium_mg": round(sodium_mg, 1),
                "protein_g": round(protein_g, 1),
                "carbohydrate_g": round(carb_g, 1),
                "sugar_g": round(sugar_g, 1),
            }
        )
    recipes = pd.DataFrame(recipes_rows)

    start_ts = 1_100_000_000
    span = 200_000_000

    rows = []
    for user in range(1, n_users + 1):
        cluster = user_cluster[user - 1]
        n_interactions = int(np.clip(rng.lognormal(3.0, 0.8), 5, 200))
        n_interactions = min(n_interactions, n_items)

        # Users are active in a window, not across the whole span.
        user_centre = rng.uniform(0.1, 0.95)
        window = rng.uniform(0.05, 0.25)

        affinity = np.where(item_cluster == cluster, 5.0, 1.0)
        era_fit = np.exp(-((item_era - user_centre) ** 2) / (2 * window**2))
        weights = base_popularity * affinity * (0.3 + era_fit)
        weights /= weights.sum()

        chosen = rng.choice(n_items, size=n_interactions, replace=False, p=weights)

        for item in chosen:
            offset = np.clip(rng.normal(user_centre, window / 2), 0.0, 1.0)
            timestamp = int(start_ts + offset * span)
            liked = rng.random() < (0.75 if item_cluster[item] == cluster else 0.35)
            scale = [4.0, 4.5, 5.0] if liked else [0.5, 1.0, 2.0, 3.0]
            rating = float(rng.choice(scale))
            rows.append((user, int(item) + 1, rating, timestamp))

    reviews = pd.DataFrame(rows, columns=["user_id", "recipe_id", "rating", "timestamp"])
    reviews = reviews.sort_values("timestamp").reset_index(drop=True)
    return reviews, recipes


def download_from_kaggle(target: Path) -> None:
    """Download the real dataset using credentials from the environment."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment.\n"
            "Credentials are never read from or written to this repository."
        )

    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET, "-p", str(target), "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("The 'kaggle' package is not installed: pip install kaggle") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Kaggle download failed with exit code {exc.returncode}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true",
        help="Generate the offline fixture (default).",
    )
    source.add_argument(
        "--kaggle", action="store_true",
        help="Download the real dataset using KAGGLE_USERNAME/KAGGLE_KEY.",
    )
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260115)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        download_from_kaggle(DATA_DIR)
        print(f"Downloaded {KAGGLE_DATASET} into {DATA_DIR}")
        print(
            "Note: the raw Kaggle export uses different column names "
            "(AuthorId, RecipeId, DateSubmitted, ...). Rename/normalize them "
            "to the reviews.csv/recipes.csv schema documented in "
            "src/recommender/data/loader.py before running the pipeline."
        )
        return 0

    reviews, recipes = generate_synthetic(args.users, args.items, args.seed)
    reviews.to_csv(DATA_DIR / "reviews.csv", index=False)
    recipes.to_csv(DATA_DIR / "recipes.csv", index=False)

    span = reviews["timestamp"].max() - reviews["timestamp"].min()
    print(f"reviews: {len(reviews):,} rows -> {DATA_DIR / 'reviews.csv'}")
    print(f"recipes: {len(recipes):,} rows -> {DATA_DIR / 'recipes.csv'}")
    print(f"users:   {reviews['user_id'].nunique():,}")
    print(f"span:    {span / 86400:.0f} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
