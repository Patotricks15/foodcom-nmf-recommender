"""Command-line entry points for training, evaluation, and recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender.data import (
    build_implicit_feedback,
    load_recipe_metadata,
    load_reviews,
    temporal_split,
)
from recommender.evaluation import evaluate, evaluate_nutrition_filter
from recommender.models import (
    ItemKNNRecommender,
    NMFRecommender,
    NutritionConstraints,
    PopularityRecommender,
    index_catalogue,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_N_FACTORS = 16


def _load():
    reviews = load_reviews(ROOT / "data" / "reviews.csv")
    recipes = load_recipe_metadata(ROOT / "data" / "recipes.csv")
    data = build_implicit_feedback(reviews)
    return data, recipes, temporal_split(data)


def _fit_models(split):
    return {
        "NMF": NMFRecommender(n_factors=DEFAULT_N_FACTORS).fit(split),
        "ItemKNN": ItemKNNRecommender(top_k_neighbours=50).fit(split),
        "Popularity": PopularityRecommender().fit(split),
    }


def train_command(_: argparse.Namespace) -> None:
    _, _, split = _load()
    model = NMFRecommender(n_factors=DEFAULT_N_FACTORS).fit(split)
    print(
        json.dumps(
            {
                "n_factors": model.n_factors,
                "n_iter": model.n_iter_,
                "reconstruction_error": model.reconstruction_error_,
            }
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    data, recipes, split = _load()
    rows = []
    models = _fit_models(split)
    for name, model in models.items():
        row = evaluate(model, split, k=args.k, partition=args.partition, model_name=name).as_row()
        rows.append(row)

    if args.nutrition_filter:
        indexed = index_catalogue(recipes, data.item_ids)
        constraints = NutritionConstraints(
            max_calories=args.max_calories, max_sodium_mg=args.max_sodium_mg
        )
        impact = evaluate_nutrition_filter(
            models["NMF"], split, indexed, constraints, k=args.k, partition=args.partition
        )
        rows.append({"model": "NMF+nutrition-filter", **impact.as_row()})

    print(json.dumps(rows, indent=2))


def recommend_command(args: argparse.Namespace) -> None:
    data, recipes, split = _load()
    matches = (data.user_ids == args.user_id).nonzero()[0]
    if matches.size == 0:
        raise SystemExit(f"Unknown user id {args.user_id}.")
    user = int(matches[0])
    model = NMFRecommender(n_factors=DEFAULT_N_FACTORS).fit(split)
    k = args.k

    if args.nutrition_filter:
        from recommender.models import apply_nutrition_filter  # noqa: PLC0415

        candidates = model.recommend(
            user, k=max(args.candidate_pool, k), exclude=split.user_positives().get(user, set())
        )
        indexed = index_catalogue(recipes, data.item_ids)
        constraints = NutritionConstraints(
            max_calories=args.max_calories, max_sodium_mg=args.max_sodium_mg
        )
        ranked = apply_nutrition_filter(candidates, indexed, constraints).kept_item_indices[:k]
    else:
        ranked = model.recommend(user, k=k, exclude=split.user_positives().get(user, set()))

    lookup = recipes.set_index("recipe_id")
    payload = [
        {
            "rank": rank,
            "recipe_id": int(data.item_ids[item]),
            "name": str(lookup.loc[data.item_ids[item], "name"]),
        }
        for rank, item in enumerate(ranked, start=1)
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.set_defaults(func=train_command)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.add_argument("--nutrition-filter", action="store_true")
    evaluate_parser.add_argument("--max-calories", type=float, default=700.0)
    evaluate_parser.add_argument("--max-sodium-mg", type=float, default=1500.0)
    evaluate_parser.set_defaults(func=evaluate_command)

    recommend = commands.add_parser("recommend")
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--candidate-pool", type=int, default=50)
    recommend.add_argument("--nutrition-filter", action="store_true")
    recommend.add_argument("--max-calories", type=float, default=700.0)
    recommend.add_argument("--max-sodium-mg", type=float, default=1500.0)
    recommend.set_defaults(func=recommend_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
