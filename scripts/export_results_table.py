"""Run the classical benchmark and export Markdown and LaTeX tables."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import (  # noqa: E402
    build_implicit_feedback,
    load_recipe_metadata,
    load_reviews,
    temporal_split,
)
from recommender.evaluation import evaluate, evaluate_nutrition_filter  # noqa: E402
from recommender.models import (  # noqa: E402
    ItemKNNRecommender,
    NMFRecommender,
    NutritionConstraints,
    PopularityRecommender,
    index_catalogue,
)


def main() -> None:
    reviews = load_reviews(ROOT / "data" / "reviews.csv")
    recipes = load_recipe_metadata(ROOT / "data" / "recipes.csv")
    data = build_implicit_feedback(reviews)
    split = temporal_split(data)

    models = {
        "NMF": NMFRecommender(n_factors=16, random_state=42).fit(split),
        "ItemKNN": ItemKNNRecommender(top_k_neighbours=50).fit(split),
        "Popularity": PopularityRecommender().fit(split),
    }
    reports = [evaluate(model, split, model_name=name).as_row() for name, model in models.items()]

    indexed = index_catalogue(recipes, data.item_ids)
    constraints = NutritionConstraints(max_calories=450.0, max_sodium_mg=600.0)
    filter_impact = evaluate_nutrition_filter(
        models["NMF"], split, indexed, constraints, k=10, candidate_pool=50
    ).as_row()

    markdown = [
        "| Model | NDCG@10 | Recall@10 | MAP@10 | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & NDCG@10 & Recall@10 & MAP@10 & Coverage \\",
        r"\midrule",
    ]
    for row in reports:
        markdown.append(
            f"| {row['model']} | {row['ndcg']:.4f} | {row['recall']:.4f} | "
            f"{row['map']:.4f} | {row['coverage']:.4f} |"
        )
        latex.append(
            f"{row['model']} & {row['ndcg']:.4f} & {row['recall']:.4f} & "
            f"{row['map']:.4f} & {row['coverage']:.4f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])

    filter_latex = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"Mean fraction of candidates removed & {filter_impact['mean_fraction_removed']:.4f} \\\\",
        f"NDCG@10 unfiltered & {filter_impact['ndcg_unfiltered']:.4f} \\\\",
        f"NDCG@10 filtered & {filter_impact['ndcg_filtered']:.4f} \\\\",
        f"NDCG@10 delta & {filter_impact['ndcg_delta']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    markdown.append("")
    markdown.append("| Nutrition filter impact (NMF, k=10, pool=50) | Value |")
    markdown.append("|---|---:|")
    removed = filter_impact["mean_fraction_removed"]
    markdown.append(f"| Mean fraction of candidates removed | {removed:.4f} |")
    markdown.append(f"| NDCG@10 unfiltered | {filter_impact['ndcg_unfiltered']:.4f} |")
    markdown.append(f"| NDCG@10 filtered | {filter_impact['ndcg_filtered']:.4f} |")
    markdown.append(f"| NDCG@10 delta | {filter_impact['ndcg_delta']:.4f} |")

    table_dir = ROOT / "paper" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "results.tex").write_text("\n".join(latex) + "\n")
    (table_dir / "filter-impact.tex").write_text("\n".join(filter_latex) + "\n")

    readme = ROOT / "README.md"
    content = readme.read_text()
    replacement = "<!-- RESULTS:START -->\n" + "\n".join(markdown) + "\n<!-- RESULTS:END -->"
    content = re.sub(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    readme.write_text(content)


if __name__ == "__main__":
    main()
