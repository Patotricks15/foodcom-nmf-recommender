# Food.com Classical NMF Recipe Recommender

Classical Non-negative Matrix Factorization (NMF), hand-written with
multiplicative update rules in NumPy, for ranking Food.com recipes against
real user review timestamps, combined with a post-hoc nutrition-aware filter.
No neural networks, no deep-learning framework, no gradient-descent-trained
embeddings.

> Status: scaffolding in progress. See the commit history for the
> incremental build-out of data loading, features, models, serving,
> infrastructure and the technical paper.

## Why NMF for recipes

Unlike signed factorizations (SVD) or pairwise-ranking embeddings (BPR), NMF
constrains both factor matrices to be non-negative. The latent components are
therefore additive and interpretable as "flavor/ingredient profiles" rather
than opaque signed directions — a natural fit for a recipe catalogue, where
combining collaborative signal with hard domain constraints (calories,
sodium, allergens) is a first-class requirement, not an afterthought.

## Dataset

[Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
(Kaggle ref `irkaal/foodcom-recipes-and-reviews`): recipes with ingredients
and nutrition facts, and roughly 1.4M timestamped user reviews/ratings.

## License and data

Code is MIT licensed. The dataset is distributed by its Kaggle maintainer
under its own terms and is **not** redistributed here;
`scripts/download_data.py` fetches it on demand or generates a synthetic
fixture with the same schema.
