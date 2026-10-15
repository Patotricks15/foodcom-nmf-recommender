"""Behavioral tests for the classical recommender models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from sklearn.decomposition import NMF as SklearnNMF
except ImportError:  # pragma: no cover - sklearn is a dev-only dependency.
    SklearnNMF = None

from recommender.evaluation import evaluate, evaluate_nutrition_filter
from recommender.models import (
    ItemKNNRecommender,
    NMFRecommender,
    NutritionConstraints,
    PopularityRecommender,
    RandomRecommender,
    apply_nutrition_filter,
    index_catalogue,
    recipe_violates,
)


def test_nmf_factors_and_reconstruction_are_non_negative(split):
    model = NMFRecommender(n_factors=12, random_state=7, max_iter=100).fit(split)
    assert model.W is not None and model.H is not None
    assert (model.W >= 0).all()
    assert (model.H >= 0).all()
    assert model.reconstruction_error_ is not None
    assert model.reconstruction_error_ >= 0.0


def test_nmf_objective_is_monotonically_non_increasing(split):
    """The Lee-Seung rule must never increase the Frobenius objective."""
    users, items = split.train_matrix_entries()
    x = np.zeros((split.n_users, split.n_items))
    x[users, items] = 1.0

    rng = np.random.default_rng(3)
    w = rng.uniform(1e-3, 1.0, size=(split.n_users, 8))
    h = rng.uniform(1e-3, 1.0, size=(8, split.n_items))
    eps = 1e-10

    errors = [float(np.linalg.norm(x - w @ h, ord="fro"))]
    for _ in range(25):
        h = h * (w.T @ x) / (w.T @ w @ h + eps)
        w = w * (x @ h.T) / (w @ h @ h.T + eps)
        errors.append(float(np.linalg.norm(x - w @ h, ord="fro")))

    diffs = np.diff(errors)
    assert (diffs <= 1e-6).all(), "Frobenius reconstruction error increased at some iteration."


def test_nmf_is_reproducible(split):
    first = NMFRecommender(n_factors=8, random_state=3, max_iter=50).fit(split)
    second = NMFRecommender(n_factors=8, random_state=3, max_iter=50).fit(split)
    np.testing.assert_allclose(first.score_user(1), second.score_user(1))


def test_nmf_rejects_non_positive_rank(split):
    with pytest.raises(ValueError, match="n_factors must be positive"):
        NMFRecommender(n_factors=0).fit(split)


@pytest.mark.skipif(SklearnNMF is None, reason="scikit-learn not installed")
def test_hand_written_nmf_reconstruction_is_competitive_with_sklearn(split):
    """Cross-check: our multiplicative-update NMF should not grossly
    under-perform scikit-learn's reference implementation on the same
    training matrix, at the same rank and comparable iteration budget.
    """
    users, items = split.train_matrix_entries()
    x = np.zeros((split.n_users, split.n_items))
    x[users, items] = 1.0

    ours = NMFRecommender(n_factors=10, random_state=5, max_iter=300, tol=1e-6).fit(split)

    reference = SklearnNMF(
        n_components=10, init="random", random_state=5, max_iter=300, solver="mu"
    )
    w_ref = reference.fit_transform(x)
    h_ref = reference.components_
    reference_error = float(np.linalg.norm(x - w_ref @ h_ref, ord="fro"))

    # Our implementation should land within a generous factor of sklearn's;
    # the two use different initializations so exact parity is not expected.
    assert ours.reconstruction_error_ < reference_error * 1.5


def test_recommendations_exclude_history(split):
    user, history = next(iter(split.user_positives().items()))
    model = NMFRecommender(n_factors=8, max_iter=50).fit(split)
    ranked = model.recommend(user, k=10, exclude=history)
    assert len(set(ranked) & history) == 0
    assert len(ranked) == 10


def test_cold_user_falls_back_to_item_bias(split):
    model = NMFRecommender(n_factors=8, max_iter=50).fit(split)
    scores = model.score_user(split.n_users + 1)
    assert scores.shape == (split.n_items,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "model",
    [
        PopularityRecommender(),
        ItemKNNRecommender(top_k_neighbours=20),
        RandomRecommender(random_state=1),
    ],
)
def test_baselines_follow_contract(model, split):
    model.fit(split)
    user, history = next(iter(split.user_positives().items()))
    ranked = model.recommend(user, k=7, exclude=history)
    assert ranked.shape == (7,)
    assert len(np.unique(ranked)) == 7


def test_evaluation_reports_ranking_metrics(split):
    model = NMFRecommender(n_factors=8, max_iter=50).fit(split)
    report = evaluate(model, split, k=10, model_name="NMF")
    assert report.model == "NMF"
    assert report.protocol == "temporal"
    assert 0.0 <= report.ndcg <= 1.0
    assert 0.0 <= report.average_precision <= 1.0
    assert 0.0 <= report.coverage <= 1.0


def test_unknown_partition_is_explicit(split):
    model = PopularityRecommender().fit(split)
    with pytest.raises(ValueError, match="partition"):
        evaluate(model, split, partition="future")


# --- Nutrition filter -------------------------------------------------------


def test_recipe_violates_detects_each_constraint_independently():
    row = pd.Series({"calories": 900.0, "sodium_mg": 200.0, "ingredients": "peanut|flour"})
    assert recipe_violates(row, NutritionConstraints(max_calories=700.0))
    assert not recipe_violates(row, NutritionConstraints(max_sodium_mg=1000.0))
    assert recipe_violates(row, NutritionConstraints(exclude_allergens=frozenset({"peanut"})))
    assert not recipe_violates(row, NutritionConstraints())


def test_apply_nutrition_filter_preserves_order_of_survivors(catalogue, interactions):
    indexed = index_catalogue(catalogue, interactions.item_ids)
    ranked = np.arange(min(20, len(indexed)))
    constraints = NutritionConstraints(max_calories=10_000.0)  # nothing violates
    outcome = apply_nutrition_filter(ranked, indexed, constraints)
    np.testing.assert_array_equal(outcome.kept_item_indices, ranked)
    assert outcome.n_removed == 0
    assert outcome.fraction_removed == 0.0


def test_apply_nutrition_filter_removes_high_calorie_recipes(catalogue, interactions):
    indexed = index_catalogue(catalogue, interactions.item_ids)
    ranked = np.arange(len(indexed))
    tight = NutritionConstraints(max_calories=float(indexed["calories"].median()))
    outcome = apply_nutrition_filter(ranked, indexed, tight)
    assert outcome.n_removed > 0
    assert 0.0 < outcome.fraction_removed < 1.0
    for item in outcome.kept_item_indices:
        assert indexed.iloc[int(item)]["calories"] <= tight.max_calories


def test_evaluate_nutrition_filter_reports_removal_and_ndcg_impact(split, interactions, catalogue):
    model = NMFRecommender(n_factors=8, max_iter=50).fit(split)
    indexed = index_catalogue(catalogue, interactions.item_ids)
    tight = NutritionConstraints(max_calories=float(indexed["calories"].quantile(0.3)))
    report = evaluate_nutrition_filter(model, split, indexed, tight, k=10, candidate_pool=40)
    assert 0.0 <= report.mean_fraction_removed <= 1.0
    assert report.n_users > 0
    assert isinstance(report.ndcg_delta, float)


def test_evaluate_nutrition_filter_rejects_undersized_pool(split, interactions, catalogue):
    model = NMFRecommender(n_factors=4, max_iter=30).fit(split)
    indexed = index_catalogue(catalogue, interactions.item_ids)
    with pytest.raises(ValueError, match="candidate_pool"):
        evaluate_nutrition_filter(
            model, split, indexed, NutritionConstraints(), k=20, candidate_pool=10
        )
