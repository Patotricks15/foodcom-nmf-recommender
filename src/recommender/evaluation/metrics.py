"""Top-N ranking metrics and nutrition-filter impact measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from recommender.data.splitting import Split
from recommender.models.filters import NutritionConstraints, apply_nutrition_filter


class Recommender(Protocol):
    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray: ...


def _ndcg_at_k(ranked: np.ndarray, targets: set[int], k: int) -> float:
    gains = sum(
        1.0 / np.log2(position + 2)
        for position, item in enumerate(ranked)
        if int(item) in targets
    )
    ideal = sum(1.0 / np.log2(position + 2) for position in range(min(len(targets), k)))
    return gains / ideal if ideal else 0.0


def _average_precision_at_k(ranked: np.ndarray, targets: set[int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for position, item in enumerate(ranked):
        if int(item) in targets:
            hits += 1
            precision_sum += hits / (position + 1)
    denominator = min(len(targets), k)
    return precision_sum / denominator if denominator else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    model: str
    protocol: str
    k: int
    ndcg: float
    recall: float
    precision: float
    average_precision: float
    hit_rate: float
    coverage: float
    n_users: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "k": self.k,
            "ndcg": round(self.ndcg, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "map": round(self.average_precision, 4),
            "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "n_users": self.n_users,
        }


def evaluate(
    model: Recommender,
    split: Split,
    k: int = 10,
    partition: str = "test",
    model_name: str = "model",
) -> EvaluationReport:
    """Evaluate ranking quality against one or more relevant recipes per user."""
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    frame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    histories = split.user_positives()
    relevant = {
        int(user): set(group["item_index"].astype(int))
        for user, group in frame.groupby("user_index")
    }
    ndcg_total = recall_total = precision_total = ap_total = 0.0
    hits = 0
    recommended: set[int] = set()

    for user, targets in relevant.items():
        ranked = model.recommend(user, k=k, exclude=histories.get(user, set()))
        recommended.update(map(int, ranked))
        positions = [position for position, item in enumerate(ranked) if int(item) in targets]
        if positions:
            hits += 1
        ndcg_total += _ndcg_at_k(ranked, targets, k)
        ap_total += _average_precision_at_k(ranked, targets, k)
        recall_total += len(positions) / len(targets)
        precision_total += len(positions) / k

    n_users = len(relevant)
    return EvaluationReport(
        model=model_name,
        protocol=split.protocol,
        k=k,
        ndcg=ndcg_total / n_users,
        recall=recall_total / n_users,
        precision=precision_total / n_users,
        average_precision=ap_total / n_users,
        hit_rate=hits / n_users,
        coverage=len(recommended) / split.n_items,
        n_users=n_users,
    )


@dataclass(frozen=True)
class FilterImpactReport:
    """How much the nutrition filter changes the ranking, at a fixed k.

    ``ndcg_unfiltered``/``ndcg_filtered`` are computed over the same
    candidate pool, so the delta isolates the effect of the filter rather
    than of a different model or protocol.
    """

    k: int
    candidate_pool: int
    ndcg_unfiltered: float
    ndcg_filtered: float
    recall_unfiltered: float
    recall_filtered: float
    mean_fraction_removed: float
    n_users: int

    @property
    def ndcg_delta(self) -> float:
        return self.ndcg_filtered - self.ndcg_unfiltered

    def as_row(self) -> dict[str, float | int]:
        return {
            "k": self.k,
            "candidate_pool": self.candidate_pool,
            "ndcg_unfiltered": round(self.ndcg_unfiltered, 4),
            "ndcg_filtered": round(self.ndcg_filtered, 4),
            "ndcg_delta": round(self.ndcg_delta, 4),
            "recall_unfiltered": round(self.recall_unfiltered, 4),
            "recall_filtered": round(self.recall_filtered, 4),
            "mean_fraction_removed": round(self.mean_fraction_removed, 4),
            "n_users": self.n_users,
        }


def evaluate_nutrition_filter(
    model: Recommender,
    split: Split,
    indexed_catalogue: pd.DataFrame,
    constraints: NutritionConstraints,
    k: int = 10,
    candidate_pool: int = 50,
    partition: str = "test",
) -> FilterImpactReport:
    """Measure how much the nutrition filter changes the top-``k`` ranking.

    For each evaluated user the model first produces ``candidate_pool``
    unfiltered recommendations. The nutrition filter is then applied to that
    pool, and both the unfiltered top-``k`` and the filtered top-``k`` (the
    first ``k`` survivors) are scored against the same held-out positives.
    The gap between the two isolates the filter's effect from the model's.

    Args:
        model: A fitted recommender exposing ``recommend``.
        split: Partition supplying training history and evaluation targets.
        indexed_catalogue: Output of :func:`recommender.models.filters.index_catalogue`.
        constraints: Nutrition limits to enforce.
        k: Final list length compared between filtered and unfiltered.
        candidate_pool: Size of the pre-filter candidate pool; must be >= k.
        partition: ``"validation"`` or ``"test"``.

    Returns:
        A :class:`FilterImpactReport`.

    Raises:
        ValueError: If ``candidate_pool`` is smaller than ``k``.
    """
    if candidate_pool < k:
        raise ValueError(f"candidate_pool ({candidate_pool}) must be >= k ({k}).")
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    frame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    histories = split.user_positives()
    relevant = {
        int(user): set(group["item_index"].astype(int))
        for user, group in frame.groupby("user_index")
    }

    ndcg_un = ndcg_f = recall_un = recall_f = fraction_removed_total = 0.0

    for user, targets in relevant.items():
        candidates = model.recommend(user, k=candidate_pool, exclude=histories.get(user, set()))
        outcome = apply_nutrition_filter(candidates, indexed_catalogue, constraints)
        fraction_removed_total += outcome.fraction_removed

        unfiltered_top_k = candidates[:k]
        filtered_top_k = outcome.kept_item_indices[:k]

        ndcg_un += _ndcg_at_k(unfiltered_top_k, targets, k)
        ndcg_f += _ndcg_at_k(filtered_top_k, targets, k)
        recall_un += len({i for i in unfiltered_top_k if int(i) in targets}) / len(targets)
        recall_f += len({i for i in filtered_top_k if int(i) in targets}) / len(targets)

    n_users = len(relevant)
    return FilterImpactReport(
        k=k,
        candidate_pool=candidate_pool,
        ndcg_unfiltered=ndcg_un / n_users,
        ndcg_filtered=ndcg_f / n_users,
        recall_unfiltered=recall_un / n_users,
        recall_filtered=recall_f / n_users,
        mean_fraction_removed=fraction_removed_total / n_users,
        n_users=n_users,
    )
