"""Public evaluation API."""

from __future__ import annotations

from recommender.evaluation.metrics import (
    EvaluationReport,
    FilterImpactReport,
    evaluate,
    evaluate_nutrition_filter,
)

__all__ = [
    "EvaluationReport",
    "FilterImpactReport",
    "evaluate",
    "evaluate_nutrition_filter",
]
