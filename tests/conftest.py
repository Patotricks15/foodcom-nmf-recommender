"""Shared fixtures.

The fixture is generated with real timestamps and real popularity drift, so
the point-in-time machinery is exercised rather than merely instantiated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from download_data import generate_synthetic  # noqa: E402

from recommender.data import (  # noqa: E402
    build_implicit_feedback,
    leave_last_one_out,
    temporal_split,
)
from recommender.features import build_features  # noqa: E402


@pytest.fixture(scope="session")
def raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small synthetic Food.com-shaped dataset."""
    return generate_synthetic(n_users=220, n_items=110, seed=7)


@pytest.fixture(scope="session")
def interactions(raw_data):
    reviews, _ = raw_data
    return build_implicit_feedback(reviews)


@pytest.fixture(scope="session")
def catalogue(raw_data) -> pd.DataFrame:
    _, recipes = raw_data
    return recipes


@pytest.fixture(scope="session")
def split(interactions):
    return temporal_split(interactions)


@pytest.fixture(scope="session")
def loo_split(interactions):
    return leave_last_one_out(interactions)


@pytest.fixture(scope="session")
def store(split, interactions, catalogue):
    return build_features(split, interactions, catalogue)
