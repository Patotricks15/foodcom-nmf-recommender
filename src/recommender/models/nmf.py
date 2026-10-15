"""Classical Non-negative Matrix Factorization via multiplicative updates.

Non-negative Matrix Factorization approximates the (non-negative) user-recipe
interaction matrix ``X`` as the product of two non-negative factor matrices,

    X ~= W H,  W >= 0,  H >= 0,

minimizing the squared Frobenius reconstruction error

    min_{W,H >= 0} || X - W H ||_F^2.

Unlike PureSVD (signed factors) or BPR-style embeddings trained by gradient
descent, every entry of ``W`` and ``H`` stays non-negative throughout
optimization. That makes each latent component additive and legible: a
column of ``H`` reads as a non-negative "flavor/ingredient profile" over
recipes, rather than a direction that can cancel out other directions. This
module is a from-scratch NumPy implementation of Lee and Seung's
multiplicative update rules; it is linear algebra on dense arrays, not a
neural network, and uses no automatic differentiation or gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


@dataclass
class NMFRecommender:
    """Multiplicative-update NMF over the binary training interaction matrix.

    Attributes:
        n_factors: Rank of the factorization.
        max_iter: Maximum number of multiplicative-update sweeps.
        tol: Relative Frobenius-error improvement below which iteration stops.
        random_state: Seed for the non-negative factor initialization.
        eps: Additive stabilizer preventing division by zero in the updates.
    """

    n_factors: int = 16
    max_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42
    eps: float = 1e-10

    W: np.ndarray | None = None
    H: np.ndarray | None = None
    popularity: np.ndarray | None = None
    reconstruction_error_: float | None = None
    n_iter_: int = 0
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> NMFRecommender:
        """Fit ``W, H >= 0`` on the binary training matrix.

        Each sweep applies, in order,

            H <- H * (W^T X) / (W^T W H),
            W <- W * (X H^T) / (W H H^T),

        which is the classical Lee-Seung rule for the squared-error
        objective. Every entry stays non-negative because both the current
        factors and the multiplicative ratio are non-negative (the ``eps``
        term only guards the denominator against exact zeros); the objective
        is non-increasing at every sweep by construction of the rule.

        Args:
            split: Leakage-safe partition supplying the training matrix.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``n_factors`` is not positive.
        """
        if self.n_factors < 1:
            raise ValueError(f"n_factors must be positive, got {self.n_factors}.")

        users, items = split.train_matrix_entries()
        x = np.zeros((split.n_users, split.n_items), dtype=np.float64)
        x[users, items] = 1.0

        rng = np.random.default_rng(self.random_state)
        # Uniform(0, 1) initialization: strictly positive, so the first
        # multiplicative update is always well defined.
        w = rng.uniform(1e-3, 1.0, size=(split.n_users, self.n_factors))
        h = rng.uniform(1e-3, 1.0, size=(self.n_factors, split.n_items))

        previous_error = None
        error = float(np.linalg.norm(x - w @ h, ord="fro"))
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):  # noqa: B007
            h *= (w.T @ x) / (w.T @ w @ h + self.eps)
            w *= (x @ h.T) / (w @ h @ h.T + self.eps)

            previous_error = error
            error = float(np.linalg.norm(x - w @ h, ord="fro"))
            if previous_error > 0 and abs(previous_error - error) / previous_error < self.tol:
                break

        self.W = w
        self.H = h
        self.reconstruction_error_ = error
        self.n_iter_ = n_iter
        self.popularity = x.sum(axis=0)
        self.n_users = split.n_users
        self.n_items = split.n_items
        return self

    def _check_fitted(self) -> None:
        if self.W is None or self.H is None or self.popularity is None:
            raise RuntimeError("Model is not fitted; call fit() first.")

    def score_user(self, user_index: int) -> np.ndarray:
        """Score every recipe, with popularity as the cold-user fallback."""
        self._check_fitted()
        assert self.W is not None
        assert self.H is not None
        assert self.popularity is not None
        if not 0 <= user_index < self.n_users:
            return self.popularity.copy()
        return self.W[user_index] @ self.H

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        """Return top-scoring unseen recipe indices."""
        if k < 1:
            raise ValueError("k must be positive.")
        scores = self.score_user(user_index).copy()
        if exclude:
            scores[list(exclude)] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        if k == 0:
            return np.array([], dtype=np.int64)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top], kind="stable")]
