"""Portfolio weights."""

from __future__ import annotations

import numpy as np
import pytest

from src.portfolio import (
    equal_weight_returns,
    long_only_min_variance_weights,
    rolling_min_variance_returns,
    shrink_covariance,
)


def test_equal_weight_skips_missing_industries():
    matrix = np.array([[0.01, np.nan, 0.03], [0.02, 0.02, 0.02]])
    out = equal_weight_returns(matrix)
    assert out[0] == pytest.approx(0.02)
    assert out[1] == pytest.approx(0.02)


def test_long_only_min_variance_on_a_diagonal_covariance():
    cov = np.diag([1.0, 4.0])
    weights = long_only_min_variance_weights(cov)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= -1e-10)
    assert weights[0] == pytest.approx(0.8, abs=0.02)
    assert weights[1] == pytest.approx(0.2, abs=0.02)
    equal = np.array([0.5, 0.5])
    assert weights @ cov @ weights < equal @ cov @ equal


def test_shrink_covariance_is_symmetric_and_positive():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(40, 5))
    cov = shrink_covariance(window, shrinkage=0.2)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_rolling_min_variance_uses_only_past_returns():
    rng = np.random.default_rng(1)
    matrix = rng.normal(0.0, 0.01, size=(40, 4))
    mask = np.zeros(40, dtype=bool)
    mask[30:] = True
    base = rolling_min_variance_returns(matrix, mask, window=20, rebalance_every=5, shrinkage=0.0)
    shocked = matrix.copy()
    shocked[35:] += 0.5
    other = rolling_min_variance_returns(shocked, mask, window=20, rebalance_every=5, shrinkage=0.0)
    # Day 35's return changes, so that day's portfolio return may change.
    # Days 30 through 34 must not, because their windows and the held weights end earlier.
    np.testing.assert_allclose(base[30:35], other[30:35])
    assert np.isnan(base[:30]).all()
    assert np.isfinite(base[30:]).all()
