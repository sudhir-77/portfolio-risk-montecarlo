"""Equal-weight and long-only minimum-variance portfolios."""

from __future__ import annotations

import numpy as np

from src.config import COV_SHRINKAGE


def equal_weight_returns(industry_matrix: np.ndarray) -> np.ndarray:
    """Cross-sectional mean. NaN industries are skipped on that day."""
    matrix = np.asarray(industry_matrix, dtype=np.float64)
    with np.errstate(all="ignore"):
        out = np.nanmean(matrix, axis=1)
    return out


def shrink_covariance(window: np.ndarray, shrinkage: float = COV_SHRINKAGE) -> np.ndarray:
    """Shrink the sample covariance toward a scaled identity. `shrinkage` is fixed."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")
    sample = np.cov(window, rowvar=False, ddof=1)
    sample = np.atleast_2d(np.asarray(sample, dtype=np.float64))
    width = sample.shape[0]
    target = float(np.trace(sample) / width)
    cov = (1.0 - shrinkage) * sample + shrinkage * target * np.eye(width)
    cov = 0.5 * (cov + cov.T)
    cov = cov + 1e-12 * np.eye(width)
    return cov


def long_only_min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights.

    On each pass, solve the unconstrained problem on the free assets and drop
    the most negative weight. That is the active-set step for this quadratic
    program. A scaled SLSQP start at equal weights was not moving, because the
    daily covariance is tiny relative to the solver tolerance.
    """
    cov = np.asarray(cov, dtype=np.float64)
    n_assets = cov.shape[0]
    free = np.ones(n_assets, dtype=bool)
    for _ in range(n_assets):
        chosen = np.flatnonzero(free)
        block = cov[np.ix_(chosen, chosen)]
        try:
            raw = np.linalg.solve(block, np.ones(chosen.size))
        except np.linalg.LinAlgError:
            break
        if float(np.min(raw)) >= -1e-10:
            raw = np.clip(raw, 0.0, None)
            total = float(raw.sum())
            if total <= 0.0:
                break
            weights = np.zeros(n_assets, dtype=np.float64)
            weights[chosen] = raw / total
            return weights
        free[chosen[int(np.argmin(raw))]] = False
    return np.full(n_assets, 1.0 / n_assets, dtype=np.float64)


def rolling_min_variance_returns(
    industry_matrix: np.ndarray,
    test_mask: np.ndarray,
    window: int,
    rebalance_every: int,
    shrinkage: float = COV_SHRINKAGE,
) -> np.ndarray:
    """Portfolio returns on test days. Weights use only rows before that day.

    Weights are recomputed every `rebalance_every` test-index step and held in between.
    """
    matrix = np.asarray(industry_matrix, dtype=np.float64)
    n_days, n_assets = matrix.shape
    out = np.full(n_days, np.nan)
    weights: np.ndarray | None = None
    last_rebalance = -10**9
    for day in np.flatnonzero(np.asarray(test_mask, dtype=bool)):
        if day < window:
            raise ValueError("minimum-variance window reaches before the sample")
        if weights is None or (day - last_rebalance) >= rebalance_every:
            history = matrix[day - window : day]
            column_ok = np.isfinite(history).all(axis=0)
            if int(column_ok.sum()) < 5:
                weights = np.full(n_assets, 1.0 / n_assets)
            else:
                cov = shrink_covariance(history[:, column_ok], shrinkage=shrinkage)
                sub_weights = long_only_min_variance_weights(cov)
                weights = np.zeros(n_assets, dtype=np.float64)
                weights[column_ok] = sub_weights
            last_rebalance = day
        assert weights is not None
        current = matrix[day]
        active = np.isfinite(current) & (weights > 0.0)
        if not np.any(active):
            continue
        held = np.zeros(n_assets, dtype=np.float64)
        held[active] = weights[active]
        held /= held.sum()
        out[day] = float(held @ np.nan_to_num(current, nan=0.0))
    return out
