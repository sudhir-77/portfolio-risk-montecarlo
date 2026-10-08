"""Return diagnostics that do not choose VaR settings."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


def return_diagnostics(
    train_returns: np.ndarray,
    equal_weight: np.ndarray | None = None,
    min_variance: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Ljung-Box and Jarque-Bera on history, plus an OLS beta when both series exist.

    The beta is a description of the minimum-variance book. It is not used to
    pick a VaR model or a threshold.
    """
    history = np.asarray(train_returns, dtype=np.float64)
    history = history[np.isfinite(history)]
    if history.size < 30:
        raise ValueError("need at least 30 historical returns for diagnostics")
    ljung = acorr_ljungbox(history**2, lags=[10], return_df=True)
    jarque_stat, jarque_p, skew, kurtosis = jarque_bera(history)
    out: dict[str, float | int | None] = {
        "train_n": int(history.size),
        "ljung_box_squared_lag10_p": float(ljung["lb_pvalue"].iloc[-1]),
        "jarque_bera_stat": float(jarque_stat),
        "jarque_bera_p": float(jarque_p),
        "train_skew": float(skew),
        "train_kurtosis": float(kurtosis),
        "minvar_beta_on_equal_weight": None,
        "minvar_r_squared": None,
        "minvar_regression_n": 0,
    }
    if equal_weight is None or min_variance is None:
        return out
    left = np.asarray(equal_weight, dtype=np.float64)
    right = np.asarray(min_variance, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 30:
        return out
    design = sm.add_constant(left[mask])
    fitted = sm.OLS(right[mask], design).fit()
    params = np.asarray(fitted.params, dtype=np.float64)
    out["minvar_beta_on_equal_weight"] = float(params[1])
    out["minvar_r_squared"] = float(fitted.rsquared)
    out["minvar_regression_n"] = int(mask.sum())
    return out
