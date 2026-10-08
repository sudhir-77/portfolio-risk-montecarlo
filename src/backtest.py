"""Kupiec and Christoffersen tests, and the backtest summary."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import chi2

METHODS = ("normal", "hs", "fhs")
PRIMARY_METHODS = ("hs", "fhs")


def _log_term(count: int, probability: float) -> float:
    """count * log(probability), with the conventions 0*log(p) = 0 and log(1) = 0."""
    if count == 0 or probability == 1.0:
        return 0.0
    if probability <= 0.0 or probability > 1.0:
        return -math.inf
    return count * math.log(probability)


def binomial_loglik(n_obs: int, n_fail: int, probability: float) -> float:
    if n_obs < 0 or n_fail < 0 or n_fail > n_obs:
        raise ValueError("invalid binomial counts")
    if probability <= 0.0 or probability >= 1.0:
        if n_fail == 0 and probability == 0.0:
            return 0.0
        if n_fail == n_obs and probability == 1.0:
            return 0.0
        return -math.inf
    return _log_term(n_obs - n_fail, 1.0 - probability) + _log_term(n_fail, probability)


def kupiec_pof(breaches: np.ndarray, probability: float) -> dict[str, float | int | None]:
    """Kupiec proportion-of-failures test. H0: breach rate equals `probability`."""
    flags = np.asarray(breaches, dtype=bool)
    n_obs = int(flags.size)
    n_fail = int(flags.sum())
    if n_obs == 0:
        return {"n": 0, "breaches": 0, "rate": None, "statistic": None, "p_value": None}
    rate = n_fail / n_obs
    loglik_null = binomial_loglik(n_obs, n_fail, probability)
    loglik_alt = binomial_loglik(n_obs, n_fail, rate)
    statistic = -2.0 * (loglik_null - loglik_alt)
    if not math.isfinite(statistic) or statistic < 0.0:
        statistic = 0.0 if rate == probability else statistic
    statistic = max(0.0, float(statistic))
    p_value = float(chi2.sf(statistic, 1))
    return {
        "n": n_obs,
        "breaches": n_fail,
        "rate": rate,
        "statistic": statistic,
        "p_value": p_value,
    }


def christoffersen_independence(breaches: np.ndarray) -> dict[str, float | int | None]:
    """Christoffersen (1998) independence test on the breach indicator.

    The statistic is not defined when a breach state is never observed, because
    one of the transition probabilities has a zero denominator.
    """
    flags = np.asarray(breaches, dtype=bool).astype(np.int64)
    if flags.size < 2:
        return {"statistic": None, "p_value": None, "n00": 0, "n01": 0, "n10": 0, "n11": 0}
    previous = flags[:-1]
    current = flags[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    empty = {
        "statistic": None,
        "p_value": None,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return empty
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    loglik_null = _log_term(n00 + n10, 1.0 - pi) + _log_term(n01 + n11, pi)
    loglik_alt = (
        _log_term(n00, 1.0 - pi01)
        + _log_term(n01, pi01)
        + _log_term(n10, 1.0 - pi11)
        + _log_term(n11, pi11)
    )
    if not math.isfinite(loglik_null) or not math.isfinite(loglik_alt):
        return empty
    statistic = max(0.0, float(-2.0 * (loglik_null - loglik_alt)))
    return {
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def breach_flags(portfolio_return: np.ndarray, var_loss: np.ndarray) -> np.ndarray:
    """A breach is a day whose return is worse than -VaR. VaR is a positive loss."""
    return np.asarray(portfolio_return, dtype=np.float64) < -np.asarray(var_loss, dtype=np.float64)


def evaluate_method(
    portfolio_return: np.ndarray,
    var95: np.ndarray,
    var99: np.ndarray,
) -> dict:
    out: dict = {}
    for level, alpha, series in (
        ("95", 0.05, var95),
        ("99", 0.01, var99),
    ):
        flags = breach_flags(portfolio_return, series)
        kupiec = kupiec_pof(flags, alpha)
        independence = christoffersen_independence(flags)
        out[f"breach_count_{level}"] = kupiec["breaches"]
        out[f"breach_rate_{level}"] = kupiec["rate"]
        out[f"kupiec_stat_{level}"] = kupiec["statistic"]
        out[f"kupiec_p_{level}"] = kupiec["p_value"]
        out[f"christoffersen_stat_{level}"] = independence["statistic"]
        out[f"christoffersen_p_{level}"] = independence["p_value"]
        out[f"mean_var_{level}"] = float(np.mean(series))
    out["n"] = int(len(portfolio_return))
    return out


def select_best_method(evaluations: dict[str, dict]) -> str:
    """Pick historical or filtered historical simulation.

    The score is distance of the 99% breach rate from the nominal 1% rate.
    Kupiec's p-value breaks ties. Parametric normal is the baseline and is
    not eligible. The choice is a label on the finished backtest; it does not
    change any forecast.
    """

    def score(name: str) -> tuple[float, float]:
        rate = float(evaluations[name]["breach_rate_99"])
        p_value = evaluations[name]["kupiec_p_99"]
        p_rank = -(float(p_value) if p_value is not None else -1.0)
        return (abs(rate - 0.01), p_rank)

    return min(PRIMARY_METHODS, key=score)


def common_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep days where the return and every VaR (and CVaR, if present) are finite."""
    columns = [
        "portfolio_return",
        "var95_normal",
        "var99_normal",
        "var95_hs",
        "var99_hs",
        "var95_fhs",
        "var99_fhs",
    ]
    columns.extend(column for column in frame.columns if column.startswith("cvar"))
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        mask &= np.isfinite(frame[column].to_numpy(dtype=np.float64))
    return frame.loc[mask].reset_index(drop=True)


def evaluate_forecast_frame(frame: pd.DataFrame) -> dict[str, dict]:
    returns = frame["portfolio_return"].to_numpy(dtype=np.float64)
    evaluations = {}
    for method in METHODS:
        evaluations[method] = evaluate_method(
            returns,
            frame[f"var95_{method}"].to_numpy(dtype=np.float64),
            frame[f"var99_{method}"].to_numpy(dtype=np.float64),
        )
    return evaluations
