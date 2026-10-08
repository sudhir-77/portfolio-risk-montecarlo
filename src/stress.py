"""Historical stress windows and one hypothetical industry shock."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import STRESS_WINDOWS, TRAIN_END


def cumulative_return(returns: np.ndarray) -> float:
    series = np.asarray(returns, dtype=np.float64)
    if series.size == 0 or not np.isfinite(series).all():
        raise ValueError("cumulative return needs a non-empty finite series")
    return float(np.prod(1.0 + series) - 1.0)


def hypothetical_portfolio_return(
    industry_matrix: np.ndarray,
    dates: pd.Series,
    cutoff: str = TRAIN_END,
) -> float:
    """Equal-weight of each industry's pre-cutoff 1st percentile.

    The shock is built only from dates before the backtest. It is a same-day
    crash in which every industry prints that historical percentile together.
    """
    matrix = np.asarray(industry_matrix, dtype=np.float64)
    cutoff_ts = pd.Timestamp(cutoff)
    history = matrix[pd.to_datetime(dates).to_numpy() < cutoff_ts.to_datetime64()]
    if history.shape[0] < 50:
        raise ValueError("not enough pre-cutoff history for the hypothetical shock")
    percentiles = np.nanpercentile(history, 1, axis=0)
    return float(np.nanmean(percentiles))


def _slice_dates(dates: pd.Series, start: str, end: str) -> np.ndarray:
    stamps = pd.to_datetime(dates)
    return ((stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))).to_numpy()


def _window_var_stats(forecasts: pd.DataFrame, start: str, end: str, worst_day: pd.Timestamp) -> dict:
    stamps = pd.to_datetime(forecasts["date"])
    mask = (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    window = forecasts.loc[mask]
    stats: dict[str, float | int | None] = {}
    for method in ("normal", "hs", "fhs"):
        column = f"var99_{method}"
        if window.empty or column not in window:
            stats[f"breach_count_99_{method}"] = None
            stats[f"var99_{method}_on_worst_day"] = None
            continue
        flags = window["portfolio_return"].to_numpy() < -window[column].to_numpy()
        stats[f"breach_count_99_{method}"] = int(np.sum(flags))
        on_day = window.loc[pd.to_datetime(window["date"]) == worst_day, column]
        stats[f"var99_{method}_on_worst_day"] = float(on_day.iloc[0]) if len(on_day) else None
    return stats


def stress_tables(
    dates: pd.Series,
    equal_weight: np.ndarray,
    min_variance: np.ndarray | None,
    forecasts: pd.DataFrame,
    industry_matrix: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for name, start, end in STRESS_WINDOWS:
        mask = _slice_dates(dates, start, end)
        returns = np.asarray(equal_weight, dtype=np.float64)[mask]
        if returns.size == 0:
            raise ValueError(f"stress window {name} has no returns")
        wealth = cumulative_return(returns)
        worst_pos = int(np.argmin(returns))
        worst_day = pd.Timestamp(pd.to_datetime(dates).to_numpy()[mask][worst_pos])
        row = {
            "window": name,
            "portfolio": "equal_weight",
            "start": start,
            "end": end,
            "n_days": int(returns.size),
            "cumulative_return": wealth,
            "worst_day": worst_day.strftime("%Y-%m-%d"),
            "worst_day_return": float(returns[worst_pos]),
        }
        row.update(_window_var_stats(forecasts, start, end, worst_day))
        rows.append(row)

        if min_variance is not None:
            mv = np.asarray(min_variance, dtype=np.float64)[mask]
            finite = np.isfinite(mv)
            if int(finite.sum()) >= max(5, int(0.8 * mv.size)):
                mv_returns = mv[finite]
                mv_worst = int(np.argmin(mv_returns))
                rows.append(
                    {
                        "window": name,
                        "portfolio": "min_variance",
                        "start": start,
                        "end": end,
                        "n_days": int(mv_returns.size),
                        "cumulative_return": cumulative_return(mv_returns),
                        "worst_day": pd.Timestamp(pd.to_datetime(dates).to_numpy()[mask][finite][mv_worst]).strftime(
                            "%Y-%m-%d"
                        ),
                        "worst_day_return": float(mv_returns[mv_worst]),
                        "breach_count_99_normal": None,
                        "var99_normal_on_worst_day": None,
                        "breach_count_99_hs": None,
                        "var99_hs_on_worst_day": None,
                        "breach_count_99_fhs": None,
                        "var99_fhs_on_worst_day": None,
                    }
                )

    shock = hypothetical_portfolio_return(industry_matrix, dates)
    median_var = {}
    for method in ("normal", "hs", "fhs"):
        column = f"var99_{method}"
        median_var[method] = float(np.median(forecasts[column].to_numpy(dtype=np.float64)))
    hypothetical = {
        "name": "synchronous_industry_1st_percentile",
        "history_cutoff": TRAIN_END,
        "portfolio_return": shock,
        "loss": float(-shock),
        "median_var99": median_var,
    }
    return pd.DataFrame(rows), hypothetical
