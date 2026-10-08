"""Parametric normal, historical, and GARCH-filtered historical VaR and CVaR.

VaR and CVaR are positive loss numbers. A forecast for index t uses returns
strictly before t.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from arch import arch_model
from scipy.stats import norm

from src.config import GARCH_MAXITER


@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    beta: float
    std_resid: np.ndarray
    sigma2_forecast: float
    converged: bool


def _windows(values: np.ndarray, window: int) -> np.ndarray:
    """Each row is values[t - window : t], and row 0 lines up with index `window`."""
    if window < 2:
        raise ValueError("window must be at least 2")
    series = np.ascontiguousarray(values, dtype=np.float64)
    if len(series) <= window:
        raise ValueError("series is shorter than the estimation window")
    view = np.lib.stride_tricks.sliding_window_view(series, window)
    return view[: len(series) - window]


def _place(length: int, window: int, values: np.ndarray) -> np.ndarray:
    out = np.full(length, np.nan, dtype=np.float64)
    out[window:] = values
    return out


def normal_var_cvar_from_moments(
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-tail normal VaR and CVaR, both as positive losses."""
    z_value = float(norm.ppf(alpha))
    var = -(mean + std * z_value)
    tail_mean = mean - std * float(norm.pdf(z_value)) / alpha
    cvar = -tail_mean
    return var, cvar


def rolling_normal_var_cvar(
    returns: np.ndarray,
    window: int,
    alphas: tuple[float, ...] = (0.05, 0.01),
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    usable = _windows(returns, window)
    mean = usable.mean(axis=1)
    centered = usable - mean[:, None]
    std = np.sqrt((centered * centered).sum(axis=1) / (window - 1))
    length = len(np.asarray(returns))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in alphas:
        var, cvar = normal_var_cvar_from_moments(mean, std, alpha)
        out[alpha] = (_place(length, window, var), _place(length, window, cvar))
    return out


def rolling_historical_var_cvar(
    returns: np.ndarray,
    window: int,
    alphas: tuple[float, ...] = (0.05, 0.01),
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    usable = _windows(returns, window)
    length = len(np.asarray(returns))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in alphas:
        quantile = np.quantile(usable, alpha, axis=1)
        tail = np.where(usable <= quantile[:, None], usable, 0.0)
        counts = (usable <= quantile[:, None]).sum(axis=1)
        tail_mean = tail.sum(axis=1) / np.maximum(counts, 1)
        var = _place(length, window, -quantile)
        cvar = _place(length, window, -tail_mean)
        out[alpha] = (var, cvar)
    return out


def filtered_var_cvar(
    mu_percent: float,
    sigma2_percent: float,
    standardized: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.01),
) -> dict[float, tuple[float, float]]:
    """Scale the empirical residual distribution by the one-step volatility forecast.

    `mu_percent` and `sigma2_percent` are in percent-return units, matching `arch`.
    The returned VaR and CVaR are decimal losses (0.02 means 2%).
    """
    residuals = np.asarray(standardized, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 20:
        raise ValueError("need at least 20 finite standardized residuals")
    sigma = math.sqrt(max(float(sigma2_percent), 1e-12))
    out: dict[float, tuple[float, float]] = {}
    for alpha in alphas:
        quantile = float(np.quantile(residuals, alpha))
        tail = residuals[residuals <= quantile]
        if tail.size == 0:
            tail = residuals[:1]
        var = -(mu_percent + sigma * quantile) / 100.0
        cvar = -(mu_percent + sigma * float(tail.mean())) / 100.0
        out[alpha] = (var, cvar)
    return out


def _usable_fit(fit: GarchFit) -> bool:
    finite_residuals = int(np.isfinite(fit.std_resid).sum())
    persistence = fit.alpha + fit.beta
    return bool(
        math.isfinite(fit.mu)
        and math.isfinite(fit.omega)
        and math.isfinite(fit.sigma2_forecast)
        and fit.sigma2_forecast > 0.0
        and fit.sigma2_forecast < 1e6
        and fit.omega > 0.0
        and fit.alpha >= 0.0
        and fit.beta >= 0.0
        and persistence < 1.05
        and finite_residuals >= 50
    )


def _pack_result(result) -> GarchFit:
    params = result.params
    forecast = result.forecast(horizon=1, reindex=False)
    sigma2 = float(np.asarray(forecast.variance)[-1, -1])
    fit = GarchFit(
        mu=float(params["mu"]),
        omega=float(params["omega"]),
        alpha=float(params["alpha[1]"]),
        beta=float(params["beta[1]"]),
        std_resid=np.asarray(result.std_resid, dtype=np.float64).copy(),
        sigma2_forecast=sigma2,
        converged=int(getattr(result, "convergence_flag", 1)) == 0,
    )
    if not _usable_fit(fit):
        raise RuntimeError("GARCH fit returned unusable parameters")
    return fit


def _fit_once(sample_percent: np.ndarray, starting: np.ndarray | None) -> GarchFit:
    model = arch_model(
        np.asarray(sample_percent, dtype=np.float64),
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )
    kwargs: dict = {
        "disp": "off",
        "show_warning": False,
        "options": {"maxiter": GARCH_MAXITER},
    }
    if starting is not None:
        kwargs["starting_values"] = np.asarray(starting, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Starting values do not satisfy")
        result = model.fit(**kwargs)
    return _pack_result(result)


def fit_garch_arch(sample_percent: np.ndarray, starting: np.ndarray | None = None) -> GarchFit:
    """Fit GARCH(1,1) on percent returns. Retries once without a warm start."""
    try:
        return _fit_once(sample_percent, starting)
    except Exception:
        if starting is None:
            raise
        return _fit_once(sample_percent, None)


def _step_variance(
    sigma2: float,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    returns_percent: np.ndarray,
    start: int,
    stop: int,
) -> float:
    """Propagate the one-step variance from index `start` through `stop - 1`."""
    current = sigma2
    for index in range(start, stop):
        shock = float(returns_percent[index]) - mu
        current = omega + alpha * shock * shock + beta * current
    return current


def rolling_filtered_historical(
    returns_decimal: np.ndarray,
    test_mask: np.ndarray,
    window: int,
    refit_every: int = 1,
    alphas: tuple[float, ...] = (0.05, 0.01),
    fitter=None,
    progress=None,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """GARCH-filtered historical simulation on the test mask.

    Parameters and standardized residuals for a refit at t are estimated on
    returns[t - window : t]. Between refits, variance is updated with those
    frozen parameters and the returns that have already been observed.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    returns_percent = np.ascontiguousarray(returns_decimal, dtype=np.float64) * 100.0
    n_obs = len(returns_percent)
    test_index = np.flatnonzero(np.asarray(test_mask, dtype=bool))
    if test_index.size == 0:
        raise ValueError("no test days")
    if int(test_index[0]) < window:
        raise ValueError("GARCH window reaches before the start of the series")

    placed = {
        alpha: (
            np.full(n_obs, np.nan, dtype=np.float64),
            np.full(n_obs, np.nan, dtype=np.float64),
        )
        for alpha in alphas
    }
    fitter = fit_garch_arch if fitter is None else fitter
    failures = 0
    nonconvergences = 0
    # `fit_day` is the last day whose parameters were estimated. `asof` is the
    # day that `sigma2` forecasts. Keeping them separate is what makes a refit
    # every N days actually refit, instead of sliding `asof` forward forever.
    state: dict | None = None
    starting: np.ndarray | None = None

    def _propagate(to_day: int) -> None:
        assert state is not None
        state["sigma2"] = _step_variance(
            float(state["sigma2"]),
            float(state["mu"]),
            float(state["omega"]),
            float(state["alpha"]),
            float(state["beta"]),
            returns_percent,
            int(state["asof"]),
            int(to_day),
        )
        state["asof"] = int(to_day)

    for step, day in enumerate(test_index):
        if progress is not None and step % 500 == 0:
            progress(step, int(test_index.size))
        need_fit = state is None or (int(day) - int(state["fit_day"])) >= refit_every
        if need_fit:
            sample = returns_percent[day - window : day]
            try:
                fit = fitter(sample, starting)
                if not fit.converged:
                    nonconvergences += 1
                starting = np.array([fit.mu, fit.omega, fit.alpha, fit.beta], dtype=np.float64)
                state = {
                    "fit_day": int(day),
                    "asof": int(day),
                    "mu": fit.mu,
                    "omega": fit.omega,
                    "alpha": fit.alpha,
                    "beta": fit.beta,
                    "z": fit.std_resid,
                    "sigma2": fit.sigma2_forecast,
                }
            except Exception:
                failures += 1
                if state is None:
                    raise
                _propagate(int(day))
        else:
            _propagate(int(day))
        assert state is not None
        scaled = filtered_var_cvar(float(state["mu"]), float(state["sigma2"]), state["z"], alphas)
        for alpha, (var, cvar) in scaled.items():
            placed[alpha][0][day] = var
            placed[alpha][1][day] = cvar

    info = {"fit_failures": failures, "nonconvergences": nonconvergences, "n_forecasts": int(test_index.size)}
    return placed, info
