"""VaR engines on small series with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from src.var_engine import (
    GarchFit,
    filtered_var_cvar,
    fit_garch_arch,
    normal_var_cvar_from_moments,
    rolling_filtered_historical,
    rolling_historical_var_cvar,
    rolling_normal_var_cvar,
)


def test_normal_moments_match_the_mills_ratio():
    var, cvar = normal_var_cvar_from_moments(np.array([0.0]), np.array([1.0]), 0.05)
    assert var[0] == pytest.approx(1.6448536269514729)
    assert cvar[0] == pytest.approx(2.0627128075074253)
    assert cvar[0] > var[0]


def test_rolling_normal_var_matches_a_hand_window_and_ignores_the_future():
    history = np.array([0.01, -0.02, 0.015, -0.005, 0.0])
    future_a = np.concatenate([history, [0.5]])
    future_b = np.concatenate([history, [-0.5]])
    var_a = rolling_normal_var_cvar(future_a, window=5, alphas=(0.01,))[0.01][0]
    var_b = rolling_normal_var_cvar(future_b, window=5, alphas=(0.01,))[0.01][0]
    assert np.isnan(var_a[:5]).all()
    assert var_a[5] == pytest.approx(0.03185483018040888, rel=1e-10)
    np.testing.assert_allclose(var_a, var_b)


def test_rolling_historical_var_and_cvar_match_sorted_order_statistics():
    history = np.array([0.05, -0.01, 0.02, -0.04, 0.00, 0.03, -0.02, 0.01])
    series = np.concatenate([history, [-1.0]])
    var, cvar = rolling_historical_var_cvar(series, window=8, alphas=(0.25,))[0.25]
    assert var[8] == pytest.approx(0.0125)
    assert cvar[8] == pytest.approx(0.03)
    assert cvar[8] >= var[8]
    shocked = series.copy()
    shocked[-1] = 1.0
    var_shocked = rolling_historical_var_cvar(shocked, window=8, alphas=(0.25,))[0.25][0]
    np.testing.assert_allclose(var, var_shocked)


def test_filtered_var_scales_with_volatility():
    residuals = np.array([-2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5] * 5, dtype=float)
    low = filtered_var_cvar(0.0, sigma2_percent=1.0, standardized=residuals, alphas=(0.05,))
    high = filtered_var_cvar(0.0, sigma2_percent=4.0, standardized=residuals, alphas=(0.05,))
    assert low[0.05][0] > 0.0
    assert high[0.05][0] == pytest.approx(2.0 * low[0.05][0])
    assert low[0.05][1] >= low[0.05][0]


def _constant_fit(sample, starting):
    return GarchFit(
        mu=0.0,
        omega=0.05,
        alpha=0.05,
        beta=0.90,
        std_resid=np.linspace(-2.5, 2.5, 80),
        sigma2_forecast=float(np.var(sample) + 1.0),
        converged=True,
    )


def test_filtered_historical_does_not_use_the_forecast_day_or_later_days():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, 80)
    shocked = base.copy()
    shocked[60:] += 0.25
    mask = np.zeros(80, dtype=bool)
    mask[40:] = True
    left, _ = rolling_filtered_historical(base, mask, window=20, refit_every=5, fitter=_constant_fit)
    right, _ = rolling_filtered_historical(shocked, mask, window=20, refit_every=5, fitter=_constant_fit)
    np.testing.assert_allclose(left[0.01][0][40:60], right[0.01][0][40:60])
    assert np.isnan(left[0.01][0][:40]).all()


def test_refit_schedule_and_failure_fallback():
    calls = {"n": 0}

    def flaky(sample, starting):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("optimizer failed")
        return _constant_fit(sample, starting)

    returns = np.full(40, 0.001)
    mask = np.zeros(40, dtype=bool)
    mask[25:] = True
    out, info = rolling_filtered_historical(returns, mask, window=20, refit_every=1, fitter=flaky)
    assert info["fit_failures"] == 1
    assert np.isfinite(out[0.05][0][25:]).all()
    assert np.isfinite(out[0.01][1][25:]).all()


def test_garch_forecast_matches_the_recursion_and_arch():
    from arch import arch_model

    rng = np.random.default_rng(1)
    sample = rng.normal(0.02, 1.0, 300)
    model = arch_model(sample, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    result = model.fit(disp="off", show_warning=False, options={"maxiter": 100})
    variance = np.asarray(result.conditional_volatility, dtype=float) ** 2
    mu = float(result.params["mu"])
    omega = float(result.params["omega"])
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])
    for index in range(1, len(sample)):
        manual = omega + alpha * (sample[index - 1] - mu) ** 2 + beta * variance[index - 1]
        assert manual == pytest.approx(variance[index], rel=1e-5, abs=1e-6)
    forecast = float(np.asarray(result.forecast(horizon=1, reindex=False).variance)[-1, -1])
    manual_forecast = omega + alpha * (sample[-1] - mu) ** 2 + beta * variance[-1]
    assert manual_forecast == pytest.approx(forecast, rel=1e-5, abs=1e-6)
    packed = fit_garch_arch(sample)
    assert packed.sigma2_forecast == pytest.approx(forecast, rel=1e-5, abs=1e-5)
    assert packed.std_resid.shape == sample.shape


def test_fit_retries_once_without_a_warm_start(monkeypatch):
    calls = []

    def fake(sample, starting):
        calls.append(starting is not None)
        if starting is not None:
            raise RuntimeError("bad start")
        return _constant_fit(sample, None)

    monkeypatch.setattr("src.var_engine._fit_once", fake)
    fit = fit_garch_arch(np.ones(40), np.array([0.0, 0.1, 0.05, 0.9]))
    assert calls == [True, False]
    assert fit.sigma2_forecast > 0


def test_real_garch_roller_is_finite():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0004, 0.01, 140)
    mask = np.zeros(140, dtype=bool)
    mask[80:] = True
    out, info = rolling_filtered_historical(returns, mask, window=60, refit_every=30)
    assert info["n_forecasts"] == 60
    assert np.isfinite(out[0.01][0][80:]).all()
    assert np.median(out[0.01][0][80:]) > 0.0
