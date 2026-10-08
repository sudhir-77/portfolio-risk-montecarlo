"""Takahashi-Alexander cash flows."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.fund_model import (
    cashflow_metrics,
    draw_growth_shocks,
    internal_rates,
    path_checks,
    run_fund_study,
    simulate_fund,
    summary_frame,
    year5_ratio,
)


def test_deterministic_year_one_and_two_match_hand_arithmetic():
    shocks = np.zeros((1, 12))
    sim = simulate_fund(
        n_paths=1,
        life=12,
        contribution_rates=(0.30,) * 12,
        growth_mean=0.12,
        growth_vol=0.0,
        yield_rate=0.05,
        bow=2.5,
        commitment=1.0,
        shocks=shocks,
    )
    assert sim.calls[0, 0] == pytest.approx(0.30)
    assert sim.dists[0, 0] == pytest.approx(0.015)
    assert sim.nav[0, 0] == pytest.approx(0.285)
    assert sim.calls[0, 1] == pytest.approx(0.21)
    assert sim.nav[0, 1] == pytest.approx(0.50274)
    assert sim.calls_cum[0, -1] <= 1.0 + 1e-12
    assert sim.nav[0, -1] == pytest.approx(0.0, abs=1e-12)


def test_paths_never_call_more_than_committed_and_nav_stays_non_negative():
    sim = simulate_fund(
        n_paths=300,
        growth_mean=0.02,
        growth_vol=0.35,
        contribution_rates=(1.5, 0.8, 0.4),
        seed=7,
    )
    checks = path_checks(sim)
    assert checks["max_calls_over_commitment"] <= 1e-8
    assert checks["min_nav"] >= -1e-12
    assert checks["n_paths"] == 300
    assert np.all(np.diff(sim.calls_cum, axis=1) >= -1e-12)


def test_zero_volatility_growth_equals_the_mean():
    sim = simulate_fund(n_paths=5, growth_mean=0.12, growth_vol=0.0, seed=3)
    # Opening NAV is zero, so year-1 gross NAV is just the call. Later years grow at 12%.
    call = sim.calls[0, 1]
    gross = sim.nav[0, 0] * 1.12 + call
    rate = max(0.05, (2 / 12) ** 2.5)
    assert sim.nav[0, 1] == pytest.approx(gross * (1.0 - rate))


def test_year5_standard_error_matches_the_definition():
    rng = np.random.default_rng(4)
    shocks = rng.standard_normal((500, 12))
    sim = simulate_fund(n_paths=500, shocks=shocks, growth_vol=0.15, growth_mean=0.12)
    summary = summary_frame(sim)
    ratio = year5_ratio(summary)
    sample = sim.nav[:, 4]
    expected_se = float(sample.std(ddof=1) / math.sqrt(500))
    assert ratio["nav_se"] == pytest.approx(expected_se)
    assert ratio["se_over_mean"] == pytest.approx(expected_se / float(sample.mean()))


def test_shared_shocks_change_only_the_growth_mean():
    study = run_fund_study(n_paths=200, seed=11)
    means = study["scenario_year5_nav_mean"]
    assert means["low"] < means["base"] < means["high"]
    assert study["checks"]["n_paths"] == 200
    flows = cashflow_metrics(simulate_fund(n_paths=30, growth_vol=0.0, seed=1))
    assert math.isfinite(flows["irr_of_mean_cash_flows"])
    assert flows["irr_valid_fraction"] == 1.0


def test_irr_matches_a_two_cash_flow_closed_form():
    # -100 at the end of year 1 and +121 at the end of year 2 solves at 21%.
    rate = internal_rates(np.array([[-100.0, 121.0]]))
    assert rate[0] == pytest.approx(0.21, abs=1e-8)


def test_draw_is_reproducible():
    left = draw_growth_shocks(20, 12, seed=42)
    right = draw_growth_shocks(20, 12, seed=42)
    np.testing.assert_array_equal(left, right)
