"""Kupiec, Christoffersen, and best-method labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    breach_flags,
    christoffersen_independence,
    common_sample,
    evaluate_method,
    kupiec_pof,
    select_best_method,
)


def test_kupiec_matches_hand_computed_likelihood_ratio():
    breaches = np.zeros(1000, dtype=bool)
    breaches[:25] = True
    result = kupiec_pof(breaches, 0.01)
    assert result["n"] == 1000
    assert result["breaches"] == 25
    assert result["rate"] == 0.025
    assert result["statistic"] == pytest.approx(16.042965938670278)
    assert result["p_value"] == pytest.approx(6.192122972318636e-05)


def test_kupiec_is_zero_when_the_rate_matches_the_null():
    breaches = np.zeros(1000, dtype=bool)
    breaches[:10] = True
    result = kupiec_pof(breaches, 0.01)
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_christoffersen_matches_hand_counted_transitions():
    # Transitions: 3 of 00, 2 of 01, 2 of 10, 1 of 11.
    flags = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0], dtype=bool)
    result = christoffersen_independence(flags)
    assert result["n00"] == 3
    assert result["n01"] == 2
    assert result["n10"] == 2
    assert result["n11"] == 1
    assert result["statistic"] == pytest.approx(0.035810130666272144)
    assert result["p_value"] == pytest.approx(0.8499081289235731)


def test_christoffersen_undefined_without_both_states():
    result = christoffersen_independence(np.zeros(10, dtype=bool))
    assert result["p_value"] is None
    assert result["n11"] == 0


def test_breach_is_strict():
    returns = np.array([-0.03, -0.02, -0.01])
    var = np.array([0.02, 0.02, 0.02])
    assert list(breach_flags(returns, var)) == [True, False, False]


def test_select_best_method_ignores_the_normal_baseline():
    evaluations = {
        "normal": {"breach_rate_99": 0.01, "kupiec_p_99": 0.99},
        "hs": {"breach_rate_99": 0.012, "kupiec_p_99": 0.2},
        "fhs": {"breach_rate_99": 0.011, "kupiec_p_99": 0.4},
    }
    assert select_best_method(evaluations) == "fhs"


def test_common_sample_drops_non_finite_var_days():
    frame = pd.DataFrame(
        {
            "portfolio_return": [0.01, 0.02],
            "var95_normal": [0.01, 0.01],
            "var99_normal": [0.02, np.nan],
            "var95_hs": [0.01, 0.01],
            "var99_hs": [0.02, 0.02],
            "var95_fhs": [0.01, 0.01],
            "var99_fhs": [0.02, 0.02],
        }
    )
    kept = common_sample(frame)
    assert len(kept) == 1
    assert kept["portfolio_return"].iloc[0] == 0.01


def test_evaluate_method_counts_breaches():
    returns = np.array([-0.05, 0.01, -0.01, 0.0])
    var99 = np.array([0.02, 0.02, 0.02, 0.02])
    var95 = np.array([0.01, 0.01, 0.01, 0.01])
    stats = evaluate_method(returns, var95, var99)
    assert stats["breach_count_99"] == 1
    assert stats["n"] == 4
    assert stats["breach_rate_99"] == 0.25
