"""Stress windows and the hypothetical shock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stress import cumulative_return, hypothetical_portfolio_return, stress_tables


def test_cumulative_return_compounds():
    assert cumulative_return(np.array([0.10, -0.50, 0.20])) == pytest.approx(-0.34)


def test_hypothetical_shock_ignores_the_backtest_period():
    history = np.tile(np.array([-0.02, -0.01, 0.0, 0.01]), (60, 1))
    future = np.array([[-0.90, -0.90, -0.90, -0.90]])
    matrix = np.vstack([history, future])
    dates = list(pd.bdate_range("2004-01-02", periods=60)) + [pd.Timestamp("2005-06-01")]
    shock = hypothetical_portfolio_return(np.asarray(matrix, dtype=float), pd.Series(dates), cutoff="2004-12-31")
    assert shock == pytest.approx(-0.005)


def test_stress_table_reports_each_named_window():
    history_dates = list(pd.bdate_range("2004-01-02", periods=80))
    stress_dates = list(
        pd.to_datetime(
            [
                "2008-09-02",
                "2008-09-03",
                "2008-10-01",
                "2009-03-31",
                "2020-02-18",
                "2020-03-20",
                "2022-01-04",
                "2022-06-01",
            ]
        )
    )
    dates = pd.Series(history_dates + stress_dates)
    stress_returns = np.array([-0.01, -0.02, 0.00, 0.01, -0.03, -0.04, -0.01, 0.02])
    equal = np.concatenate([np.zeros(len(history_dates)), stress_returns])
    industry = np.zeros((len(dates), 4))
    industry[: len(history_dates)] = np.array([-0.02, -0.01, 0.0, 0.01])
    forecasts = pd.DataFrame(
        {
            "date": ["2008-09-02", "2020-02-18", "2022-01-04"],
            "portfolio_return": [-0.01, -0.03, -0.01],
            "var99_normal": [0.005, 0.02, 0.02],
            "var99_hs": [0.02, 0.02, 0.02],
            "var99_fhs": [0.02, 0.02, 0.02],
        }
    )
    table, hypothetical = stress_tables(dates, equal, None, forecasts, industry)
    assert set(table["window"]) == {"gfc_2008", "covid_2020", "rates_2022"}
    gfc = table.loc[table["window"] == "gfc_2008"].iloc[0]
    assert gfc["cumulative_return"] == pytest.approx(cumulative_return(stress_returns[:4]))
    assert gfc["worst_day"] == "2008-09-03"
    assert gfc["breach_count_99_normal"] == 1
    assert gfc["breach_count_99_fhs"] == 0
    assert hypothetical["loss"] == pytest.approx(-hypothetical["portfolio_return"])
    assert hypothetical["portfolio_return"] == pytest.approx(-0.005)
