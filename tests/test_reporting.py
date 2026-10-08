"""Diagnostics, documents, and a short end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import GARCH_WINDOW, N_PATHS, NORMAL_WINDOW, TEST_END, TEST_START
from src.diagnostics import return_diagnostics
from src.pipeline import main
from src.reporting import (
    BANNED_RESUME_SNIPPETS,
    attach_coverage,
    build_metrics,
    render_report,
    render_resume_notes,
    update_readme_table,
)


def test_locked_methodology_constants():
    assert NORMAL_WINDOW == 252
    assert GARCH_WINDOW == 1000
    assert N_PATHS == 10_000
    assert TEST_START == "2005-01-01"
    assert TEST_END == "2024-12-31"


def test_diagnostics_ljung_box_and_beta():
    rng = np.random.default_rng(0)
    history = rng.normal(0.0004, 0.01, 400)
    equal = rng.normal(0.0004, 0.01, 200)
    stats = return_diagnostics(history, equal, 0.5 * equal + rng.normal(0, 0.001, 200))
    assert stats["train_n"] == 400
    assert 0.0 <= stats["ljung_box_squared_lag10_p"] <= 1.0
    assert 0.0 <= stats["jarque_bera_p"] <= 1.0
    assert stats["minvar_beta_on_equal_weight"] == pytest.approx(0.5, abs=0.15)


def _evaluations():
    def one(rate, p_value, breaches):
        return {
            "n": 1000,
            "breach_count_95": 50,
            "breach_rate_95": 0.05,
            "kupiec_stat_95": 0.0,
            "kupiec_p_95": 0.5,
            "christoffersen_stat_95": 0.1,
            "christoffersen_p_95": 0.4,
            "mean_var_95": 0.015,
            "breach_count_99": breaches,
            "breach_rate_99": rate,
            "kupiec_stat_99": 1.0,
            "kupiec_p_99": p_value,
            "christoffersen_stat_99": 0.2,
            "christoffersen_p_99": 0.3,
            "mean_var_99": 0.025,
        }

    return {
        "normal": one(0.02, 0.01, 20),
        "hs": one(0.012, 0.2, 12),
        "fhs": one(0.010, 0.8, 10),
    }


def _metrics():
    fund = {
        "year5": {"nav_mean": 0.5, "nav_se": 0.001, "se_over_mean": 0.002},
        "checks": {"n_paths": 10000, "max_calls_over_commitment": 0.0, "min_nav": 0.0},
        "scenario_year5_nav_mean": {"low": 0.4, "base": 0.5, "high": 0.7},
        "cashflows": {"irr_mean": 0.11, "irr_median": 0.10, "irr_of_mean_cash_flows": 0.105, "npv_of_mean_cash_flows": 0.02},
    }
    stress_rows = [
        {
            "window": name,
            "portfolio": "equal_weight",
            "cumulative_return": value,
            "worst_day": "2008-10-15",
            "worst_day_return": -0.08,
        }
        for name, value in (("gfc_2008", -0.4), ("covid_2020", -0.25), ("rates_2022", -0.12))
    ]
    return build_metrics(
        mode="full",
        evaluations=_evaluations(),
        best_method="fhs",
        baseline_window=252,
        fund=fund,
        diagnostics={"minvar_beta_on_equal_weight": 0.72, "ljung_box_squared_lag10_p": 0.001, "jarque_bera_p": 0.0},
        stress_rows=stress_rows,
        hypothetical={"portfolio_return": -0.03, "loss": 0.03, "median_var99": {"normal": 0.02, "hs": 0.02, "fhs": 0.02}},
        data_manifest={"sha256": "abc", "n_rows": 25000, "n_industries": 49},
        n_train=10000,
        n_test=5000,
        garch_info={"fit_failures": 0, "nonconvergences": 0, "n_forecasts": 5000},
        seed=42,
    )


def test_resume_notes_use_metrics_and_plain_language(tmp_path: Path):
    metrics = _metrics()
    metrics["runtime_seconds"] = 12.5
    metrics["hardware"]["peak_rss_gb"] = 0.4
    notes = render_resume_notes(metrics)
    report = render_report(metrics)
    lowered = notes.lower()
    for snippet in BANNED_RESUME_SNIPPETS:
        assert snippet.lower() not in lowered
    assert "0.8" in notes or "0.80" in notes
    assert "10000" in notes
    blob = json.dumps(metrics)
    assert "10000" in blob
    assert "GARCH-filtered historical simulation" in notes
    assert "failed" not in report.lower() or "Some success" not in report
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS_TABLE_START -->\nold\n<!-- RESULTS_TABLE_END -->\n", encoding="utf-8")
    update_readme_table(readme, metrics)
    text = readme.read_text(encoding="utf-8")
    assert "intro" in text
    assert "Kupiec POF" in text
    assert "old" not in text


def test_attach_coverage_updates_metrics(tmp_path: Path):
    metrics = _metrics()
    metrics_path = tmp_path / "metrics.json"
    coverage_path = tmp_path / "coverage.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    coverage_path.write_text(json.dumps({"totals": {"percent_covered": 88.0}}), encoding="utf-8")
    row = attach_coverage(metrics_path, coverage_path)
    assert row is not None and row["passed"]
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["metrics"]["test_coverage"] == pytest.approx(0.88)


def test_help_and_cuda_flag():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit):
        main(["--device", "cuda"])


def test_quick_end_to_end(tmp_path: Path):
    code = main(
        [
            "--quick",
            "--outdir",
            str(tmp_path / "results"),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )
    assert code == 0
    out = tmp_path / "results"
    forecasts = __import__("pandas").read_csv(out / "var_forecasts.csv")
    expected = [
        "row_id",
        "date",
        "portfolio_return",
        "var95_normal",
        "var99_normal",
        "var95_hs",
        "var99_hs",
        "var95_fhs",
        "var99_fhs",
    ]
    assert list(forecasts.columns) == expected
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for method in ("normal", "hs", "fhs"):
        rate = float(np.mean(forecasts["portfolio_return"] < -forecasts[f"var99_{method}"]))
        assert rate == pytest.approx(metrics["metrics"]["backtest"][method]["breach_rate_99"])
    summary = __import__("pandas").read_csv(out / "fund_paths_summary.csv")
    assert list(summary.columns) == [
        "year",
        "nav_mean",
        "nav_se",
        "calls_cum_mean",
        "dist_cum_mean",
        "commitment",
    ]
    checks = json.loads((out / "fund_path_checks.json").read_text(encoding="utf-8"))
    assert set(checks) == {"max_calls_over_commitment", "min_nav", "n_paths"}
    assert checks["max_calls_over_commitment"] <= 1e-8
    assert checks["min_nav"] >= -1e-9
    assert checks["n_paths"] == 400
    year5 = summary.loc[summary["year"] == 5].iloc[0]
    assert metrics["metrics"]["fund"]["year5_se_over_mean"] == pytest.approx(year5["nav_se"] / year5["nav_mean"])
    nav = __import__("pandas").read_csv(out / "fund_year5_nav.csv")
    se = float(nav["nav"].std(ddof=1) / np.sqrt(len(nav)))
    assert se == pytest.approx(float(year5["nav_se"]), rel=1e-6, abs=1e-8)
    assert not (tmp_path / "data" / "splits.json").exists()
    assert (out / "breach_rates.png").exists()
    assert metrics["splits"]["french_49_daily"]["val"] == 0
    assert metrics["hardware"]["peak_rss_gb"] > 0
    assert metrics["runtime_seconds"] > 0
