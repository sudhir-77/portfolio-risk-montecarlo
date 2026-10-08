"""Figures written next to the numeric results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_breach_rates(evaluations: dict, path: Path) -> None:
    methods = ["normal", "hs", "fhs"]
    labels = ["Normal", "Historical", "GARCH-FHS"]
    x_pos = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates_95 = [evaluations[name]["breach_rate_95"] * 100 for name in methods]
    rates_99 = [evaluations[name]["breach_rate_99"] * 100 for name in methods]
    ax.bar(x_pos - width / 2, rates_95, width, label="95% VaR")
    ax.bar(x_pos + width / 2, rates_99, width, label="99% VaR")
    ax.axhline(5.0, color="gray", linestyle="--", linewidth=1, label="5% nominal")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1% nominal")
    ax.set_xticks(x_pos, labels)
    ax.set_ylabel("Breach rate (%)")
    ax.set_title("Out-of-sample VaR breach rates")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_cumulative_breaches(forecasts: pd.DataFrame, path: Path) -> None:
    dates = pd.to_datetime(forecasts["date"])
    returns = forecasts["portfolio_return"].to_numpy(dtype=np.float64)
    expected = 0.01 * np.arange(1, len(forecasts) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, label in (
        ("var99_normal", "Normal"),
        ("var99_hs", "Historical"),
        ("var99_fhs", "GARCH-FHS"),
    ):
        hits = np.cumsum(returns < -forecasts[column].to_numpy(dtype=np.float64))
        ax.plot(dates, hits, label=label)
    ax.plot(dates, expected, color="black", linestyle="--", label="Expected at 1%")
    ax.set_title("Cumulative 99% VaR breaches")
    ax.set_ylabel("Breaches")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_var_slice(forecasts: pd.DataFrame, path: Path) -> None:
    frame = forecasts.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    window = frame[(frame["date"] >= "2020-02-01") & (frame["date"] <= "2020-04-30")]
    if len(window) < 20:
        window = frame.tail(min(120, len(frame)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(window["date"], window["portfolio_return"], label="Portfolio return", color="black")
    ax.plot(window["date"], -window["var99_fhs"], label="-99% VaR, GARCH-FHS")
    ax.plot(window["date"], -window["var99_normal"], label="-99% VaR, normal", linestyle="--")
    ax.set_title("Returns and 99% VaR thresholds")
    ax.set_ylabel("Return")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_fund_scenarios(scenarios: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in (("low", "#b85c38"), ("base", "#3d5a80"), ("high", "#2a9d8f")):
        part = scenarios.loc[scenarios["scenario"] == name]
        ax.plot(part["year"], part["nav_mean"], label=f"{name} mean", color=color)
        if name == "base":
            ax.fill_between(part["year"], part["nav_p10"], part["nav_p90"], color=color, alpha=0.2, label="base p10-p90")
    ax.set_xlabel("Fund year")
    ax.set_ylabel("NAV per unit of commitment")
    ax.set_title("Takahashi-Alexander NAV scenarios")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_stress(stress: pd.DataFrame, path: Path) -> None:
    equal = stress.loc[stress["portfolio"] == "equal_weight"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = equal["window"].tolist()
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, equal["cumulative_return"].to_numpy(dtype=float) * 100, color="#3d5a80")
    minvar = stress.loc[stress["portfolio"] == "min_variance"]
    if not minvar.empty:
        aligned = [float(minvar.loc[minvar["window"] == name, "cumulative_return"].iloc[0]) * 100 if name in set(minvar["window"]) else np.nan for name in labels]
        ax.plot(x_pos, aligned, color="#b85c38", marker="o", linestyle="None", label="Min-variance")
        ax.legend(frameon=False)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos, labels)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Equal-weight portfolio during stress windows")
    _save(fig, path)


def write_plots(
    forecasts: pd.DataFrame,
    evaluations: dict,
    scenarios: pd.DataFrame,
    stress: pd.DataFrame,
    outdir: Path,
) -> list[str]:
    names = {
        "breach_rates.png": lambda path: plot_breach_rates(evaluations, path),
        "cumulative_breaches_99.png": lambda path: plot_cumulative_breaches(forecasts, path),
        "var_slice.png": lambda path: plot_var_slice(forecasts, path),
        "fund_nav_scenarios.png": lambda path: plot_fund_scenarios(scenarios, path),
        "stress_cumulative.png": lambda path: plot_stress(stress, path),
    }
    written = []
    for name, drawer in names.items():
        drawer(outdir / name)
        written.append(name)
    return written
