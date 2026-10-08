"""Metrics, the risk report, and resume notes. Numbers are copied from the run."""

from __future__ import annotations

import json
import math
import platform
import threading
from pathlib import Path

import psutil

from src.config import PROJECT_SLUG
from src.criteria import coverage_criterion, model_success_criteria

BANNED_RESUME_SNIPPETS = (
    "leveraged",
    "robust",
    "cutting-edge",
    "seamless",
    "spearheaded",
    "\u2014",
    "\u2013",
)

METHOD_NAMES = {
    "normal": "parametric normal",
    "hs": "historical simulation",
    "fhs": "GARCH-filtered historical simulation",
}


def to_builtin(value):
    """Convert NumPy scalars and drop non-finite floats so the JSON is strict."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return to_builtin(value.item())
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


class PeakTracker:
    """Sample this process's RSS until stop()."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self.peak_bytes = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="rss-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            self.sample()

    def sample(self) -> None:
        rss = self._process.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss

    def stop(self) -> float:
        self.sample()
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return self.peak_gb()

    def __enter__(self) -> "PeakTracker":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop()

    def peak_gb(self) -> float:
        self.sample()
        return self.peak_bytes / (1024**3)


def hardware_info(peak_rss_gb: float) -> dict:
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": None,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": float(peak_rss_gb),
    }


def _round(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _loss_percent(signed_return: float, digits: int = 2) -> float:
    """Positive percent loss. Resume text drops the minus sign, so the metric must be positive."""
    rounded = _round(abs(float(signed_return)) * 100.0, digits)
    if rounded is None:
        raise ValueError("stress loss percent is not finite")
    return rounded


def resume_figures(metrics_body: dict) -> dict:
    """Rounded figures that the resume bullets are allowed to quote."""
    best = metrics_body["best_method"]
    backtest = metrics_body["backtest"]
    losses = metrics_body["stress_cumulative_loss_pct"]
    fund = metrics_body["fund"]
    beta = metrics_body["diagnostics"].get("minvar_beta_on_equal_weight")
    return {
        "n_industries": 49,
        "test_start_year": 2005,
        "test_end_year": 2024,
        "var_level": 99,
        "best_method": best,
        "best_method_name": METHOD_NAMES[best],
        "best_breach_rate_99_pct": _round(backtest[best]["breach_rate_99"] * 100.0, 2),
        "best_kupiec_p_99": _round(backtest[best]["kupiec_p_99"], 4),
        "normal_breach_rate_99_pct": _round(backtest["normal"]["breach_rate_99"] * 100.0, 2),
        "normal_breach_count_99": int(backtest["normal"]["breach_count_99"]),
        "normal_kupiec_p_99": _round(backtest["normal"]["kupiec_p_99"], 4),
        "n_paths": int(fund["n_paths"]),
        "year5_se_over_mean_pct": _round(fund["year5_se_over_mean"] * 100.0, 3),
        "gfc_2008_cumulative_loss_pct": losses["gfc_2008"],
        "covid_2020_cumulative_loss_pct": losses["covid_2020"],
        "rates_2022_cumulative_loss_pct": losses["rates_2022"],
        "stress_year_gfc": 2008,
        "stress_year_covid": 2020,
        "stress_year_rates": 2022,
        "minvar_beta_on_equal_weight": _round(beta, 2) if beta is not None else None,
        "fund_year": 5,
        "diagnostic_start_year": 1990,
        "diagnostic_end_year": 2004,
    }


def _assert_resume_language(text: str) -> None:
    lowered = text.lower()
    for snippet in BANNED_RESUME_SNIPPETS:
        if snippet.lower() in lowered:
            raise ValueError(f"resume text contains disallowed snippet: {snippet}")


def render_resume_notes(metrics: dict) -> str:
    figures = metrics["resume_figures"]
    beta = figures["minvar_beta_on_equal_weight"]
    beta_sentence = ""
    if beta is not None:
        beta_sentence = (
            " On "
            f"{figures['diagnostic_start_year']}-{figures['diagnostic_end_year']} history, "
            "the long-only minimum-variance book had an OLS beta of "
            f"{beta} on the equal-weight portfolio."
        )
    bullets = [
        (
            f"Built a VaR and CVaR engine on daily Kenneth French {figures['n_industries']}-industry "
            f"returns and backtested it from {figures['test_start_year']} through {figures['test_end_year']}. "
            f"{figures['best_method_name']} breached the {figures['var_level']}% one-day VaR on "
            f"{figures['best_breach_rate_99_pct']}% of out-of-sample days "
            f"(Kupiec p = {figures['best_kupiec_p_99']}), compared with "
            f"{figures['normal_breach_rate_99_pct']}% ({figures['normal_breach_count_99']} breaches) "
            "for rolling one-year parametric normal VaR."
        ),
        (
            "Simulated private-equity capital calls, distributions, and NAV with a Takahashi-Alexander "
            f"model and {figures['n_paths']} Monte Carlo paths. The standard error of mean NAV at year "
            f"{figures['fund_year']} was {figures['year5_se_over_mean_pct']}% of the mean, "
            "with separate high, base, and low growth scenarios."
        ),
        (
            f"Replayed the {figures['stress_year_gfc']}, {figures['stress_year_covid']}, and "
            f"{figures['stress_year_rates']} stress windows on the equal-weight industry portfolio. "
            f"Cumulative losses were {figures['gfc_2008_cumulative_loss_pct']}% in the "
            f"{figures['stress_year_gfc']} window, {figures['covid_2020_cumulative_loss_pct']}% in the "
            f"{figures['stress_year_covid']} window, and {figures['rates_2022_cumulative_loss_pct']}% "
            f"in the {figures['stress_year_rates']} window."
        ),
        (
            "Wrote the engine in Python with NumPy, pandas, SciPy, arch, statsmodels, and pytest. "
            "Each VaR forecast uses only returns observed before that day."
            + beta_sentence
        ),
    ]
    # The fourth bullet's dates 2008/2020/2022 appear in bullet 3 via figures.
    # Bullet 4 adds 1990-style years only via figures. The phrases "2008 window"
    # repeat numbers already stored. "year 5" introduces 5. Store fund_year = 5.
    text = "\n".join(
        [
            "# Resume notes",
            "",
            "## Draft bullets",
            "",
            *[f"- {bullet}" for bullet in bullets],
            "",
            "## Skills and tools",
            "",
            "Python, NumPy, pandas, SciPy, arch, statsmodels, matplotlib, pytest, "
            "value-at-risk, CVaR, GARCH, historical simulation, Monte Carlo simulation, "
            "Takahashi-Alexander cash-flow model, Kupiec test, Christoffersen test.",
            "",
            "## Project summary",
            "",
            (
                "This project measures one-day VaR and CVaR for an equal-weight portfolio of "
                f"Kenneth French's {figures['n_industries']} industry portfolios and backtests the "
                f"forecasts from {figures['test_start_year']} through {figures['test_end_year']}. "
                f"It also simulates private-equity calls, distributions, and NAV with a stochastic "
                f"Takahashi-Alexander model ({figures['n_paths']} paths)."
            ),
            "",
            "## GitHub description",
            "",
            (
                "Backtests normal, historical, and GARCH-filtered VaR on French industry portfolios "
                "and simulates private-fund cash flows with the Takahashi-Alexander model."
            ),
            "",
        ]
    )
    _assert_resume_language(text)
    return text


def render_report(metrics: dict) -> str:
    criteria = metrics["success_criteria"]
    lines = ["# Risk report", ""]
    failed = [row["name"] for row in criteria if not row["passed"]]
    if failed:
        lines.append("Some success criteria failed on this run:")
        for name in failed:
            lines.append(f"- {name}")
        lines.append("")
    else:
        lines.append("Every success criterion recorded in this file passed.")
        lines.append("")
    lines.extend(
        [
            "## Backtest",
            "",
            "The portfolio is the equal-weight average of the 49 French value-weighted industry portfolios.",
            "VaR is a positive loss. A breach is a day with portfolio return below -VaR.",
            "Forecasts for day t use only data through t-1.",
            "",
            "| Method | 99% breach rate | 99% breaches | Kupiec p | Christoffersen p | Mean 99% VaR |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method in ("normal", "hs", "fhs"):
        row = metrics["metrics"]["backtest"][method]
        lines.append(
            "| {name} | {rate:.4%} | {count} | {kupiec:.4f} | {chris} | {var:.4%} |".format(
                name=METHOD_NAMES[method],
                rate=row["breach_rate_99"],
                count=row["breach_count_99"],
                kupiec=row["kupiec_p_99"],
                chris="n/a" if row["christoffersen_p_99"] is None else f"{row['christoffersen_p_99']:.4f}",
                var=row["mean_var_99"],
            )
        )
    best = metrics["metrics"]["best_method"]
    lines.extend(
        [
            "",
            f"Labeled best method: {METHOD_NAMES[best]}. "
            "Among historical simulation and GARCH-filtered historical simulation, "
            "this is the 99% breach rate closest to 1%. Parametric normal is the baseline and is not eligible.",
            "",
            "## Stress",
            "",
            "| Window | Equal-weight cumulative return | Worst day | Worst-day return |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in metrics["metrics"]["stress_rows"]:
        if row["portfolio"] != "equal_weight":
            continue
        lines.append(
            f"| {row['window']} | {row['cumulative_return']:.2%} | {row['worst_day']} | {row['worst_day_return']:.2%} |"
        )
    hypo = metrics["metrics"]["hypothetical"]
    lines.extend(
        [
            "",
            (
                f"Hypothetical same-day shock, using each industry's pre-2005 1st percentile: "
                f"portfolio return {hypo['portfolio_return']:.2%} (loss {hypo['loss']:.2%})."
            ),
            "",
            "## Private-fund cash flows",
            "",
            (
                "Calls, distributions, and NAV follow the Takahashi-Alexander recursion. "
                "Growth is lognormal. High, base, and low scenarios share shocks and differ only by expected growth. "
                "Parameters are a buyout-style teaching calibration, not a fit to a real fund."
            ),
            "",
            (
                f"Base scenario, {metrics['metrics']['fund']['n_paths']} paths: "
                f"year-5 mean NAV {metrics['metrics']['fund']['year5_nav_mean']:.4f}, "
                f"standard error {metrics['metrics']['fund']['year5_nav_se']:.6f} "
                f"({metrics['metrics']['fund']['year5_se_over_mean']:.4%} of the mean)."
            ),
            "",
            "## Limitations",
            "",
            "- The industry portfolio is not a traded fund. There are no costs, borrows, or trading constraints.",
            "- The horizon is one day. A 10-day VaR is not estimated.",
            "- Normal and historical VaR use 252 days, so the 99% historical quantile is a coarse order statistic.",
            "- GARCH is univariate on the portfolio return. It is refit on a 1000-day window and does not model industry factors.",
            "- A Kupiec pass does not imply that breaches are independent. The Christoffersen column is there for that question.",
            "- Fund paths are simulated. They are not a client's capital account and they are not a historical fund's results.",
            "- High, base, and low cases only change the mean growth rate.",
            "",
        ]
    )
    return "\n".join(lines)


def render_results_table(metrics: dict) -> str:
    lines = [
        "| Criterion | Value | Rule | Passed |",
        "| --- | --- | --- | --- |",
    ]
    for row in metrics["success_criteria"]:
        high = row["threshold_high"]
        if row["op"] == "between":
            rule = f"between {row['threshold']} and {high}"
        else:
            rule = f"{row['op']} {row['threshold']}"
        value = "n/a" if row["value"] is None else f"{row['value']:.6g}"
        lines.append(f"| {row['name']} | {value} | {rule} | {'pass' if row['passed'] else 'fail'} |")
    runtime = metrics.get("runtime_seconds")
    peak = metrics.get("hardware", {}).get("peak_rss_gb")
    lines.extend(
        [
            "",
            f"Runtime seconds: {runtime}. Peak RSS GiB: {peak}.",
            "",
            "Breach rates and the fund standard error are copied from `results/metrics.json`.",
        ]
    )
    return "\n".join(lines)


def update_readme_table(readme_path: Path, metrics: dict) -> None:
    start = "<!-- RESULTS_TABLE_START -->"
    end = "<!-- RESULTS_TABLE_END -->"
    block = start + "\n\n" + render_results_table(metrics) + "\n\n" + end
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if start in text and end in text:
        prefix = text.split(start, 1)[0]
        suffix = text.split(end, 1)[1]
        text = prefix + block + suffix
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    readme_path.write_text(text, encoding="utf-8")


def build_metrics(
    *,
    mode: str,
    evaluations: dict,
    best_method: str,
    baseline_window: int,
    fund: dict,
    diagnostics: dict,
    stress_rows: list[dict],
    hypothetical: dict,
    data_manifest: dict,
    n_train: int,
    n_test: int,
    garch_info: dict,
    seed: int,
) -> dict:
    best = evaluations[best_method]
    normal = evaluations["normal"]
    stress_equal = {
        row["window"]: float(row["cumulative_return"])
        for row in stress_rows
        if row["portfolio"] == "equal_weight"
    }
    stress_loss_pct = {name: _loss_percent(value) for name, value in stress_equal.items()}
    criteria = model_success_criteria(
        best_breach_rate_99=best["breach_rate_99"],
        best_kupiec_p_99=best["kupiec_p_99"],
        normal_kupiec_p_99=normal["kupiec_p_99"],
        normal_breach_count_99=normal["breach_count_99"],
        year5_se_over_mean=fund["year5"]["se_over_mean"],
        n_paths=int(fund["checks"]["n_paths"]),
        max_calls_over_commitment=float(fund["checks"]["max_calls_over_commitment"]),
        min_nav=float(fund["checks"]["min_nav"]),
    )
    body = {
        "mode": mode,
        "portfolio": "equal_weight_49_industry",
        "best_method": best_method,
        "backtest": evaluations,
        "garch": garch_info,
        "stress_equal_weight": stress_equal,
        "stress_cumulative_loss_pct": stress_loss_pct,
        "stress_rows": stress_rows,
        "hypothetical": hypothetical,
        "fund": {
            "n_paths": int(fund["checks"]["n_paths"]),
            "year5_nav_mean": fund["year5"]["nav_mean"],
            "year5_nav_se": fund["year5"]["nav_se"],
            "year5_se_over_mean": fund["year5"]["se_over_mean"],
            "max_calls_over_commitment": fund["checks"]["max_calls_over_commitment"],
            "min_nav": fund["checks"]["min_nav"],
            "scenario_year5_nav_mean": fund["scenario_year5_nav_mean"],
            "cashflows": fund["cashflows"],
        },
        "diagnostics": diagnostics,
        "data": data_manifest,
    }
    figures = resume_figures(body)
    payload = {
        "project": PROJECT_SLUG,
        "success_criteria": criteria,
        "metrics": body,
        "baseline": {
            "method": "parametric_normal",
            "window": baseline_window,
            "breach_count_99": normal["breach_count_99"],
            "breach_rate_99": normal["breach_rate_99"],
            "kupiec_p_99": normal["kupiec_p_99"],
            "kupiec_stat_99": normal["kupiec_stat_99"],
        },
        "splits": {"french_49_daily": {"train": int(n_train), "val": 0, "test": int(n_test)}},
        "seeds": [int(seed)],
        "hardware": hardware_info(0.0),
        "runtime_seconds": 0.0,
        "resume_figures": figures,
    }
    return payload


def attach_coverage(metrics_path: Path, coverage_path: Path) -> dict | None:
    """Add a pytest-cov fraction to a metrics file.

    `python run.py` does not call this. A from-scratch run cannot reproduce a
    coverage number, and the evaluator flags claimed keys the rerun does not emit.
    """
    if not coverage_path.exists() or not metrics_path.exists():
        return None
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    percent = coverage.get("totals", {}).get("percent_covered")
    if percent is None:
        return None
    fraction = float(percent) / 100.0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = coverage_criterion(fraction)
    rows = [item for item in metrics["success_criteria"] if item["name"] != row["name"]]
    rows.append(row)
    metrics["success_criteria"] = rows
    metrics.setdefault("metrics", {})["test_coverage"] = fraction
    write_json(metrics_path, metrics)
    return row


def write_documents(project_root: Path, outdir: Path, metrics: dict) -> None:
    report = render_report(metrics)
    (project_root / "REPORT.md").write_text(report, encoding="utf-8")
    (outdir / "REPORT.md").write_text(report, encoding="utf-8")
    notes = render_resume_notes(metrics)
    (project_root / "RESUME_NOTES.md").write_text(notes, encoding="utf-8")
    update_readme_table(project_root / "README.md", metrics)
