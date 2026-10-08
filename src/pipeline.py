"""End-to-end run: data, VaR backtest, stress, and fund Monte Carlo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import common_sample, evaluate_forecast_frame, select_best_method
from src.config import (
    COV_SHRINKAGE,
    GARCH_REFIT_EVERY,
    GARCH_WINDOW,
    HS_WINDOW,
    MINVAR_REBALANCE_EVERY,
    MINVAR_WINDOW,
    MIN_INDUSTRIES_PER_DAY,
    N_PATHS,
    NORMAL_WINDOW,
    PROJECT_SLUG,
    QUICK_N_PATHS,
    SEED,
    TRAIN_END,
    VAR_ALPHAS,
)
from src.data import industry_columns, load_industry_returns, resolve_backtest_bounds
from src.diagnostics import return_diagnostics
from src.fund_model import run_fund_study
from src.plots import write_plots
from src.portfolio import equal_weight_returns, rolling_min_variance_returns
from src.reporting import PeakTracker, build_metrics, hardware_info, write_documents, write_json
from src.stress import stress_tables
from src.var_engine import rolling_filtered_historical, rolling_historical_var_cvar, rolling_normal_var_cvar

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _analysis_sample(panel: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    matrix = panel[columns].to_numpy(dtype=np.float64)
    keep = np.isfinite(matrix).sum(axis=1) >= MIN_INDUSTRIES_PER_DAY
    kept = panel.loc[keep].reset_index(drop=True)
    industry = kept[columns].to_numpy(dtype=np.float64)
    portfolio = equal_weight_returns(industry)
    if not np.isfinite(portfolio).all():
        raise RuntimeError("equal-weight returns include non-finite values")
    return kept, industry, portfolio


def _check_var_units(frame: pd.DataFrame) -> None:
    """Catch a sign or percent/decimal mix-up. This is not a calibration target."""
    for column in ("var99_normal", "var99_hs", "var99_fhs"):
        median = float(np.median(frame[column].to_numpy(dtype=np.float64)))
        if not 0.005 <= median <= 0.15:
            raise RuntimeError(
                f"{column} median is {median:.4f}. "
                "Expected a one-day 99% loss between 0.5% and 15% for this portfolio."
            )


def _forecast_frame(panel, portfolio, test_mask, normal, historical, filtered) -> pd.DataFrame:
    index = np.flatnonzero(test_mask)
    dates = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    frame = pd.DataFrame(
        {
            "row_id": panel["row_id"].to_numpy()[index],
            "date": dates[index],
            "portfolio_return": portfolio[index],
            "var95_normal": normal[0.05][0][index],
            "var99_normal": normal[0.01][0][index],
            "cvar95_normal": normal[0.05][1][index],
            "cvar99_normal": normal[0.01][1][index],
            "var95_hs": historical[0.05][0][index],
            "var99_hs": historical[0.01][0][index],
            "cvar95_hs": historical[0.05][1][index],
            "cvar99_hs": historical[0.01][1][index],
            "var95_fhs": filtered[0.05][0][index],
            "var99_fhs": filtered[0.01][0][index],
            "cvar95_fhs": filtered[0.05][1][index],
            "cvar99_fhs": filtered[0.01][1][index],
        }
    )
    return common_sample(frame)


def _backtest_table(evaluations: dict) -> pd.DataFrame:
    rows = []
    for method, stats in evaluations.items():
        rows.append({"method": method, **stats})
    return pd.DataFrame(rows)


def _run_study(
    *,
    quick: bool,
    data_dir: Path,
    outdir: Path,
    write_docs: bool,
    project_root: Path,
    tracker: PeakTracker,
    started: float,
) -> dict:
    """Reproduce the study. `quick=True` shortens the backtest and the fund sample."""
    outdir.mkdir(parents=True, exist_ok=True)
    mode = "quick" if quick else "full"
    print(f"mode={mode} data={data_dir} out={outdir}", flush=True)

    panel, manifest = load_industry_returns(data_dir)
    columns = industry_columns(panel)
    panel, industry, portfolio = _analysis_sample(panel, columns)
    dates = pd.to_datetime(panel["date"])
    print(
        f"rows={len(panel)} industries={len(columns)} "
        f"{dates.iloc[0].date()} to {dates.iloc[-1].date()} sha256={manifest['sha256'][:12]}",
        flush=True,
    )

    test_start, test_end = resolve_backtest_bounds(data_dir, quick=quick)
    test_mask = ((dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))).to_numpy()
    if not test_mask.any():
        raise RuntimeError(f"no rows in the test window {test_start} to {test_end}")
    first_test = int(np.flatnonzero(test_mask)[0])
    needed = max(NORMAL_WINDOW, HS_WINDOW, GARCH_WINDOW, MINVAR_WINDOW)
    if first_test < needed:
        raise RuntimeError(f"only {first_test} rows of history before the test; need {needed}")

    print("normal and historical VaR", flush=True)
    normal = rolling_normal_var_cvar(portfolio, NORMAL_WINDOW, VAR_ALPHAS)
    historical = rolling_historical_var_cvar(portfolio, HS_WINDOW, VAR_ALPHAS)
    print("GARCH-filtered historical simulation", flush=True)

    def _progress(done: int, total: int) -> None:
        print(f"GARCH forecast {done} / {total}", flush=True)

    filtered, garch_info = rolling_filtered_historical(
        portfolio,
        test_mask,
        GARCH_WINDOW,
        refit_every=GARCH_REFIT_EVERY,
        alphas=VAR_ALPHAS,
        progress=_progress,
    )
    forecasts = _forecast_frame(panel, portfolio, test_mask, normal, historical, filtered)
    if forecasts.empty:
        raise RuntimeError("no finite VaR forecasts")
    _check_var_units(forecasts)
    n_test_calendar = int(test_mask.sum())
    print(f"forecast rows={len(forecasts)} calendar test days={n_test_calendar}", flush=True)

    var_columns = [
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
    cvar_columns = [
        "row_id",
        "date",
        "cvar95_normal",
        "cvar99_normal",
        "cvar95_hs",
        "cvar99_hs",
        "cvar95_fhs",
        "cvar99_fhs",
    ]
    var_path = outdir / "var_forecasts.csv"
    forecasts[var_columns].to_csv(var_path, index=False)
    forecasts[cvar_columns].to_csv(outdir / "cvar_forecasts.csv", index=False)
    saved = pd.read_csv(var_path)
    evaluations = evaluate_forecast_frame(saved)
    for method in ("normal", "hs", "fhs"):
        evaluations[method]["mean_cvar_95"] = float(forecasts[f"cvar95_{method}"].mean())
        evaluations[method]["mean_cvar_99"] = float(forecasts[f"cvar99_{method}"].mean())
    best_method = select_best_method(evaluations)
    _backtest_table(evaluations).to_csv(outdir / "backtest_summary.csv", index=False)
    print(f"best method={best_method}", flush=True)

    print("minimum-variance portfolio", flush=True)
    minvar = rolling_min_variance_returns(
        industry,
        test_mask,
        window=MINVAR_WINDOW,
        rebalance_every=MINVAR_REBALANCE_EVERY,
        shrinkage=COV_SHRINKAGE,
    )
    diag_mask = ((dates >= pd.Timestamp("1990-01-01")) & (dates <= pd.Timestamp(TRAIN_END))).to_numpy()
    diag_minvar = None
    if diag_mask.any() and int(np.flatnonzero(diag_mask)[0]) >= MINVAR_WINDOW:
        diag_minvar = rolling_min_variance_returns(
            industry,
            diag_mask,
            window=MINVAR_WINDOW,
            rebalance_every=MINVAR_REBALANCE_EVERY,
            shrinkage=COV_SHRINKAGE,
        )
    history_mask = (dates <= pd.Timestamp(TRAIN_END)).to_numpy()
    diagnostics = return_diagnostics(
        portfolio[history_mask],
        None if diag_minvar is None else portfolio[diag_mask],
        None if diag_minvar is None else diag_minvar[diag_mask],
    )

    stress, hypothetical = stress_tables(dates, portfolio, minvar, forecasts, industry)
    stress.to_csv(outdir / "stress_summary.csv", index=False)
    dataset_index = pd.DataFrame({"row_id": panel["row_id"], "date": dates.dt.strftime("%Y-%m-%d")})
    dataset_index.to_csv(outdir / "dataset_index.csv", index=False)

    n_paths = QUICK_N_PATHS if quick else N_PATHS
    print(f"fund Monte Carlo paths={n_paths}", flush=True)
    fund = run_fund_study(n_paths, SEED)
    summary = fund["summary"]
    summary.to_csv(outdir / "fund_paths_summary.csv", index=False)
    summary_saved = pd.read_csv(outdir / "fund_paths_summary.csv")
    year5_saved = summary_saved.loc[summary_saved["year"] == 5].iloc[0]
    fund["year5"] = {
        "nav_mean": float(year5_saved["nav_mean"]),
        "nav_se": float(year5_saved["nav_se"]),
        "se_over_mean": float(year5_saved["nav_se"] / year5_saved["nav_mean"]),
    }
    write_json(outdir / "fund_path_checks.json", fund["checks"])
    fund["scenarios"].to_csv(outdir / "fund_scenarios.csv", index=False)
    fund["convergence"].to_csv(outdir / "fund_convergence.csv", index=False)
    pd.DataFrame(
        {"path_id": np.arange(len(fund["year5_nav"]), dtype=np.int64), "nav": fund["year5_nav"]}
    ).to_csv(outdir / "fund_year5_nav.csv", index=False)

    print("plots", flush=True)
    write_plots(forecasts, evaluations, fund["scenarios"], stress, outdir)

    n_train = int((dates < pd.Timestamp(test_start)).sum())
    runtime = time.perf_counter() - started
    peak = tracker.peak_gb()
    metrics = build_metrics(
        mode=mode,
        evaluations=evaluations,
        best_method=best_method,
        baseline_window=NORMAL_WINDOW,
        fund=fund,
        diagnostics=diagnostics,
        stress_rows=stress.to_dict(orient="records"),
        hypothetical=hypothetical,
        data_manifest=manifest,
        n_train=n_train,
        n_test=int(len(saved)),
        garch_info=garch_info,
        seed=SEED,
    )
    metrics["runtime_seconds"] = runtime
    metrics["hardware"] = hardware_info(peak)
    write_json(outdir / "metrics.json", metrics)
    if write_docs:
        write_documents(project_root, outdir, metrics)
    print(f"runtime_seconds={runtime:.1f} peak_rss_gb={peak:.3f}", flush=True)
    for row in metrics["success_criteria"]:
        print(f"  {'PASS' if row['passed'] else 'FAIL'} {row['name']}: {row['value']}", flush=True)
    print(f"project={PROJECT_SLUG}", flush=True)
    return metrics


def run(
    *,
    quick: bool,
    data_dir: Path,
    outdir: Path,
    write_docs: bool,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Run the study and always stop the memory sampler."""
    started = time.perf_counter()
    tracker = PeakTracker()
    tracker.start()
    try:
        return _run_study(
            quick=quick,
            data_dir=data_dir,
            outdir=outdir,
            write_docs=write_docs,
            project_root=project_root,
            tracker=tracker,
            started=started,
        )
    finally:
        tracker.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portfolio and private-fund risk engine")
    parser.add_argument("--quick", action="store_true", help="Shorter backtest and fewer fund paths")
    parser.add_argument("--device", default="cpu", help="cpu only")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.device != "cpu":
        raise SystemExit("This project runs on CPU only.")
    data_dir = args.data_dir if args.data_dir is not None else PROJECT_ROOT / "data"
    outdir = args.outdir if args.outdir is not None else PROJECT_ROOT / "results"
    write_docs = (not args.quick) and outdir.resolve() == (PROJECT_ROOT / "results").resolve()
    run(
        quick=args.quick,
        data_dir=data_dir,
        outdir=outdir,
        write_docs=write_docs,
        project_root=PROJECT_ROOT,
    )
    return 0
