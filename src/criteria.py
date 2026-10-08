"""Pure pass/fail rules for the project success criteria."""

from __future__ import annotations

import math


def _finite(value: float | None) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def meets(value: float | None, op: str, threshold: float, threshold_high: float | None = None) -> bool:
    if not _finite(value):
        return False
    number = float(value)
    if op == "between":
        if threshold_high is None:
            raise ValueError("between requires threshold_high")
        return float(threshold) <= number <= float(threshold_high)
    if op == ">=":
        return number >= float(threshold)
    if op == ">":
        return number > float(threshold)
    if op == "<=":
        return number <= float(threshold)
    if op == "<":
        return number < float(threshold)
    raise ValueError(f"unknown comparison {op}")


def _row(
    name: str,
    value: float | None,
    op: str,
    threshold: float,
    threshold_high: float | None,
    passed: bool | None = None,
) -> dict:
    return {
        "name": name,
        "value": None if value is None else float(value),
        "op": op,
        "threshold": float(threshold),
        "threshold_high": None if threshold_high is None else float(threshold_high),
        "passed": bool(meets(value, op, threshold, threshold_high) if passed is None else passed),
    }


def model_success_criteria(
    best_breach_rate_99: float | None,
    best_kupiec_p_99: float | None,
    normal_kupiec_p_99: float | None,
    normal_breach_count_99: int | None,
    year5_se_over_mean: float | None,
    n_paths: int,
    max_calls_over_commitment: float,
    min_nav: float,
) -> list[dict]:
    """Criteria the market-data run can measure. Coverage is attached after pytest."""
    if best_kupiec_p_99 is None or normal_kupiec_p_99 is None or normal_breach_count_99 is None:
        baseline_gap = None
        baseline_pass = False
    else:
        baseline_gap = float(best_kupiec_p_99) - float(normal_kupiec_p_99)
        baseline_pass = bool(baseline_gap > 0.0 and normal_breach_count_99 >= 0)

    cashflow_violation = max(
        float(max_calls_over_commitment),
        max(0.0, -float(min_nav)),
    )
    convergence_pass = bool(
        n_paths >= 10_000 and meets(year5_se_over_mean, "<", 0.005, None)
    )
    return [
        _row(
            "99% one-day VaR breach rate of the best method, out of sample 2005-2024",
            best_breach_rate_99,
            "between",
            0.005,
            0.016,
        ),
        _row(
            "Kupiec POF test for the best method at 99%",
            best_kupiec_p_99,
            ">=",
            0.05,
            None,
        ),
        _row(
            "Best method vs. parametric-normal baseline",
            baseline_gap,
            ">",
            0.0,
            None,
            passed=baseline_pass,
        ),
        _row(
            "Monte Carlo convergence of fund NAV at year 5",
            year5_se_over_mean,
            "<",
            0.005,
            None,
            passed=convergence_pass,
        ),
        _row(
            "Cash-flow model consistency",
            cashflow_violation,
            "<=",
            1e-8,
            None,
        ),
    ]


def coverage_criterion(fraction_covered: float) -> dict:
    return _row("Test coverage of src/", fraction_covered, ">=", 0.75, None)
