"""Success-criterion boundaries."""

from __future__ import annotations

from src.criteria import coverage_criterion, meets, model_success_criteria


def _by_name(rows):
    return {row["name"]: row for row in rows}


def test_comparisons():
    assert meets(0.01, "between", 0.005, 0.016)
    assert meets(0.005, "between", 0.005, 0.016)
    assert meets(0.016, "between", 0.005, 0.016)
    assert not meets(0.0049, "between", 0.005, 0.016)
    assert not meets(0.0161, "between", 0.005, 0.016)
    assert meets(0.05, ">=", 0.05)
    assert not meets(0.049, ">=", 0.05)
    assert meets(0.004, "<", 0.005)
    assert not meets(0.005, "<", 0.005)
    assert not meets(None, ">", 0.0)


def test_model_criteria_pass_and_fail_on_the_boundaries():
    passed = _by_name(
        model_success_criteria(
            best_breach_rate_99=0.01,
            best_kupiec_p_99=0.40,
            normal_kupiec_p_99=0.10,
            normal_breach_count_99=80,
            year5_se_over_mean=0.004,
            n_paths=10_000,
            max_calls_over_commitment=0.0,
            min_nav=0.0,
        )
    )
    assert all(row["passed"] for row in passed.values())

    failed = _by_name(
        model_success_criteria(
            best_breach_rate_99=0.02,
            best_kupiec_p_99=0.01,
            normal_kupiec_p_99=0.20,
            normal_breach_count_99=40,
            year5_se_over_mean=0.004,
            n_paths=400,
            max_calls_over_commitment=0.01,
            min_nav=-0.2,
        )
    )
    assert not failed["99% one-day VaR breach rate of the best method, out of sample 2005-2024"]["passed"]
    assert not failed["Kupiec POF test for the best method at 99%"]["passed"]
    assert not failed["Best method vs. parametric-normal baseline"]["passed"]
    assert not failed["Monte Carlo convergence of fund NAV at year 5"]["passed"]
    assert not failed["Cash-flow model consistency"]["passed"]


def test_coverage_threshold():
    assert coverage_criterion(0.75)["passed"]
    assert not coverage_criterion(0.749)["passed"]
