"""Takahashi-Alexander private-fund cash flows with lognormal NAV growth."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    BOW,
    COMMITMENT,
    CONTRIBUTION_RATES,
    DISCOUNT_RATE,
    FUND_LIFE,
    GROWTH_MEAN,
    GROWTH_VOL,
    SCENARIO_GROWTH,
    YIELD,
)


@dataclass
class FundSim:
    nav: np.ndarray
    calls: np.ndarray
    dists: np.ndarray
    calls_cum: np.ndarray
    dists_cum: np.ndarray
    commitment: float
    growth_mean: float
    growth_vol: float


def draw_growth_shocks(n_paths: int, life: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_paths, life))


def _growth_from_shocks(shocks: np.ndarray, growth_mean: float, growth_vol: float) -> np.ndarray:
    if growth_mean <= -1.0:
        raise ValueError("growth_mean must be greater than -1")
    if growth_vol < 0.0:
        raise ValueError("growth_vol must be non-negative")
    log_mean = math.log1p(growth_mean) - 0.5 * growth_vol * growth_vol
    return np.expm1(log_mean + growth_vol * shocks)


def simulate_fund(
    n_paths: int,
    life: int = FUND_LIFE,
    contribution_rates: tuple[float, ...] = CONTRIBUTION_RATES,
    growth_mean: float = GROWTH_MEAN,
    growth_vol: float = GROWTH_VOL,
    yield_rate: float = YIELD,
    bow: float = BOW,
    commitment: float = COMMITMENT,
    seed: int = 0,
    shocks: np.ndarray | None = None,
) -> FundSim:
    """One path of the Takahashi-Alexander recursion, vectorized across paths.

    Within each year the order is: call unfunded capital, grow opening NAV,
    add the call, distribute a fraction of that gross amount, and carry the
    rest as ending NAV.

    The distribution rate is max(yield, (year / life) ** bow), capped at 1.
    Growth is lognormal so a shock cannot push 1 + growth through zero.
    """
    if life < 1:
        raise ValueError("life must be positive")
    if bow <= 0.0:
        raise ValueError("bow must be positive")
    if not 0.0 <= yield_rate <= 1.0:
        raise ValueError("yield_rate must be between 0 and 1")
    if commitment <= 0.0:
        raise ValueError("commitment must be positive")
    if shocks is None:
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        shocks = draw_growth_shocks(n_paths, life, seed)
    shocks = np.asarray(shocks, dtype=np.float64)
    if shocks.ndim != 2 or shocks.shape[1] != life:
        raise ValueError("shocks must have shape (n_paths, life)")
    n_paths = int(shocks.shape[0])
    growth = _growth_from_shocks(shocks, growth_mean, growth_vol)

    nav = np.zeros(n_paths, dtype=np.float64)
    pic = np.zeros(n_paths, dtype=np.float64)
    calls = np.zeros((n_paths, life), dtype=np.float64)
    dists = np.zeros((n_paths, life), dtype=np.float64)
    navs = np.zeros((n_paths, life), dtype=np.float64)

    for year in range(1, life + 1):
        rate = float(contribution_rates[min(year - 1, len(contribution_rates) - 1)])
        rate = min(max(rate, 0.0), 1.0)
        unfunded = np.maximum(commitment - pic, 0.0)
        call = rate * unfunded
        pic = pic + call
        if float(np.max(pic - commitment)) > 1e-8:
            raise RuntimeError("capital calls exceeded commitment")
        pic = np.minimum(pic, commitment)

        nav_gross = nav * (1.0 + growth[:, year - 1]) + call
        if float(np.min(nav_gross)) < -1e-8:
            raise RuntimeError("gross NAV went negative")
        nav_gross = np.maximum(nav_gross, 0.0)
        distribution_rate = min(1.0, max(yield_rate, (year / life) ** bow))
        dist = distribution_rate * nav_gross
        nav = nav_gross - dist
        calls[:, year - 1] = call
        dists[:, year - 1] = dist
        navs[:, year - 1] = nav

    return FundSim(
        nav=navs,
        calls=calls,
        dists=dists,
        calls_cum=np.cumsum(calls, axis=1),
        dists_cum=np.cumsum(dists, axis=1),
        commitment=float(commitment),
        growth_mean=float(growth_mean),
        growth_vol=float(growth_vol),
    )


def path_checks(sim: FundSim) -> dict[str, float | int]:
    excess = float(np.max(sim.calls_cum - sim.commitment))
    return {
        "max_calls_over_commitment": float(max(0.0, excess)),
        "min_nav": float(np.min(sim.nav)),
        "n_paths": int(sim.nav.shape[0]),
    }


def summary_frame(sim: FundSim) -> pd.DataFrame:
    n_paths = sim.nav.shape[0]
    if n_paths > 1:
        nav_se = sim.nav.std(axis=0, ddof=1) / math.sqrt(n_paths)
    else:
        nav_se = np.zeros(sim.nav.shape[1], dtype=np.float64)
    return pd.DataFrame(
        {
            "year": np.arange(1, sim.nav.shape[1] + 1, dtype=np.int64),
            "nav_mean": sim.nav.mean(axis=0),
            "nav_se": nav_se,
            "calls_cum_mean": sim.calls_cum.mean(axis=0),
            "dist_cum_mean": sim.dists_cum.mean(axis=0),
            "commitment": sim.commitment,
        }
    )


def scenario_frame(sim: FundSim, scenario: str) -> pd.DataFrame:
    frame = summary_frame(sim)
    frame.insert(0, "scenario", scenario)
    frame["nav_p10"] = np.quantile(sim.nav, 0.10, axis=0)
    frame["nav_p50"] = np.quantile(sim.nav, 0.50, axis=0)
    frame["nav_p90"] = np.quantile(sim.nav, 0.90, axis=0)
    return frame


def year5_ratio(summary: pd.DataFrame) -> dict[str, float]:
    row = summary.loc[summary["year"] == 5].iloc[0]
    mean = float(row["nav_mean"])
    se = float(row["nav_se"])
    ratio = se / mean if mean > 0.0 else 1.0
    return {"nav_mean": mean, "nav_se": se, "se_over_mean": ratio}


def convergence_table(year5_nav: np.ndarray, n_paths: int) -> pd.DataFrame:
    grid = [100, 250, 500, 1000, 2500, 5000, 10000]
    counts = [count for count in grid if count <= n_paths]
    if n_paths not in counts:
        counts.append(n_paths)
    rows = []
    for count in counts:
        sample = np.asarray(year5_nav[:count], dtype=np.float64)
        mean = float(sample.mean())
        se = float(sample.std(ddof=1) / math.sqrt(count)) if count > 1 else 0.0
        rows.append(
            {
                "n_paths": count,
                "year5_nav_mean": mean,
                "year5_nav_se": se,
                "se_over_mean": se / mean if mean > 0.0 else 1.0,
            }
        )
    return pd.DataFrame(rows)


def _npv(rate: np.ndarray, cashflows: np.ndarray) -> np.ndarray:
    years = np.arange(1, cashflows.shape[1] + 1, dtype=np.float64)
    discount = np.power(1.0 + rate[:, None], years)
    return (cashflows / discount).sum(axis=1)


def internal_rates(cashflows: np.ndarray) -> np.ndarray:
    """Annual IRR for each row. NaN when the root is outside (-95%, 1000%)."""
    cashflows = np.asarray(cashflows, dtype=np.float64)
    n_paths = cashflows.shape[0]
    low = np.full(n_paths, -0.95)
    high = np.full(n_paths, 10.0)
    value_low = _npv(low, cashflows)
    value_high = _npv(high, cashflows)
    ok = (value_low > 0.0) & (value_high < 0.0)
    for _ in range(70):
        mid = (low + high) / 2.0
        value = _npv(mid, cashflows)
        # NPV falls as the discount rate rises, so a positive NPV means the root is higher.
        low = np.where(value > 0.0, mid, low)
        high = np.where(value > 0.0, high, mid)
    rate = (low + high) / 2.0
    rate = np.where(ok, rate, np.nan)
    return rate


def cashflow_metrics(sim: FundSim, discount_rate: float = DISCOUNT_RATE) -> dict[str, float]:
    flows = -sim.calls + sim.dists
    flows = flows.copy()
    flows[:, -1] = flows[:, -1] + sim.nav[:, -1]
    rates = internal_rates(flows)
    valid = rates[np.isfinite(rates)]
    mean_flow = flows.mean(axis=0, keepdims=True)
    mean_irr = internal_rates(mean_flow)
    years = np.arange(1, flows.shape[1] + 1, dtype=np.float64)
    npv = float(np.sum(mean_flow[0] / np.power(1.0 + discount_rate, years)))
    return {
        "irr_mean": float(valid.mean()) if valid.size else float("nan"),
        "irr_median": float(np.median(valid)) if valid.size else float("nan"),
        "irr_of_mean_cash_flows": float(mean_irr[0]),
        "irr_valid_fraction": float(valid.size / rates.size),
        "npv_of_mean_cash_flows": npv,
        "discount_rate": float(discount_rate),
    }


def run_fund_study(n_paths: int, seed: int) -> dict:
    """Base, high, and low scenarios share the same normal shocks."""
    shocks = draw_growth_shocks(n_paths, FUND_LIFE, seed)
    simulations = {
        name: simulate_fund(n_paths, growth_mean=mean, growth_vol=GROWTH_VOL, shocks=shocks)
        for name, mean in SCENARIO_GROWTH.items()
    }
    base = simulations["base"]
    summary = summary_frame(base)
    checks = path_checks(base)
    # Also require the property on the high and low growth draws.
    for name, sim in simulations.items():
        extra = path_checks(sim)
        if extra["max_calls_over_commitment"] > 1e-8 or extra["min_nav"] < -1e-9:
            raise RuntimeError(f"cash-flow property failed for the {name} scenario")
    scenarios = pd.concat(
        [scenario_frame(sim, name) for name, sim in simulations.items()],
        ignore_index=True,
    )
    year5 = year5_ratio(summary)
    return {
        "summary": summary,
        "checks": checks,
        "scenarios": scenarios,
        "convergence": convergence_table(base.nav[:, 4], n_paths),
        "year5_nav": base.nav[:, 4].copy(),
        "year5": year5,
        "cashflows": cashflow_metrics(base),
        "scenario_year5_nav_mean": {
            name: float(sim.nav[:, 4].mean()) for name, sim in simulations.items()
        },
    }
