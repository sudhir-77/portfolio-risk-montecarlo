"""Fixed project choices.

Windows, fund parameters, and the backtest calendar are set here so they are
not adjusted from 2005-2024 breach counts.
"""

from __future__ import annotations

PROJECT_SLUG = "portfolio-risk-montecarlo"

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "49_Industry_Portfolios_Daily_CSV.zip"
)
ZIP_NAME = "49_Industry_Portfolios_Daily_CSV.zip"
MIN_ROWS = 20_000
N_INDUSTRIES = 49
# Drop a day only when too many industries are missing to form a portfolio.
MIN_INDUSTRIES_PER_DAY = 25

# Official evaluation window. Validation is unused: nothing is tuned on a slice
# of the backtest.
TEST_START = "2005-01-01"
TEST_END = "2024-12-31"
TRAIN_END = "2004-12-31"

# Quick mode is a short smoke sample. It does not choose the full-run settings.
QUICK_TEST_START = "2016-01-01"
QUICK_TEST_END = "2016-06-30"

# Baseline specified by the project brief: one-year rolling normal VaR.
NORMAL_WINDOW = 252
# Same window as the baseline so historical simulation sees the same days.
HS_WINDOW = 252
# A 1% quantile needs more than the two or three tail points in a 252-day
# sample. Two years of raw history is still thin; 1000 days is the residual
# sample used for filtered historical simulation.
GARCH_WINDOW = 1000
# 1 means every forecast day gets its own fit on data through the previous day.
GARCH_REFIT_EVERY = 1
GARCH_MAXITER = 100

VAR_LEVELS = (0.95, 0.99)
VAR_ALPHAS = (0.05, 0.01)

SEED = 42

# Long-only minimum-variance book. Shrinkage is fixed, not estimated on the test set.
COV_SHRINKAGE = 0.20
MINVAR_WINDOW = 252
MINVAR_REBALANCE_EVERY = 63

N_PATHS = 10_000
QUICK_N_PATHS = 400
COMMITMENT = 1.0
FUND_LIFE = 12
# Buyout-style teaching calibration of the Takahashi-Alexander recursion.
# These are model inputs, not estimates from a real fund's cash flows.
BOW = 2.5
YIELD = 0.05
GROWTH_MEAN = 0.12
GROWTH_VOL = 0.15
CONTRIBUTION_RATES = (
    0.30,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.08,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
)
SCENARIO_GROWTH = {"low": 0.02, "base": 0.12, "high": 0.20}
DISCOUNT_RATE = 0.08

STRESS_WINDOWS = (
    ("gfc_2008", "2008-09-01", "2009-03-31"),
    ("covid_2020", "2020-02-15", "2020-03-31"),
    ("rates_2022", "2022-01-03", "2022-10-31"),
)

USER_AGENT = "portfolio-risk-montecarlo academic research (education; local)"
HTTP_TIMEOUT_SECONDS = 120
