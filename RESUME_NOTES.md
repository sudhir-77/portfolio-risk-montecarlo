# Resume notes

## Draft bullets

- Built a VaR and CVaR engine on daily Kenneth French 49-industry returns and backtested it from 2005 through 2024. GARCH-filtered historical simulation breached the 99% one-day VaR on 1.25% of out-of-sample days (Kupiec p = 0.0841), compared with 2.62% (132 breaches) for rolling one-year parametric normal VaR.
- Simulated private-equity capital calls, distributions, and NAV with a Takahashi-Alexander model and 10000 Monte Carlo paths. The standard error of mean NAV at year 5 was 0.234% of the mean, with separate high, base, and low growth scenarios.
- Replayed the 2008, 2020, and 2022 stress windows on the equal-weight industry portfolio. Cumulative returns were -40.47% in the 2008 window, -27.57% in the 2020 window, and -12.7% in the 2022 window.
- Wrote the engine in Python with NumPy, pandas, SciPy, arch, statsmodels, and pytest. Each VaR forecast uses only returns observed before that day. On 1990-2004 history, the long-only minimum-variance book had an OLS beta of 0.62 on the equal-weight portfolio.

## Skills and tools

Python, NumPy, pandas, SciPy, arch, statsmodels, matplotlib, pytest, value-at-risk, CVaR, GARCH, historical simulation, Monte Carlo simulation, Takahashi-Alexander cash-flow model, Kupiec test, Christoffersen test.

## Project summary

This project measures one-day VaR and CVaR for an equal-weight portfolio of Kenneth French's 49 industry portfolios and backtests the forecasts from 2005 through 2024. It also simulates private-equity calls, distributions, and NAV with a stochastic Takahashi-Alexander model (10000 paths).

## GitHub description

Backtests normal, historical, and GARCH-filtered VaR on French industry portfolios and simulates private-fund cash flows with the Takahashi-Alexander model.
