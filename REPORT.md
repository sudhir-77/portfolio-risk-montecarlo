# Risk report

Every success criterion recorded in this file passed.

## Backtest

The portfolio is the equal-weight average of the 49 French value-weighted industry portfolios.
VaR is a positive loss. A breach is a day with portfolio return below -VaR.
Forecasts for day t use only data through t-1.

| Method | 99% breach rate | 99% breaches | Kupiec p | Christoffersen p | Mean 99% VaR |
| --- | --- | --- | --- | --- | --- |
| parametric normal | 2.6227% | 132 | 0.0000 | 0.0008 | 2.6299% |
| historical simulation | 1.8081% | 91 | 0.0000 | 0.0071 | 3.1017% |
| GARCH-filtered historical simulation | 1.2517% | 63 | 0.0841 | 0.8170 | 2.9307% |

Labeled best method: GARCH-filtered historical simulation. Among historical simulation and GARCH-filtered historical simulation, this is the 99% breach rate closest to 1%. Parametric normal is the baseline and is not eligible.

## Stress

| Window | Equal-weight cumulative return | Worst day | Worst-day return |
| --- | --- | --- | --- |
| gfc_2008 | -40.47% | 2008-12-01 | -9.49% |
| covid_2020 | -27.57% | 2020-03-16 | -11.51% |
| rates_2022 | -12.70% | 2022-06-13 | -4.30% |

Hypothetical same-day shock, using each industry's pre-2005 1st percentile: portfolio return -4.17% (loss 4.17%).

## Private-fund cash flows

Calls, distributions, and NAV follow the Takahashi-Alexander recursion. Growth is lognormal. High, base, and low scenarios share shocks and differ only by expected growth. Parameters are a buyout-style teaching calibration, not a fit to a real fund.

Base scenario, 10000 paths: year-5 mean NAV 0.7875, standard error 0.001841 (0.2337% of the mean).

## Limitations

- The industry portfolio is not a traded fund. There are no costs, borrows, or trading constraints.
- The horizon is one day. A 10-day VaR is not estimated.
- Normal and historical VaR use 252 days, so the 99% historical quantile is a coarse order statistic.
- GARCH is univariate on the portfolio return. It is refit on a 1000-day window and does not model industry factors.
- A Kupiec pass does not imply that breaches are independent. The Christoffersen column is there for that question.
- Fund paths are simulated. They are not a client's capital account and they are not a historical fund's results.
- High, base, and low cases only change the mean growth rate.
