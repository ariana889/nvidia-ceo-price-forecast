# nvidia-ceo-forecast

Minute-by-minute quantile forecasts (5%, 25%, 50%, 75%, 95%) of every trading market of Polymarket's [Next CEO of NVIDIA?](https://polymarket.com/event/next-ceo-of-nvidia) event, four hours ahead.

[![make.fun bounty](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01)

`node main.ts` reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<case>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Model

Empirical predictive distribution of standardised price changes:

- `sigma_t` — MAD of 1-minute changes over the trailing 1440 minutes, refreshed every 30 minutes of history.
- `z(t,h) = (p[t+h] - p[t]) / (sigma_t * sqrt(h))` for horizons 5…240 minutes, winsorised at ±12 and pooled across the case's markets.
- The forecast is `q_tau(h) = clip(x + sigma_now * sqrt(h) * z_tau(h), 0, 1)`, with `z_tau` the empirical quantile at each level; monotonicity is enforced after rounding.

No network, no randomness: deterministic per input file. MIT licensed.
