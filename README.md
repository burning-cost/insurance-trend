# insurance-trend

Loss cost trend analysis for UK personal lines insurance pricing.

## The problem

Every UK motor and household pricing actuary does loss cost trend analysis every quarter. The workflow is: take aggregate accident-period experience data, fit a log-linear trend to frequency and severity separately, project forward to the next rating period, and report a trend rate with a confidence interval.

Currently this is done in Excel, SAS, or bespoke R scripts. There is no Python library for it. `chainladder-python` handles reserving triangles but does nothing for pricing trend — it applies user-specified factors, it does not fit them from data.

The post-2021 inflationary environment has made this more urgent: UK motor claims inflation ran at 34% from 2019 to 2023 versus CPI of 21% — a 13 percentage point superimposed component that CPI alone does not capture. A library that cannot identify structural breaks (COVID lockdown, Ogden rate change) will produce misleading trend estimates.

## What this library does

```python
from insurance_trend import LossCostTrendFitter, ExternalIndex

# Fetch ONS motor repair index for severity deflation
motor_repair_idx = ExternalIndex.from_ons('HPTH')

fitter = LossCostTrendFitter(
    periods=df['accident_quarter'],
    claim_counts=df['claim_count'],
    earned_exposure=df['earned_vehicles'],
    total_paid=df['paid_claims'],
    external_index=motor_repair_idx,
)

result = fitter.fit(
    detect_breaks=True,   # auto-detect COVID, Ogden rate change
    seasonal=True,        # quarterly seasonal dummies
)

print(result.trend_rate)     # e.g. 0.085 — 8.5% pa loss cost trend
print(result.decompose())    # freq_trend, sev_trend, superimposed
fig = result.plot()          # 3-panel diagnostic figure
```

## Classes

- **`FrequencyTrendFitter`** — log-linear OLS on log(claims/exposure). Optional WLS, quarterly seasonal dummies, structural break detection via ruptures PELT, piecewise refitting on detected breaks, bootstrap CI, local linear trend alternative.

- **`SeverityTrendFitter`** — same as frequency, plus optional external index deflation. When an index is supplied, the fit runs on deflated severity and `superimposed_inflation()` gives the residual trend not explained by the index.

- **`LossCostTrendFitter`** — wraps the frequency and severity fitters, combines results, provides `decompose()` and `projected_loss_cost()`.

- **`ExternalIndex`** — fetches ONS time series from the public API (no auth required), with a catalogue of UK insurance-relevant codes. Also accepts user-supplied CSV for BCIS and other subscription data.

## Why log-linear

The industry baseline. Fits `log(y) = alpha + beta*t + seasonal + epsilon` via OLS. The annual trend rate is `exp(beta * periods_per_year) - 1`. The model is transparent, easily explainable to a regulator, and fast enough to bootstrap 1000 replicates in under a second.

The local linear trend alternative (`method='local_linear_trend'`) uses statsmodels `UnobservedComponents` with a Kalman filter — useful when the trend itself is changing, but requires longer series and is harder to explain.

## Structural breaks

The ruptures PELT algorithm runs on the log-transformed series. If a break is detected, the library warns and refits piecewise. The trend rate from the final segment is what gets reported — this is the defensible choice for projection, since you are projecting from the current regime.

Pass `changepoints=[8, 20]` to impose known breaks (e.g. 2020 Q1, 2025 Q1) rather than using auto-detection.

## ONS series catalogue

| Key | ONS code | Description |
|-----|----------|-------------|
| `motor_repair` | HPTH | SPPI G4520 Maintenance and repair of motor vehicles (2015=100) |
| `motor_insurance_cpi` | L7JE | CPI 12.5.4.1 Motor vehicle insurance |
| `vehicle_maintenance_rpi` | CZEA | RPI Maintenance of motor vehicles |
| `building_maintenance` | D7DO | CPI 04.3.2 Services for maintenance and repair of dwellings |
| `household_maintenance_weights` | CJVD | CPI Weights 04.3 Maintenance and repair |

For household severity, use D7DO as a free proxy. BCIS is more appropriate for reinstatement cost trend — load it via `ExternalIndex.from_csv()`.

## Inputs

Aggregate accident-period data. Minimum viable: 6 quarters. Recommended: 12–20 quarters.

| Column | Description |
|--------|-------------|
| `periods` | Quarter identifiers, e.g. `'2020Q1'` |
| `claim_counts` | Number of claims in the period |
| `earned_exposure` | Earned exposure (vehicle-years, policy-years, etc.) |
| `total_paid` | Total paid claims |

Both pandas and Polars DataFrames/Series are accepted as inputs. All outputs are Polars.

## Installation

```bash
pip install insurance-trend
```

## Dependencies

pandas, numpy, statsmodels, scipy, ruptures, matplotlib, requests, polars.

No scikit-learn, TensorFlow, or PyTorch.

## Mix adjustment

V1 does not include mix adjustment. If your portfolio composition has shifted (more young drivers, different vehicle types), apparent trends may reflect mix change rather than genuine inflation. Pre-process to mix-adjusted frequency/severity before passing to the fitters if this matters for your use case.

## Scope

This library is for pricing trend — forward projection of aggregate accident-period data. It is not a reserving tool. Use `chainladder-python` for triangle development to ultimate; use `insurance-trend` for what comes after.
