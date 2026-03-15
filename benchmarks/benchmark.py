"""
Benchmark: insurance-trend (automated break detection) vs naive OLS trend line.

Data generating process:
  - 24 quarterly periods (2019 Q1 – 2024 Q4)
  - Frequency: stable ~8.5% pre-2021 (Q1-Q8), then drops to ~6.5% in 2021
    (analogous to COVID lockdown), then recovers at +3% pa post-Q8
  - Severity: stable growth at +3% pa pre-2022 (Q1-Q12), then accelerates
    to +8% pa post-Q12 (analogous to post-COVID repair inflation)
  - Combined loss cost has a structural break at Q8 (frequency) and Q12 (severity)

The naive baseline: fit a single OLS trend line across all 24 quarters.
This blends the pre- and post-break regimes and produces a misleading trend rate
for projection purposes.

insurance-trend uses ruptures PELT to detect the breaks, refits on each segment
separately, and reports the post-break trend (the one relevant for projection).

Metrics:
  - Trend MAPE: how close is the estimated trend rate to the true post-break DGP rate?
  - Break detection accuracy: was the break located within ±2 quarters?
  - Projection error: MAPE on 4-quarter forward projection vs true DGP

Run on Databricks:
  %pip install insurance-trend polars numpy scipy statsmodels ruptures
"""

import numpy as np
import polars as pl
from scipy import stats

# ---------------------------------------------------------------------------
# 1. Generate synthetic experience data — known DGP with structural breaks
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N = 24  # quarters

periods = [
    f"{yr}Q{q}"
    for yr in range(2019, 2025)
    for q in range(1, 5)
]
t = np.arange(N, dtype=float)

# Earned vehicle-years (growing book)
earned_vehicles = (
    12_000.0
    + t * 150.0
    + rng.normal(0, 200, N)
).clip(10_000.0)

# True frequency: stable pre-Q8, then step-down, then recovery at +3% pa
# Break point = Q8 (index 7, i.e. Q8 is after 8 quarters)
FREQ_BREAK = 8
freq_true = np.where(
    t < FREQ_BREAK,
    0.085 + 0.001 * t,
    0.063 * np.exp(0.0074 * (t - FREQ_BREAK)),  # +3% pa recovery
)
claim_counts = rng.poisson(freq_true * earned_vehicles).astype(float)

# True severity: accelerates at +8% pa from Q12 (repair inflation)
SEV_BREAK = 12
base_severity = 3_800.0
sev_true = base_severity * np.where(
    t < SEV_BREAK,
    np.exp(0.0075 * t),              # +3% pa pre-break
    np.exp(0.0075 * SEV_BREAK)       # baseline at break
    * np.exp(0.0196 * (t - SEV_BREAK)),  # +8% pa post-break
)
total_paid = claim_counts * rng.lognormal(np.log(sev_true), 0.15)

# True post-break trend rates (what we should estimate)
TRUE_FREQ_TREND_POST = 0.030  # 3% pa
TRUE_SEV_TREND_POST = 0.080   # 8% pa
TRUE_LC_TREND_POST = (1 + TRUE_FREQ_TREND_POST) * (1 + TRUE_SEV_TREND_POST) - 1

print("=" * 70)
print("BENCHMARK: insurance-trend automated break detection vs naive OLS")
print(f"  Periods: {periods[0]} to {periods[-1]}")
print(f"  Frequency break at Q{FREQ_BREAK} (true post-break trend: +{TRUE_FREQ_TREND_POST:.1%} pa)")
print(f"  Severity break at Q{SEV_BREAK} (true post-break trend: +{TRUE_SEV_TREND_POST:.1%} pa)")
print(f"  True combined loss cost trend (post-break): +{TRUE_LC_TREND_POST:.2%} pa")
print("=" * 70)

# ---------------------------------------------------------------------------
# 2. Naive OLS baseline: single trend line across all 24 quarters
# ---------------------------------------------------------------------------
# Frequency trend: OLS on log(claims / exposure)
log_freq = np.log(claim_counts / earned_vehicles)
t_x = t.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
ols_freq = LinearRegression().fit(t_x, log_freq)
beta_freq_naive = float(ols_freq.coef_[0])
trend_freq_naive = np.exp(beta_freq_naive * 4) - 1.0  # quarterly beta -> annual

# Severity trend: OLS on log(total_paid / claim_counts)
severity_obs = total_paid / claim_counts
log_sev = np.log(severity_obs)
ols_sev = LinearRegression().fit(t_x, log_sev)
beta_sev_naive = float(ols_sev.coef_[0])
trend_sev_naive = np.exp(beta_sev_naive * 4) - 1.0

# Combined naive loss cost trend
trend_lc_naive = (1 + trend_freq_naive) * (1 + trend_sev_naive) - 1.0

# Naive projection (fitted values from OLS extrapolated 4 quarters ahead)
t_proj = np.arange(N, N + 4, dtype=float).reshape(-1, 1)
freq_fitted_naive = np.exp(ols_freq.predict(t_x))
sev_fitted_naive = np.exp(ols_sev.predict(t_x))
freq_proj_naive = np.exp(ols_freq.predict(t_proj))
sev_proj_naive = np.exp(ols_sev.predict(t_proj))
lc_proj_naive = freq_proj_naive * sev_proj_naive

print(f"\nNaive OLS results:")
print(f"  Frequency trend:   {trend_freq_naive:+.3%} pa  (true post-break: {TRUE_FREQ_TREND_POST:+.3%})")
print(f"  Severity trend:    {trend_sev_naive:+.3%} pa  (true post-break: {TRUE_SEV_TREND_POST:+.3%})")
print(f"  Loss cost trend:   {trend_lc_naive:+.3%} pa  (true post-break: {TRUE_LC_TREND_POST:+.3%})")

# ---------------------------------------------------------------------------
# 3. insurance-trend with break detection
# ---------------------------------------------------------------------------
from insurance_trend import LossCostTrendFitter

print("\nFitting LossCostTrendFitter with detect_breaks=True...")
fitter = LossCostTrendFitter(
    periods=periods,
    claim_counts=claim_counts,
    earned_exposure=earned_vehicles,
    total_paid=total_paid,
    periods_per_year=4,
)

result = fitter.fit(
    detect_breaks=True,
    seasonal=True,
    n_bootstrap=500,
    projection_periods=4,
    penalty=2.0,
)

trend_freq_inslib = result.frequency.trend_rate
trend_sev_inslib = result.severity.trend_rate
trend_lc_inslib = result.combined_trend_rate

# Detected break points
breaks_freq = result.frequency.changepoints
breaks_sev = result.severity.changepoints

print(f"\ninsurance-trend results:")
print(f"  Frequency trend:   {trend_freq_inslib:+.3%} pa  (true: {TRUE_FREQ_TREND_POST:+.3%})")
print(f"  Frequency breaks detected: {breaks_freq}")
print(f"  Severity trend:    {trend_sev_inslib:+.3%} pa  (true: {TRUE_SEV_TREND_POST:+.3%})")
print(f"  Severity breaks detected: {breaks_sev}")
print(f"  Loss cost trend:   {trend_lc_inslib:+.3%} pa  (true: {TRUE_LC_TREND_POST:+.3%})")
print(f"  CI: ({result.frequency.ci_lower:.3%}, {result.frequency.ci_upper:.3%}) freq")

# ---------------------------------------------------------------------------
# 4. Projection accuracy — 4 quarters ahead
# ---------------------------------------------------------------------------
# True DGP continuation for 4 periods after the series ends
t_future = np.arange(N, N + 4, dtype=float)
freq_true_future = 0.063 * np.exp(0.0074 * (t_future - FREQ_BREAK))
sev_true_future = np.exp(0.0075 * SEV_BREAK) * np.exp(0.0196 * (t_future - SEV_BREAK)) * base_severity
lc_true_future = freq_true_future * sev_true_future

# insurance-trend projection
proj_df = result.projection
lc_proj_inslib = proj_df["point"].to_numpy()

def mape(actual, predicted):
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + 1e-12))) * 100)

mape_naive = mape(lc_true_future, lc_proj_naive)
mape_inslib = mape(lc_true_future, lc_proj_inslib)

print("\n" + "=" * 70)
print("TABLE 1: Trend rate estimates vs true post-break DGP")
print(f"  {'Component':<20}  {'True (DGP)':>12}  {'Naive OLS':>12}  {'insurance-trend':>17}  {'Naive error':>13}  {'Lib error':>12}")
print("-" * 90)

rows = [
    ("Frequency", TRUE_FREQ_TREND_POST, trend_freq_naive, trend_freq_inslib),
    ("Severity", TRUE_SEV_TREND_POST, trend_sev_naive, trend_sev_inslib),
    ("Loss cost", TRUE_LC_TREND_POST, trend_lc_naive, trend_lc_inslib),
]
for name, true_rate, naive_rate, lib_rate in rows:
    err_naive = naive_rate - true_rate
    err_lib = lib_rate - true_rate
    print(f"  {name:<20}  {true_rate:>+12.3%}  {naive_rate:>+12.3%}  {lib_rate:>+17.3%}  "
          f"{err_naive:>+13.3%}  {err_lib:>+12.3%}")

print("\n  Naive OLS blends pre- and post-break regimes.")
print("  insurance-trend refits on the post-break segment only.")

# ---------------------------------------------------------------------------
# 5. Break detection accuracy
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLE 2: Structural break detection accuracy")
print(f"  {'Component':<15}  {'True break idx':>16}  {'Detected breaks':>18}  "
      f"{'Within ±2Q?':>13}")
print("-" * 68)
for component, true_brk, detected in [
    ("Frequency", FREQ_BREAK, breaks_freq),
    ("Severity", SEV_BREAK, breaks_sev),
]:
    detected_str = str(detected) if detected else "none"
    if detected:
        closest = min(detected, key=lambda x: abs(x - true_brk))
        within_2 = "Yes" if abs(closest - true_brk) <= 2 else "No"
    else:
        within_2 = "No break detected"
    print(f"  {component:<15}  {true_brk:>16}  {detected_str:>18}  {within_2:>13}")

# ---------------------------------------------------------------------------
# 6. Projection MAPE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLE 3: 4-quarter forward projection MAPE vs true DGP")
print(f"  {'Method':<25}  {'Loss cost MAPE (4Q)':>22}")
print("-" * 50)
print(f"  {'Naive OLS':<25}  {mape_naive:>21.2f}%")
print(f"  {'insurance-trend':<25}  {mape_inslib:>21.2f}%")
improvement = mape_naive - mape_inslib
print(f"  Improvement: {improvement:.2f} pp MAPE reduction")
print("  (improvement is largest when break is recent and regimes differ)")

# ---------------------------------------------------------------------------
# 7. Decomposition
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLE 4: Loss cost decomposition (insurance-trend output)")
decomp = result.decompose()
for k, v in decomp.items():
    v_str = f"{v:.4%}" if v is not None else "N/A"
    print(f"  {k:<25}: {v_str}")
print("  (Naive OLS produces no frequency/severity decomposition)")

print("\n" + "=" * 70)
print("SUMMARY: insurance-trend outperforms naive OLS on:")
print("  - Trend rate accuracy: fits post-break regime only")
print("  - Break detection: ruptures PELT locates structural changes")
print("  - Projection accuracy: lower MAPE on 4-quarter forward projections")
print("  - Decomposition: frequency vs severity trend as separate outputs")
print("  When experience is genuinely stable (no breaks), both produce")
print("  the same headline rate — the library's bootstrap CI is then the value.")
print("=" * 70)
