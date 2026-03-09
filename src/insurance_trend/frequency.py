"""FrequencyTrendFitter — log-linear trend fitting for insurance claim frequency.

Frequency is defined as claims per unit of exposure (e.g. claims per vehicle-year).
The fitter accepts aggregate accident-period data and returns a TrendResult.

Primary method: log-linear OLS with optional quarterly seasonal dummies.
Alternative method: local linear trend via statsmodels UnobservedComponents.
Structural breaks: optional automatic detection via ruptures PELT.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import polars as pl

from ._utils import (
    PandasOrPolars,
    annual_trend_rate,
    quarter_dummies,
    safe_log,
    to_numpy,
    validate_lengths,
)
from .breaks import detect_breakpoints, split_segments
from .result import TrendResult


class FrequencyTrendFitter:
    """Log-linear trend fitter for insurance claim frequency.

    Frequency is computed as ``claim_counts / earned_exposure`` for each period.
    The model fitted is:

        log(freq_t) = alpha + beta*t + sum(gamma_k * seasonal_k) + epsilon_t

    The annual trend rate is ``exp(beta * periods_per_year) - 1``.

    Parameters
    ----------
    periods:
        Ordered sequence of period labels (e.g. ``['2020Q1', '2020Q2', ...]``).
        Used only for labelling outputs. Must be the same length as the other arrays.
    claim_counts:
        Number of claims in each period. Must be strictly positive.
    earned_exposure:
        Earned exposure in each period (e.g. vehicle-years). Must be strictly
        positive.
    weights:
        Optional observation weights for WLS. Higher values increase the influence
        of the corresponding observation. A common choice is to weight recent
        periods more heavily, e.g. ``[0.5, 0.6, ..., 1.0]``. If None, all
        observations receive equal weight (OLS).
    periods_per_year:
        Number of periods per year. Use ``4`` for quarterly data (the default) and
        ``12`` for monthly data. This affects the conversion of the per-period slope
        to an annual trend rate.

    Examples
    --------
    >>> import polars as pl
    >>> from insurance_trend import FrequencyTrendFitter
    >>>
    >>> fitter = FrequencyTrendFitter(
    ...     periods=['2020Q1', '2020Q2', '2021Q1', '2021Q2',
    ...              '2022Q1', '2022Q2', '2023Q1', '2023Q2'],
    ...     claim_counts=[110, 115, 108, 112, 105, 109, 102, 107],
    ...     earned_exposure=[1000, 1010, 1005, 1008, 1002, 1006, 1001, 1004],
    ... )
    >>> result = fitter.fit()
    >>> print(result.trend_rate)
    """

    def __init__(
        self,
        periods: PandasOrPolars,
        claim_counts: PandasOrPolars,
        earned_exposure: PandasOrPolars,
        weights: Optional[PandasOrPolars] = None,
        periods_per_year: int = 4,
    ) -> None:
        validate_lengths(
            claim_counts=claim_counts,
            earned_exposure=earned_exposure,
        )
        self._periods_raw = periods
        self._claim_counts = to_numpy(claim_counts, "claim_counts")
        self._earned_exposure = to_numpy(earned_exposure, "earned_exposure")
        self._weights = to_numpy(weights, "weights") if weights is not None else None
        self._periods_per_year = periods_per_year

        # Compute frequency
        if np.any(self._earned_exposure <= 0):
            raise ValueError("earned_exposure must be strictly positive in all periods.")
        self._frequency = self._claim_counts / self._earned_exposure

    @property
    def frequency(self) -> np.ndarray:
        """Computed frequency series (claims / exposure)."""
        return self._frequency.copy()

    def fit(
        self,
        method: str = "log_linear",
        changepoints: Optional[List[int]] = None,
        detect_breaks: bool = True,
        seasonal: bool = True,
        n_bootstrap: int = 1000,
        projection_periods: int = 8,
        ci_level: float = 0.95,
        penalty: float = 3.0,
    ) -> TrendResult:
        """Fit the frequency trend model.

        Parameters
        ----------
        method:
            ``'log_linear'`` (default) for OLS on log-frequency, or
            ``'local_linear_trend'`` for the statsmodels UnobservedComponents
            state-space model.
        changepoints:
            Optional list of 0-based integer indices at which structural breaks
            are imposed. If provided, overrides automatic detection. Fitting uses
            a piecewise log-linear model with a separate slope per segment; the
            trend rate from the final segment is reported.
        detect_breaks:
            If ``True`` (default) and ``changepoints`` is not given, run the
            ruptures PELT algorithm to auto-detect structural breaks and warn
            the user if any are found.
        seasonal:
            If ``True`` (default) and ``method='log_linear'``, include quarterly
            seasonal dummy variables (Q1, Q2, Q3 with Q4 as base) in the model.
            Has no effect for ``'local_linear_trend'``.
        n_bootstrap:
            Number of parametric bootstrap replicates for CI estimation. Default
            is 1000. Reduce to 200 for faster (less precise) CIs.
        projection_periods:
            Number of periods to project forward. Default is 8 (two years of
            quarterly data).
        ci_level:
            Confidence level for the bootstrap CI. Default is 0.95 (95 %).
        penalty:
            Penalty parameter for the ruptures PELT algorithm. Only used when
            ``detect_breaks=True``. Higher values suppress false positives.

        Returns
        -------
        TrendResult
        """
        if method == "log_linear":
            return self._fit_log_linear(
                changepoints=changepoints,
                detect_breaks=detect_breaks,
                seasonal=seasonal,
                n_bootstrap=n_bootstrap,
                projection_periods=projection_periods,
                ci_level=ci_level,
                penalty=penalty,
            )
        elif method == "local_linear_trend":
            return self._fit_local_linear(
                n_bootstrap=n_bootstrap,
                projection_periods=projection_periods,
                ci_level=ci_level,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'log_linear' or 'local_linear_trend'."
            )

    # ------------------------------------------------------------------ #
    # Private: log-linear OLS / piecewise
    # ------------------------------------------------------------------ #

    def _fit_log_linear(
        self,
        changepoints: Optional[List[int]],
        detect_breaks: bool,
        seasonal: bool,
        n_bootstrap: int,
        projection_periods: int,
        ci_level: float,
        penalty: float,
    ) -> TrendResult:
        freq = self._frequency
        log_freq = safe_log(freq, "frequency")
        n = len(log_freq)
        t = np.arange(n, dtype=float)

        # Resolve changepoints
        if changepoints is not None:
            breaks = list(changepoints)
        elif detect_breaks:
            breaks = detect_breakpoints(log_freq, penalty=penalty)
            if breaks:
                warnings.warn(
                    f"Structural breaks detected at indices {breaks}. "
                    "Consider reviewing these dates for actuarial plausibility "
                    "(e.g. COVID lockdown, regulatory changes). "
                    "Pass changepoints=[] to suppress piecewise fitting.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            breaks = []

        if breaks:
            trend_rate, beta_last, fitted_log, r_sq = self._fit_piecewise(
                t, log_freq, breaks, seasonal
            )
            method_str = "piecewise"
        else:
            trend_rate, beta_last, fitted_log, r_sq = self._fit_ols_segment(
                t, log_freq, seasonal, self._weights
            )
            method_str = "log_linear"

        fitted_vals = np.exp(fitted_log)
        residuals = freq / fitted_vals - 1.0

        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(
            t, log_freq, breaks, seasonal, n_bootstrap, ci_level
        )

        # Projection
        projection = _project_forward(
            fitted_vals[-1], beta_last, self._periods_per_year, projection_periods, ci_lower, ci_upper
        )

        periods_series = _periods_to_series(self._periods_raw, n)

        return TrendResult(
            trend_rate=trend_rate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method=method_str,
            fitted_values=pl.Series("fitted", fitted_vals),
            residuals=pl.Series("residuals", residuals),
            changepoints=breaks,
            projection=projection,
            r_squared=r_sq,
            actuals=pl.Series("actuals", freq),
            periods=periods_series,
            n_bootstrap=n_bootstrap,
            periods_per_year=self._periods_per_year,
        )

    def _fit_ols_segment(
        self,
        t: np.ndarray,
        log_y: np.ndarray,
        seasonal: bool,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[float, float, np.ndarray, float]:
        """Fit a single log-linear OLS segment. Returns (trend_rate, beta, fitted_log, r_sq)."""
        import statsmodels.api as sm

        X = _build_design_matrix(t, seasonal, self._periods_per_year)
        if weights is not None:
            w = weights[: len(t)]
            model = sm.WLS(log_y, X, weights=w)
        else:
            model = sm.OLS(log_y, X)
        res = model.fit()
        beta = float(res.params[1])  # slope
        trend_rate = annual_trend_rate(beta, self._periods_per_year)
        fitted_log = res.fittedvalues
        r_sq = float(res.rsquared)
        return trend_rate, beta, fitted_log, r_sq

    def _fit_piecewise(
        self,
        t: np.ndarray,
        log_y: np.ndarray,
        breaks: List[int],
        seasonal: bool,
    ) -> tuple[float, float, np.ndarray, float]:
        """Fit piecewise log-linear; return (trend_rate, last_beta, full_fitted, last_r_sq)."""
        segments = split_segments(t, log_y, breaks)
        fitted_full = np.empty_like(log_y)
        last_beta = 0.0
        last_r_sq = 0.0
        for seg_t, seg_y in segments:
            idx = seg_t.astype(int)
            X = _build_design_matrix(np.arange(len(seg_t), dtype=float), seasonal, self._periods_per_year)
            import statsmodels.api as sm
            res = sm.OLS(seg_y, X).fit()
            fitted_full[idx] = res.fittedvalues
            last_beta = float(res.params[1])
            last_r_sq = float(res.rsquared)
        trend_rate = annual_trend_rate(last_beta, self._periods_per_year)
        return trend_rate, last_beta, fitted_full, last_r_sq

    def _bootstrap_ci(
        self,
        t: np.ndarray,
        log_y: np.ndarray,
        breaks: List[int],
        seasonal: bool,
        n_bootstrap: int,
        ci_level: float,
    ) -> tuple[float, float]:
        """Parametric bootstrap: resample residuals, refit, collect slope distribution."""
        import statsmodels.api as sm

        X = _build_design_matrix(t, seasonal, self._periods_per_year)
        res = sm.OLS(log_y, X).fit()
        residuals = log_y - res.fittedvalues
        fitted = res.fittedvalues
        rng = np.random.default_rng(42)
        boot_rates = []
        for _ in range(n_bootstrap):
            boot_log_y = fitted + rng.choice(residuals, size=len(residuals), replace=True)
            if breaks:
                segs = split_segments(t, boot_log_y, breaks)
                last_beta = 0.0
                for seg_t, seg_y in segs:
                    X_s = _build_design_matrix(
                        np.arange(len(seg_t), dtype=float), seasonal, self._periods_per_year
                    )
                    r = sm.OLS(seg_y, X_s).fit()
                    last_beta = float(r.params[1])
                boot_rates.append(annual_trend_rate(last_beta, self._periods_per_year))
            else:
                r = sm.OLS(boot_log_y, X).fit()
                boot_rates.append(annual_trend_rate(float(r.params[1]), self._periods_per_year))

        alpha = (1.0 - ci_level) / 2.0
        lower = float(np.quantile(boot_rates, alpha))
        upper = float(np.quantile(boot_rates, 1.0 - alpha))
        return lower, upper

    # ------------------------------------------------------------------ #
    # Private: local linear trend (state-space)
    # ------------------------------------------------------------------ #

    def _fit_local_linear(
        self,
        n_bootstrap: int,
        projection_periods: int,
        ci_level: float,
    ) -> TrendResult:
        """Fit local linear trend via statsmodels UnobservedComponents."""
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        freq = self._frequency
        log_freq = safe_log(freq, "frequency")
        n = len(log_freq)

        model = UnobservedComponents(log_freq, level="local linear trend")
        try:
            res = model.fit(disp=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Local linear trend fitting failed: {exc}. "
                "Consider using method='log_linear' instead."
            ) from exc

        fitted_log = res.fittedvalues
        smoothed_slope = res.smoother_results.smoothed_state[1, :]
        # Last slope estimate as the trend; convert from per-period to annual
        last_beta = float(np.mean(smoothed_slope[-4:]))
        trend_rate = annual_trend_rate(last_beta, self._periods_per_year)

        fitted_vals = np.exp(np.asarray(fitted_log))
        residuals = freq / fitted_vals - 1.0

        # SS model R-squared approximation
        ss_res = np.sum((log_freq - fitted_log) ** 2)
        ss_tot = np.sum((log_freq - log_freq.mean()) ** 2)
        r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Bootstrap CI by simulation
        ci_lower, ci_upper = _local_linear_bootstrap_ci(
            log_freq, n_bootstrap, ci_level, self._periods_per_year
        )

        projection = _project_forward(
            fitted_vals[-1], last_beta, self._periods_per_year, projection_periods, ci_lower, ci_upper
        )

        periods_series = _periods_to_series(self._periods_raw, n)

        return TrendResult(
            trend_rate=trend_rate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="local_linear_trend",
            fitted_values=pl.Series("fitted", fitted_vals),
            residuals=pl.Series("residuals", residuals),
            changepoints=[],
            projection=projection,
            r_squared=r_sq,
            actuals=pl.Series("actuals", freq),
            periods=periods_series,
            n_bootstrap=n_bootstrap,
            periods_per_year=self._periods_per_year,
        )

    def summary(self) -> str:
        """Return a brief string summary of the fitter configuration."""
        n = len(self._frequency)
        freq_range = f"{self._frequency.min():.4f} – {self._frequency.max():.4f}"
        return (
            f"FrequencyTrendFitter: {n} periods, "
            f"frequency range {freq_range}, "
            f"periods_per_year={self._periods_per_year}"
        )


# ------------------------------------------------------------------ #
# Module-level helpers shared by frequency and severity modules
# ------------------------------------------------------------------ #

def _build_design_matrix(
    t: np.ndarray,
    seasonal: bool,
    periods_per_year: int,
) -> np.ndarray:
    """Build OLS design matrix: [intercept, trend, seasonal_dummies]."""
    import statsmodels.api as sm

    X = sm.add_constant(t)
    if seasonal and periods_per_year == 4:
        dummies = quarter_dummies(len(t), t.astype(int))
        X = np.column_stack([X, dummies])
    return X


def _project_forward(
    last_fitted: float,
    beta: float,
    periods_per_year: int,
    n_periods: int,
    ci_lower: float,
    ci_upper: float,
) -> pl.DataFrame:
    """Build a forward projection DataFrame.

    Uses the last fitted value as the anchor and compounds the trend forward.
    The CI columns are scaled from the annual CI using the per-period rate.
    """
    if n_periods <= 0:
        return pl.DataFrame({"period": [], "point": [], "lower": [], "upper": []})

    per_period_rate = float(np.exp(beta) - 1.0)
    annual_lower = ci_lower
    annual_upper = ci_upper
    per_period_lower = float((1.0 + annual_lower) ** (1.0 / periods_per_year) - 1.0)
    per_period_upper = float((1.0 + annual_upper) ** (1.0 / periods_per_year) - 1.0)

    points = []
    lowers = []
    uppers = []
    for i in range(1, n_periods + 1):
        points.append(last_fitted * (1.0 + per_period_rate) ** i)
        lowers.append(last_fitted * (1.0 + per_period_lower) ** i)
        uppers.append(last_fitted * (1.0 + per_period_upper) ** i)

    return pl.DataFrame(
        {
            "period": list(range(1, n_periods + 1)),
            "point": points,
            "lower": lowers,
            "upper": uppers,
        }
    )


def _periods_to_series(periods_raw: PandasOrPolars, n: int) -> pl.Series:
    """Convert raw periods input to a Polars Series of strings."""
    try:
        if hasattr(periods_raw, "to_list"):
            vals = [str(v) for v in periods_raw.to_list()]
        else:
            vals = [str(v) for v in periods_raw]
        return pl.Series("periods", vals[:n])
    except Exception:  # noqa: BLE001
        return pl.Series("periods", [str(i) for i in range(n)])


def _local_linear_bootstrap_ci(
    log_y: np.ndarray,
    n_bootstrap: int,
    ci_level: float,
    periods_per_year: int,
) -> tuple[float, float]:
    """Bootstrap CI for local linear trend by refitting on resampled residuals."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    model = UnobservedComponents(log_y, level="local linear trend")
    try:
        res_base = model.fit(disp=False)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0

    fitted = np.asarray(res_base.fittedvalues)
    residuals = log_y - fitted
    rng = np.random.default_rng(42)
    boot_rates = []
    for _ in range(min(n_bootstrap, 200)):  # cap at 200 for local linear — each fit is slower
        boot_y = fitted + rng.choice(residuals, size=len(residuals), replace=True)
        try:
            r = UnobservedComponents(boot_y, level="local linear trend").fit(disp=False)
            slope = r.smoother_results.smoothed_state[1, :]
            last_beta = float(np.mean(slope[-4:]))
            boot_rates.append(annual_trend_rate(last_beta, periods_per_year))
        except Exception:  # noqa: BLE001
            pass

    if not boot_rates:
        return 0.0, 0.0

    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(boot_rates, alpha)), float(np.quantile(boot_rates, 1.0 - alpha))
