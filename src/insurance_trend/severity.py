"""SeverityTrendFitter — log-linear trend fitting for insurance claim severity.

Severity is defined as total paid / claim count for each period.

This fitter supports optional external index deflation (e.g. ONS HPTH for
motor repair costs), allowing decomposition into:

    Total severity trend = economic inflation (from index) + superimposed inflation

The superimposed inflation represents the part of claims cost growth that is
not explained by general price indices — structural factors such as repair
complexity, ADAS recalibration, or litigation trends.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import polars as pl

from ._utils import (
    PandasOrPolars,
    annual_trend_rate,
    safe_log,
    to_numpy,
    validate_lengths,
)
from .breaks import detect_breakpoints, split_segments
from .frequency import (
    _build_design_matrix,
    _local_linear_bootstrap_ci,
    _periods_to_series,
    _project_forward,
)
from .result import TrendResult


class SeverityTrendFitter:
    """Log-linear trend fitter for insurance claim severity.

    Severity is computed as ``total_paid / claim_counts`` for each period.

    When an ``external_index`` is provided, severity is deflated by the index
    *before* fitting. The residual trend after deflation is the superimposed
    inflation component.

    Parameters
    ----------
    periods:
        Ordered sequence of period labels.
    total_paid:
        Total paid claims for each period. Must be strictly positive.
    claim_counts:
        Number of claims in each period. Must be strictly positive.
    external_index:
        Optional external inflation index aligned to the same periods. Values
        are re-based to the first observation (index[0] = 1.0) internally.
        Typical choice: ONS HPTH for motor repair, ONS D7DO for household.
    weights:
        Optional observation weights for WLS. If None, equal weights (OLS).
    periods_per_year:
        Number of periods per year. Use ``4`` for quarterly data (default).

    Examples
    --------
    >>> from insurance_trend import SeverityTrendFitter, ExternalIndex
    >>>
    >>> idx = ExternalIndex.from_ons('HPTH')   # requires network access
    >>> fitter = SeverityTrendFitter(
    ...     periods=['2020Q1', ..., '2023Q4'],
    ...     total_paid=[...],
    ...     claim_counts=[...],
    ...     external_index=idx,
    ... )
    >>> result = fitter.fit()
    >>> print(fitter.superimposed_inflation())
    """

    def __init__(
        self,
        periods: PandasOrPolars,
        total_paid: PandasOrPolars,
        claim_counts: PandasOrPolars,
        external_index: Optional[PandasOrPolars] = None,
        weights: Optional[PandasOrPolars] = None,
        periods_per_year: int = 4,
    ) -> None:
        validate_lengths(total_paid=total_paid, claim_counts=claim_counts)
        self._periods_raw = periods
        self._total_paid = to_numpy(total_paid, "total_paid")
        self._claim_counts = to_numpy(claim_counts, "claim_counts")
        self._weights = to_numpy(weights, "weights") if weights is not None else None
        self._periods_per_year = periods_per_year
        self._external_index: Optional[np.ndarray] = None
        self._index_trend_rate: Optional[float] = None
        self._fitted_result: Optional[TrendResult] = None

        if np.any(self._claim_counts <= 0):
            raise ValueError("claim_counts must be strictly positive in all periods.")
        if np.any(self._total_paid <= 0):
            raise ValueError("total_paid must be strictly positive in all periods.")

        self._severity = self._total_paid / self._claim_counts

        if external_index is not None:
            idx = to_numpy(external_index, "external_index")
            if len(idx) < len(self._severity):
                raise ValueError(
                    f"external_index length ({len(idx)}) is shorter than the "
                    f"number of periods ({len(self._severity)}). Trim or align the index."
                )
            # Use the subset matching the severity length
            idx = idx[: len(self._severity)]
            # Re-base to first observation
            if idx[0] == 0:
                raise ValueError("external_index first value must be non-zero for re-basing.")
            self._external_index = idx / idx[0]

    @property
    def severity(self) -> np.ndarray:
        """Raw (non-deflated) severity series."""
        return self._severity.copy()

    @property
    def deflated_severity(self) -> Optional[np.ndarray]:
        """Severity deflated by the external index, or None if no index provided."""
        if self._external_index is None:
            return None
        return self._severity / self._external_index

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
        """Fit the severity trend model.

        When an external index was provided at construction, the fit is performed
        on the deflated severity. Call :meth:`superimposed_inflation` after fitting
        to get the residual trend.

        Parameters
        ----------
        method:
            ``'log_linear'`` (default) or ``'local_linear_trend'``.
        changepoints:
            Explicit structural break indices (0-based). Overrides auto-detection.
        detect_breaks:
            Auto-detect structural breaks via ruptures if ``changepoints`` is None.
        seasonal:
            Include quarterly seasonal dummies (log-linear method only).
        n_bootstrap:
            Number of bootstrap replicates for CI estimation.
        projection_periods:
            Number of periods to project forward.
        ci_level:
            Confidence level for bootstrap CI.
        penalty:
            Penalty for ruptures PELT (higher = fewer breaks).

        Returns
        -------
        TrendResult
            Trend fitted to (optionally deflated) severity.
        """
        # Compute index trend rate for superimposed inflation calculation
        if self._external_index is not None:
            self._index_trend_rate = self._compute_index_trend_rate()

        series_to_fit = self.deflated_severity if self._external_index is not None else self._severity

        if method == "log_linear":
            result = self._fit_log_linear(
                series=series_to_fit,
                changepoints=changepoints,
                detect_breaks=detect_breaks,
                seasonal=seasonal,
                n_bootstrap=n_bootstrap,
                projection_periods=projection_periods,
                ci_level=ci_level,
                penalty=penalty,
            )
        elif method == "local_linear_trend":
            result = self._fit_local_linear(
                series=series_to_fit,
                n_bootstrap=n_bootstrap,
                projection_periods=projection_periods,
                ci_level=ci_level,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'log_linear' or 'local_linear_trend'."
            )

        self._fitted_result = result
        return result

    def superimposed_inflation(self) -> Optional[float]:
        """Return the superimposed inflation rate (annual).

        When an external index is provided, the fit is performed on
        ``severity / index``. The trend of that deflated series IS the
        superimposed inflation — the portion of severity growth not explained
        by the external economic index.

        Returns ``None`` if no external index was provided or :meth:`fit` has
        not been called yet.

        Returns
        -------
        float or None
        """
        if self._fitted_result is None or self._index_trend_rate is None:
            return None
        # The fit was performed on severity / index, so _fitted_result.trend_rate
        # already IS the superimposed inflation. No further deflation is needed.
        return float(self._fitted_result.trend_rate)

    def _compute_index_trend_rate(self) -> float:
        """Fit a log-linear trend to the external index and return the annual rate."""
        import statsmodels.api as sm

        idx = self._external_index
        log_idx = safe_log(idx, "external_index")
        n = len(log_idx)
        t = np.arange(n, dtype=float)
        X = sm.add_constant(t)
        res = sm.OLS(log_idx, X).fit()
        beta = float(res.params[1])
        return annual_trend_rate(beta, self._periods_per_year)

    def _fit_log_linear(
        self,
        series: np.ndarray,
        changepoints: Optional[List[int]],
        detect_breaks: bool,
        seasonal: bool,
        n_bootstrap: int,
        projection_periods: int,
        ci_level: float,
        penalty: float,
    ) -> TrendResult:
        log_y = safe_log(series, "severity")
        n = len(log_y)
        t = np.arange(n, dtype=float)

        if changepoints is not None:
            breaks = list(changepoints)
        elif detect_breaks:
            breaks = detect_breakpoints(log_y, penalty=penalty)
            if breaks:
                warnings.warn(
                    f"Structural breaks detected at indices {breaks} in severity series. "
                    "Review for COVID-19 suppression, Ogden rate change, or other "
                    "actuarially significant events.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            breaks = []

        if breaks:
            trend_rate, beta_last, fitted_log, r_sq = _fit_piecewise_ols(
                t, log_y, breaks, seasonal, self._periods_per_year
            )
            method_str = "piecewise"
        else:
            trend_rate, beta_last, fitted_log, r_sq = _fit_ols_segment(
                t, log_y, seasonal, self._periods_per_year, self._weights
            )
            method_str = "log_linear"

        fitted_vals = np.exp(fitted_log)
        residuals = series / fitted_vals - 1.0

        ci_lower, ci_upper = _bootstrap_ci(
            t, log_y, breaks, seasonal, self._periods_per_year, n_bootstrap, ci_level
        )

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
            actuals=pl.Series("actuals", series),
            periods=periods_series,
            n_bootstrap=n_bootstrap,
            periods_per_year=self._periods_per_year,
        )

    def _fit_local_linear(
        self,
        series: np.ndarray,
        n_bootstrap: int,
        projection_periods: int,
        ci_level: float,
    ) -> TrendResult:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        log_y = safe_log(series, "severity")
        n = len(log_y)

        model = UnobservedComponents(log_y, level="local linear trend")
        try:
            res = model.fit(disp=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Local linear trend fitting failed: {exc}. "
                "Consider method='log_linear' instead."
            ) from exc

        fitted_log = np.asarray(res.fittedvalues)
        smoothed_slope = res.smoother_results.smoothed_state[1, :]
        last_beta = float(np.mean(smoothed_slope[-4:]))
        trend_rate = annual_trend_rate(last_beta, self._periods_per_year)

        fitted_vals = np.exp(fitted_log)
        residuals = series / fitted_vals - 1.0

        ss_res = np.sum((log_y - fitted_log) ** 2)
        ss_tot = np.sum((log_y - log_y.mean()) ** 2)
        r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        ci_lower, ci_upper = _local_linear_bootstrap_ci(
            log_y, n_bootstrap, ci_level, self._periods_per_year
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
            actuals=pl.Series("actuals", series),
            periods=periods_series,
            n_bootstrap=n_bootstrap,
            periods_per_year=self._periods_per_year,
        )

    def summary(self) -> str:
        """Return a brief text summary of the fitter configuration."""
        n = len(self._severity)
        sev_range = f"{self._severity.min():.2f} – {self._severity.max():.2f}"
        has_idx = self._external_index is not None
        return (
            f"SeverityTrendFitter: {n} periods, "
            f"severity range {sev_range}, "
            f"external_index={'yes' if has_idx else 'no'}, "
            f"periods_per_year={self._periods_per_year}"
        )


# ------------------------------------------------------------------ #
# Module-level helpers (used by both severity and loss_cost)
# ------------------------------------------------------------------ #

def _fit_ols_segment(
    t: np.ndarray,
    log_y: np.ndarray,
    seasonal: bool,
    periods_per_year: int,
    weights: Optional[np.ndarray] = None,
) -> tuple[float, float, np.ndarray, float]:
    """Single-segment log-linear OLS. Returns (trend_rate, beta, fitted_log, r_sq)."""
    import statsmodels.api as sm

    X = _build_design_matrix(t, seasonal, periods_per_year)
    if weights is not None:
        w = weights[: len(t)]
        res = sm.WLS(log_y, X, weights=w).fit()
    else:
        res = sm.OLS(log_y, X).fit()
    beta = float(res.params[1])
    return annual_trend_rate(beta, periods_per_year), beta, np.asarray(res.fittedvalues), float(res.rsquared)


def _fit_piecewise_ols(
    t: np.ndarray,
    log_y: np.ndarray,
    breaks: List[int],
    seasonal: bool,
    periods_per_year: int,
) -> tuple[float, float, np.ndarray, float]:
    """Piecewise log-linear OLS across detected breaks."""
    import statsmodels.api as sm

    segments = split_segments(t, log_y, breaks)
    fitted_full = np.empty_like(log_y)
    last_beta = 0.0
    last_r_sq = 0.0
    for seg_t, seg_y in segments:
        idx = seg_t.astype(int)
        # P0-3 fix: pass seg_t (global time indices) instead of local np.arange,
        # so that quarterly seasonal dummies are assigned correct calendar phase.
        X = _build_design_matrix(seg_t, seasonal, periods_per_year)
        res = sm.OLS(seg_y, X).fit()
        fitted_full[idx] = res.fittedvalues
        last_beta = float(res.params[1])
        last_r_sq = float(res.rsquared)
    return annual_trend_rate(last_beta, periods_per_year), last_beta, fitted_full, last_r_sq


def _bootstrap_ci(
    t: np.ndarray,
    log_y: np.ndarray,
    breaks: List[int],
    seasonal: bool,
    periods_per_year: int,
    n_bootstrap: int,
    ci_level: float,
) -> tuple[float, float]:
    """Parametric bootstrap CI for a log-linear (or piecewise) fit."""
    import statsmodels.api as sm

    # P0-2 fix: compute residuals from the correct model (piecewise when breaks
    # exist, single OLS otherwise). Using full-series OLS residuals in the
    # piecewise case inflates them ~37x and produces absurdly wide CIs.
    if breaks:
        segments = split_segments(t, log_y, breaks)
        fitted_full = np.empty_like(log_y)
        for seg_t, seg_y in segments:
            idx = seg_t.astype(int)
            # P0-3 fix applied here too: use seg_t for global seasonal phase.
            X_s = _build_design_matrix(seg_t, seasonal, periods_per_year)
            r = sm.OLS(seg_y, X_s).fit()
            fitted_full[idx] = r.fittedvalues
        residuals = log_y - fitted_full
        fitted = fitted_full
    else:
        X = _build_design_matrix(t, seasonal, periods_per_year)
        res = sm.OLS(log_y, X).fit()
        residuals = log_y - np.asarray(res.fittedvalues)
        fitted = np.asarray(res.fittedvalues)

    rng = np.random.default_rng(42)
    boot_rates = []
    for _ in range(n_bootstrap):
        boot_y = fitted + rng.choice(residuals, size=len(residuals), replace=True)
        if breaks:
            segs = split_segments(t, boot_y, breaks)
            last_beta = 0.0
            for seg_t, seg_y in segs:
                # P0-3 fix: use seg_t for global seasonal phase in bootstrap too.
                X_s = _build_design_matrix(seg_t, seasonal, periods_per_year)
                r = sm.OLS(seg_y, X_s).fit()
                last_beta = float(r.params[1])
            boot_rates.append(annual_trend_rate(last_beta, periods_per_year))
        else:
            X = _build_design_matrix(t, seasonal, periods_per_year)
            r = sm.OLS(boot_y, X).fit()
            boot_rates.append(annual_trend_rate(float(r.params[1]), periods_per_year))

    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(boot_rates, alpha)), float(np.quantile(boot_rates, 1.0 - alpha))
