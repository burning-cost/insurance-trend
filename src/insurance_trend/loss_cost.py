"""LossCostTrendFitter — combined frequency × severity trend analysis.

Loss cost = frequency × severity = (claims / exposure) × (paid / claims) = paid / exposure.

This class fits frequency and severity trends separately, then combines them
into a single LossCostTrendResult with the full decomposition and joint projection.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import polars as pl

from ._utils import PandasOrPolars, to_numpy, validate_lengths
from .frequency import FrequencyTrendFitter
from .result import LossCostTrendResult
from .severity import SeverityTrendFitter


class LossCostTrendFitter:
    """Combined frequency × severity loss cost trend fitter.

    This is the primary entry point for most pricing actuary workflows. It
    internally creates a :class:`FrequencyTrendFitter` and a
    :class:`SeverityTrendFitter`, fits them separately, then combines the
    results into a :class:`LossCostTrendResult`.

    Parameters
    ----------
    periods:
        Ordered period labels (e.g. quarter identifiers).
    claim_counts:
        Number of claims per period. Strictly positive.
    earned_exposure:
        Earned exposure per period (e.g. vehicle-years). Strictly positive.
    total_paid:
        Total paid claims per period. Strictly positive.
    external_index:
        Optional external inflation index for severity deflation. See
        :class:`SeverityTrendFitter` for details.
    weights:
        Optional observation weights. Applied to both frequency and severity fits.
    periods_per_year:
        Number of periods per year. Default 4 (quarterly).

    Examples
    --------
    >>> from insurance_trend import LossCostTrendFitter
    >>>
    >>> fitter = LossCostTrendFitter(
    ...     periods=['2020Q1', '2020Q2', '2021Q1', '2021Q2',
    ...              '2022Q1', '2022Q2', '2023Q1', '2023Q2'],
    ...     claim_counts=[110, 115, 108, 112, 105, 109, 102, 107],
    ...     earned_exposure=[1000, 1010, 1005, 1008, 1002, 1006, 1001, 1004],
    ...     total_paid=[550000, 580000, 562000, 573000, 545000, 558000, 538000, 552000],
    ... )
    >>> result = fitter.fit()
    >>> print(result.decompose())
    """

    def __init__(
        self,
        periods: PandasOrPolars,
        claim_counts: PandasOrPolars,
        earned_exposure: PandasOrPolars,
        total_paid: PandasOrPolars,
        external_index: Optional[PandasOrPolars] = None,
        weights: Optional[PandasOrPolars] = None,
        periods_per_year: int = 4,
    ) -> None:
        if periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be a positive integer, got {periods_per_year!r}. "
                "Use 4 for quarterly data or 12 for monthly data."
            )
        validate_lengths(
            claim_counts=claim_counts,
            earned_exposure=earned_exposure,
            total_paid=total_paid,
        )
        self._periods_raw = periods
        self._claim_counts = to_numpy(claim_counts, "claim_counts")
        self._earned_exposure = to_numpy(earned_exposure, "earned_exposure")
        self._total_paid = to_numpy(total_paid, "total_paid")
        self._external_index = external_index
        self._weights = weights
        self._periods_per_year = periods_per_year

        # Pre-compute loss cost for reference
        self._loss_cost = self._total_paid / self._earned_exposure

    @property
    def loss_cost(self) -> np.ndarray:
        """Observed loss cost series (total paid / earned exposure)."""
        return self._loss_cost.copy()

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
    ) -> LossCostTrendResult:
        """Fit both the frequency and severity components and combine.

        Parameters
        ----------
        method:
            ``'log_linear'`` or ``'local_linear_trend'``. Applied to both
            frequency and severity.
        changepoints:
            Structural break indices applied to both frequency and severity fits.
        detect_breaks:
            Auto-detect structural breaks in both series if ``changepoints`` is None.
        seasonal:
            Include quarterly seasonal dummies (log-linear method only).
        n_bootstrap:
            Bootstrap replicates for CI estimation.
        projection_periods:
            Number of periods to project forward.
        ci_level:
            Confidence level for bootstrap CI.
        penalty:
            Ruptures PELT penalty.

        Returns
        -------
        LossCostTrendResult
        """
        freq_fitter = FrequencyTrendFitter(
            periods=self._periods_raw,
            claim_counts=self._claim_counts,
            earned_exposure=self._earned_exposure,
            weights=self._weights,
            periods_per_year=self._periods_per_year,
        )
        freq_result = freq_fitter.fit(
            method=method,
            changepoints=changepoints,
            detect_breaks=detect_breaks,
            seasonal=seasonal,
            n_bootstrap=n_bootstrap,
            projection_periods=projection_periods,
            ci_level=ci_level,
            penalty=penalty,
        )

        sev_fitter = SeverityTrendFitter(
            periods=self._periods_raw,
            total_paid=self._total_paid,
            claim_counts=self._claim_counts,
            external_index=self._external_index,
            weights=self._weights,
            periods_per_year=self._periods_per_year,
        )
        sev_result = sev_fitter.fit(
            method=method,
            changepoints=changepoints,
            detect_breaks=detect_breaks,
            seasonal=seasonal,
            n_bootstrap=n_bootstrap,
            projection_periods=projection_periods,
            ci_level=ci_level,
            penalty=penalty,
        )

        si = sev_fitter.superimposed_inflation()

        combined_trend = (1.0 + freq_result.trend_rate) * (1.0 + sev_result.trend_rate) - 1.0

        projection = self._combined_projection(
            freq_result, sev_result, projection_periods
        )

        return LossCostTrendResult(
            frequency=freq_result,
            severity=sev_result,
            combined_trend_rate=float(combined_trend),
            superimposed_inflation=si,
            projection=projection,
        )

    def projected_loss_cost(
        self,
        future_periods: int = 8,
        ci: float = 0.95,
        method: str = "log_linear",
    ) -> pl.DataFrame:
        """Fit and return the projected loss cost DataFrame directly.

        Convenience method that calls :meth:`fit` and returns the projection
        DataFrame from the combined result.

        Parameters
        ----------
        future_periods:
            Number of periods to project forward.
        ci:
            Confidence level for the projection fan.
        method:
            Fitting method (``'log_linear'`` or ``'local_linear_trend'``).

        Returns
        -------
        Polars DataFrame with columns ``period``, ``point``, ``lower``, ``upper``.
        """
        result = self.fit(
            method=method,
            projection_periods=future_periods,
            ci_level=ci,
        )
        return result.projection

    def _combined_projection(
        self,
        freq_result,
        sev_result,
        projection_periods: int,
    ) -> pl.DataFrame:
        """Build a combined loss cost projection by multiplying freq × sev fans."""
        freq_proj = freq_result.projection
        sev_proj = sev_result.projection

        if len(freq_proj) == 0 or len(sev_proj) == 0:
            return pl.DataFrame({"period": [], "point": [], "lower": [], "upper": []})

        min_len = min(len(freq_proj), len(sev_proj))
        fp = freq_proj.head(min_len)
        sp = sev_proj.head(min_len)

        # Last observed loss cost as anchor
        last_lc = self._loss_cost[-1]

        # Scale projections relative to the last fitted frequency and severity
        last_freq_fitted = freq_result.fitted_values[-1]
        last_sev_fitted = sev_result.fitted_values[-1]

        freq_point = fp["point"].to_numpy() / float(last_freq_fitted)
        freq_lower = fp["lower"].to_numpy() / float(last_freq_fitted)
        freq_upper = fp["upper"].to_numpy() / float(last_freq_fitted)

        sev_point = sp["point"].to_numpy() / float(last_sev_fitted)
        sev_lower = sp["lower"].to_numpy() / float(last_sev_fitted)
        sev_upper = sp["upper"].to_numpy() / float(last_sev_fitted)

        return pl.DataFrame(
            {
                "period": list(range(1, min_len + 1)),
                "point": (last_lc * freq_point * sev_point).tolist(),
                "lower": (last_lc * freq_lower * sev_lower).tolist(),
                "upper": (last_lc * freq_upper * sev_upper).tolist(),
            }
        )

    def summary(self) -> str:
        """Brief string summary of the fitter configuration."""
        n = len(self._loss_cost)
        lc_range = f"{self._loss_cost.min():.2f} – {self._loss_cost.max():.2f}"
        has_idx = self._external_index is not None
        return (
            f"LossCostTrendFitter: {n} periods, "
            f"loss cost range {lc_range}, "
            f"external_index={'yes' if has_idx else 'no'}, "
            f"periods_per_year={self._periods_per_year}"
        )
