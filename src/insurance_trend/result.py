"""TrendResult and LossCostTrendResult dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class TrendResult:
    """The output of a single-component trend fit (frequency or severity).

    Attributes
    ----------
    trend_rate:
        Annual trend rate as a decimal. E.g. ``0.085`` means 8.5 % per annum.
    ci_lower:
        Lower bound of the 95 % (by default) bootstrap confidence interval.
    ci_upper:
        Upper bound of the 95 % bootstrap confidence interval.
    method:
        Name of the fitting method used: ``'log_linear'``, ``'piecewise'``, or
        ``'local_linear_trend'``.
    fitted_values:
        Polars Series of fitted (back-transformed) values on the original scale,
        aligned to the input observations.
    residuals:
        Polars Series of residuals: ``actual / fitted - 1`` (multiplicative
        residuals on the original scale).
    changepoints:
        List of detected or user-supplied structural break indices.
    projection:
        Polars DataFrame with columns ``period``, ``point``, ``lower``, ``upper``
        containing the forward projection and confidence fan.
    r_squared:
        R-squared from the log-space OLS fit. For piecewise models this applies
        to the final segment.
    actuals:
        Polars Series of the original observed values that were fitted.
    periods:
        Polars Series of the original period labels.
    n_bootstrap:
        Number of bootstrap replicates used for CI calculation.
    periods_per_year:
        Number of periods per year (4 for quarterly, 12 for monthly).
    """

    trend_rate: float
    ci_lower: float
    ci_upper: float
    method: str
    fitted_values: pl.Series
    residuals: pl.Series
    changepoints: list
    projection: pl.DataFrame
    r_squared: float
    actuals: pl.Series = field(default_factory=lambda: pl.Series("actuals", []))
    periods: pl.Series = field(default_factory=lambda: pl.Series("periods", []))
    n_bootstrap: int = 1000
    periods_per_year: int = 4

    def trend_factor(self, n_periods: float) -> float:
        """Return the compound trend factor over ``n_periods`` periods.

        Parameters
        ----------
        n_periods:
            Number of periods (using the same unit as the input data, typically
            quarters). For an 18-month projection from quarterly data pass ``6``.

        Returns
        -------
        The compounded trend factor, e.g. ``1.127`` for 12.7 % cumulative trend.
        """
        return float((1.0 + self.trend_rate) ** (n_periods / self.periods_per_year))

    def plot(self) -> Any:
        """Return a three-panel matplotlib Figure.

        Panel 1 — Actual vs Fitted values on the original scale.
        Panel 2 — Multiplicative residuals with a zero reference line.
        Panel 3 — Forward projection fan (point estimate + 90 % and 95 % CI bands).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plot import trend_diagnostic_plot  # avoid circular import at module level

        return trend_diagnostic_plot(self)

    def summary(self) -> str:
        """Return a human-readable text summary of the trend result."""
        lines = [
            f"Method          : {self.method}",
            f"Trend rate (pa) : {self.trend_rate:.2%}",
            f"95% CI          : ({self.ci_lower:.2%}, {self.ci_upper:.2%})",
            f"R-squared       : {self.r_squared:.4f}",
            f"Changepoints    : {self.changepoints}",
            f"Bootstrap N     : {self.n_bootstrap}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TrendResult(trend_rate={self.trend_rate:.4f}, "
            f"ci=({self.ci_lower:.4f}, {self.ci_upper:.4f}), "
            f"method='{self.method}', r_squared={self.r_squared:.4f})"
        )


@dataclass
class LossCostTrendResult:
    """Combined frequency x severity trend result.

    Attributes
    ----------
    frequency:
        ``TrendResult`` for the frequency component.
    severity:
        ``TrendResult`` for the severity component.
    combined_trend_rate:
        Annual loss cost trend = (1 + freq) * (1 + sev) - 1.
    superimposed_inflation:
        Severity trend net of any external economic index, i.e. the part of
        severity trend not explained by general economic inflation.
    projection:
        Polars DataFrame: ``period``, ``point``, ``lower``, ``upper`` for the
        combined loss cost projected forward.
    """

    frequency: TrendResult
    severity: TrendResult
    combined_trend_rate: float
    superimposed_inflation: float | None
    projection: pl.DataFrame

    def decompose(self) -> dict:
        """Return a dict decomposing the loss cost trend into its components.

        Returns
        -------
        Dict with keys ``freq_trend``, ``sev_trend``, ``combined_trend``, and
        ``superimposed`` (the last is ``None`` if no external index was provided).
        """
        return {
            "freq_trend": self.frequency.trend_rate,
            "sev_trend": self.severity.trend_rate,
            "combined_trend": self.combined_trend_rate,
            "superimposed": self.superimposed_inflation,
        }

    def trend_factor(self, n_periods: float) -> float:
        """Combined loss cost compound trend factor over ``n_periods`` periods."""
        return float((1.0 + self.combined_trend_rate) ** (n_periods / self.frequency.periods_per_year))

    def plot(self) -> Any:
        """Return a multi-panel diagnostic figure covering frequency, severity,
        and combined loss cost projection."""
        from .plot import loss_cost_diagnostic_plot

        return loss_cost_diagnostic_plot(self)

    def summary(self) -> str:
        """Human-readable summary of the combined loss cost trend."""
        si = (
            f"{self.superimposed_inflation:.2%}"
            if self.superimposed_inflation is not None
            else "N/A (no external index)"
        )
        lines = [
            "=== Loss Cost Trend Summary ===",
            f"Frequency trend (pa)         : {self.frequency.trend_rate:.2%}",
            f"Severity trend (pa)          : {self.severity.trend_rate:.2%}",
            f"Combined loss cost trend (pa): {self.combined_trend_rate:.2%}",
            f"Superimposed inflation (pa)  : {si}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LossCostTrendResult("
            f"freq={self.frequency.trend_rate:.4f}, "
            f"sev={self.severity.trend_rate:.4f}, "
            f"combined={self.combined_trend_rate:.4f})"
        )
