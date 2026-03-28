"""MultiIndexDecomposer — decompose severity trend across multiple external indices.

Whilst a single external index (e.g. HPTH motor repair SPPI) can isolate
superimposed inflation, real pricing work often requires finer attribution.
Is the severity trend driven by parts costs, labour rates, or third-party
claims inflation? This module answers that question by regressing log(severity)
simultaneously on the logs of all candidate indices, then converting OLS
coefficients into annualised percentage contributions.

The model is:

    log(sev_t) = alpha + beta_1 * log(idx1_t) + beta_2 * log(idx2_t) + ... + epsilon_t

Each beta_k is the elasticity of severity with respect to index k. The annual
contribution of index k is then:

    annual_contribution_k = exp(beta_k * log_growth_k_pa) - 1

where log_growth_k_pa is the annualised log-growth rate of index k over the
sample. The residual (superimposed inflation) is the portion of total severity
trend unexplained by any of the indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from ._utils import (
    PandasOrPolars,
    annual_trend_rate,
    safe_log,
    to_numpy,
    validate_lengths,
)


# ---------------------------------------------------------------------------
# UK insurance market event calendar for break attribution
# ---------------------------------------------------------------------------

UK_INSURANCE_EVENTS: dict[str, str] = {
    "2012Q1": "Referral fee ban (LASPO Act) — legal costs step-change",
    "2013Q2": "Whiplash reforms announced — soft tissue claims reduction",
    "2016Q2": "Ogden rate change to -0.75% — catastrophic injury uplift",
    "2020Q1": "COVID-19 lockdowns — sharp frequency suppression",
    "2020Q3": "Post-lockdown bounce in severity (backlog claims)",
    "2021Q2": "Civil Liability Act whiplash tariff effective — PI step-down",
    "2022Q1": "Energy/supply chain inflation peak — repair parts surge",
    "2022Q4": "Ogden rate change to -0.25% — partial reversal",
    "2023Q1": "FCA Consumer Duty pricing rules effective",
    "2023Q3": "Used car values normalising after chip shortage peak",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiIndexResult:
    """Results from a multi-index severity decomposition.

    Attributes
    ----------
    coefficients:
        OLS elasticity of log(severity) with respect to each log(index).
        A coefficient of 0.6 means a 1% rise in that index is associated
        with a 0.6% rise in severity, holding all other indices constant.
    annual_contributions:
        Annualised percentage contribution of each index to total severity
        trend. Computed as exp(coeff * log_growth_pa) - 1 for each index.
    r_squared:
        R-squared of the log-space OLS regression.
    residual_rate:
        Annualised superimposed inflation — severity trend not explained
        by any of the supplied indices. Positive means severity is growing
        faster than the indices suggest; negative means the reverse.
    decomposition_table:
        Polars DataFrame with columns: component, coefficient,
        annual_contribution_pct, share_of_total_pct. Includes one row per
        index plus a Residual row.
    total_severity_trend:
        Annualised total severity trend, estimated from the fitted trend of
        log(severity) over the sample.
    periods_per_year:
        As supplied to the constructor.
    """

    coefficients: dict[str, float]
    annual_contributions: dict[str, float]
    r_squared: float
    residual_rate: float
    decomposition_table: pl.DataFrame
    total_severity_trend: float
    periods_per_year: int

    def summary(self) -> str:
        """Return a formatted text summary of the decomposition.

        Returns
        -------
        str
            Multi-line string suitable for printing to a console or logging.
        """
        lines = [
            "=== Multi-Index Severity Decomposition ===",
            f"Total severity trend (pa) : {self.total_severity_trend:.2%}",
            f"Residual (superimposed)   : {self.residual_rate:.2%}",
            f"R-squared                 : {self.r_squared:.4f}",
            "",
            "Index contributions:",
        ]
        for name, contrib in self.annual_contributions.items():
            coef = self.coefficients[name]
            lines.append(f"  {name:<30s}  coeff={coef:+.4f}  pa={contrib:+.2%}")
        lines.append(f"  {'Residual':<30s}  coeff=  N/A   pa={self.residual_rate:+.2%}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_idx = len(self.coefficients)
        return (
            f"MultiIndexResult("
            f"n_indices={n_idx}, "
            f"total_trend={self.total_severity_trend:.4f}, "
            f"residual={self.residual_rate:.4f}, "
            f"r_squared={self.r_squared:.4f})"
        )


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class MultiIndexDecomposer:
    """Decompose severity trend across multiple external indices.

    Uses OLS regression of log(severity) on log(index_1), log(index_2), ...
    to allocate total trend into index-explained components plus a residual.

    The residual is the superimposed inflation: severity growth not captured
    by any of the supplied economic indices. In motor insurance this typically
    reflects structural factors such as ADAS recalibration costs, claims
    management practices, or litigation trends.

    Parameters
    ----------
    periods:
        Ordered sequence of period labels. Used for labelling only.
    severity:
        Average claim cost per period (total paid / claim count). Must be
        strictly positive in every period.
    indices:
        Mapping of index name to index values. Each array must be the same
        length as ``severity``. Typical choices: ONS HPTH (motor repair
        SPPI), ONS D7DO (building maintenance), CPI sub-indices.
    weights:
        Optional array of observation weights for WLS. Useful for
        down-weighting older or thinner periods. If None, OLS is used.
    periods_per_year:
        Number of periods per year. Use 4 for quarterly data (default)
        or 12 for monthly data.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_trend import MultiIndexDecomposer
    >>>
    >>> n = 20
    >>> t = np.arange(n, dtype=float)
    >>> # Severity driven 60% by parts costs (6% pa) and 40% by labour (4% pa)
    >>> parts = np.exp(0.06 / 4 * t)
    >>> labour = np.exp(0.04 / 4 * t)
    >>> severity = 5000.0 * parts ** 0.6 * labour ** 0.4
    >>> periods = [f"{y}Q{q}" for y in range(2019, 2024) for q in range(1, 5)][:n]
    >>>
    >>> decomposer = MultiIndexDecomposer(
    ...     periods=periods,
    ...     severity=severity,
    ...     indices={"parts": parts, "labour": labour},
    ... )
    >>> result = decomposer.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        periods: PandasOrPolars,
        severity: PandasOrPolars,
        indices: dict[str, PandasOrPolars],
        weights: Optional[PandasOrPolars] = None,
        periods_per_year: int = 4,
    ) -> None:
        if periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be a positive integer, got {periods_per_year!r}."
            )
        if not indices:
            raise ValueError(
                "indices must contain at least one external index. "
                "For single-series superimposed inflation use SeverityTrendFitter."
            )

        self._periods_raw = periods
        self._severity = to_numpy(severity, "severity")
        self._periods_per_year = periods_per_year

        if np.any(self._severity <= 0):
            raise ValueError(
                "severity must be strictly positive in all periods. "
                f"Found {np.sum(self._severity <= 0)} non-positive value(s)."
            )

        # Validate and store indices
        self._index_names: list[str] = list(indices.keys())
        self._index_arrays: dict[str, np.ndarray] = {}
        for name, arr in indices.items():
            arr_np = to_numpy(arr, f"index '{name}'")
            if len(arr_np) != len(self._severity):
                raise ValueError(
                    f"Index '{name}' has length {len(arr_np)}, but severity has "
                    f"length {len(self._severity)}. All series must be the same length."
                )
            if np.any(arr_np <= 0):
                raise ValueError(
                    f"Index '{name}' contains non-positive values. "
                    "All index values must be strictly positive for log transformation."
                )
            self._index_arrays[name] = arr_np

        self._weights: Optional[np.ndarray] = (
            to_numpy(weights, "weights") if weights is not None else None
        )
        if self._weights is not None and len(self._weights) != len(self._severity):
            raise ValueError(
                f"weights length ({len(self._weights)}) must match severity length "
                f"({len(self._severity)})."
            )

    def fit(self) -> MultiIndexResult:
        """Run the multi-index decomposition.

        Fits OLS (or WLS if weights were supplied) of log(severity) on the
        logs of all supplied indices, then converts elasticity coefficients
        into annualised contribution rates.

        Returns
        -------
        MultiIndexResult
            Full decomposition result including coefficients, contributions,
            R-squared, residual rate, and decomposition table.
        """
        import statsmodels.api as sm

        n = len(self._severity)
        log_sev = safe_log(self._severity, "severity")

        # Build design matrix: intercept + one column per log(index)
        log_index_cols = []
        for name in self._index_names:
            log_index_cols.append(safe_log(self._index_arrays[name], f"index '{name}'"))

        X_raw = np.column_stack(log_index_cols)  # shape (n, k)
        X = sm.add_constant(X_raw, prepend=True)  # (n, k+1); first col is intercept

        if self._weights is not None:
            ols_result = sm.WLS(log_sev, X, weights=self._weights).fit()
        else:
            ols_result = sm.OLS(log_sev, X).fit()

        # params[0] is the intercept; params[1:] are index elasticities
        intercept = float(ols_result.params[0])
        elasticities: dict[str, float] = {}
        for i, name in enumerate(self._index_names):
            elasticities[name] = float(ols_result.params[i + 1])

        r_squared = float(ols_result.rsquared)

        # ------------------------------------------------------------------
        # Annualised growth rate of each index over the sample
        # ------------------------------------------------------------------
        # Fit a simple log-linear slope to each index to get its annualised
        # growth rate. This is the cleanest approach because it is consistent
        # with how the severity trend is measured — both are log-linear slopes
        # converted via exp(beta * ppy) - 1.
        t = np.arange(n, dtype=float)
        X_trend = sm.add_constant(t)

        index_log_growth_pa: dict[str, float] = {}
        for name in self._index_names:
            log_idx = safe_log(self._index_arrays[name], f"index '{name}'")
            idx_res = sm.OLS(log_idx, X_trend).fit()
            beta_idx = float(idx_res.params[1])
            # annualised log-growth (not compound): beta * periods_per_year
            index_log_growth_pa[name] = beta_idx * self._periods_per_year

        # ------------------------------------------------------------------
        # Annual contribution of each index
        # ------------------------------------------------------------------
        # annual_contribution_k = exp(coeff_k * annualised_log_growth_k) - 1
        # This is the compound annual rate attributable to index k, given its
        # observed growth rate and the estimated elasticity.
        annual_contributions: dict[str, float] = {}
        for name in self._index_names:
            contrib_log = elasticities[name] * index_log_growth_pa[name]
            annual_contributions[name] = float(np.exp(contrib_log) - 1.0)

        # ------------------------------------------------------------------
        # Total severity trend (independently estimated from log(sev) slope)
        # ------------------------------------------------------------------
        sev_res = sm.OLS(log_sev, X_trend).fit()
        total_sev_beta = float(sev_res.params[1])
        total_severity_trend = annual_trend_rate(total_sev_beta, self._periods_per_year)

        # ------------------------------------------------------------------
        # Residual (superimposed inflation)
        # ------------------------------------------------------------------
        # Sum of index annual contributions (compound combination)
        total_index_explained = float(
            np.prod([1.0 + c for c in annual_contributions.values()]) - 1.0
        )
        # Additive residual: total_trend - index_explained
        # We use the additive approximation here for interpretability; for
        # small rates the error is negligible, and actuaries expect additive
        # decomposition tables.
        residual_rate = total_severity_trend - total_index_explained

        # ------------------------------------------------------------------
        # Decomposition table
        # ------------------------------------------------------------------
        components = list(self._index_names) + ["Residual"]
        coefficients_col = [elasticities[n] for n in self._index_names] + [float("nan")]
        contrib_pct_col = [
            annual_contributions[n] * 100.0 for n in self._index_names
        ] + [residual_rate * 100.0]

        # Share of total trend (signed: negative share possible if residual < 0)
        if abs(total_severity_trend) > 1e-10:
            share_col = [c / total_severity_trend * 100.0 for c in contrib_pct_col]
        else:
            share_col = [float("nan")] * len(components)

        table = pl.DataFrame(
            {
                "component": components,
                "coefficient": coefficients_col,
                "annual_contribution_pct": contrib_pct_col,
                "share_of_total_pct": share_col,
            }
        )

        return MultiIndexResult(
            coefficients=elasticities,
            annual_contributions=annual_contributions,
            r_squared=r_squared,
            residual_rate=residual_rate,
            decomposition_table=table,
            total_severity_trend=total_severity_trend,
            periods_per_year=self._periods_per_year,
        )
