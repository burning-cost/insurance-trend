"""InflationDecomposer — Harvey structural time series decomposition of claims inflation.

Insurance claims inflation is not a single number. It is a mixture of:

- A *structural trend*: persistent drift in average cost caused by vehicle
  complexity, labour shortages, social/legal inflation, and technological change.
  Pricing actuaries need to embed this permanently in future rate projections.

- A *cyclical component*: mean-reverting oscillation caused by credit hire market
  swings, used car price cycles, and post-COVID repair backlog effects. Embedding
  this as a permanent trend is a common and costly mistake.

- *Seasonal effects*: calendar patterns (Q4 weather, summer holiday accident
  spikes).

- *Irregular*: genuine noise — thin data, random large losses, estimation error.

The standard log-linear OLS trend conflates all four. This module separates them
using the Harvey (1989) structural time series model, estimated via maximum
likelihood using ``statsmodels.tsa.statespace.UnobservedComponents``.

The Harvey stochastic cycle model:

    level:   mu_t  = mu_{t-1} + beta_{t-1} + eta_t      (eta ~ N(0, sigma^2_level))
    slope:   beta_t = beta_{t-1} + zeta_t               (zeta ~ N(0, sigma^2_slope))
    cycle:   psi_t  = rho * cos(lambda) * psi_{t-1} - rho * sin(lambda) * psi*_{t-1} + kappa_t
             psi*_t = rho * sin(lambda) * psi_{t-1} + rho * cos(lambda) * psi*_{t-1} + kappa*_t
    obs:     y_t = mu_t + psi_t + [seasonal_t] + eps_t

where lambda = 2*pi / cycle_period (radians), and rho in [0, 1] damps the cycle
towards zero over time. statsmodels estimates all variance parameters by MLE.

After fitting, components are extracted via the Kalman smoother using the
``res.level.smoothed``, ``res.trend.smoothed``, ``res.cycle.smoothed``, and
``res.seasonal.smoothed`` properties on the
``UnobservedComponentsResults`` object.

References
----------
Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman
Filter*. Cambridge University Press.

statsmodels documentation: statsmodels.tsa.statespace.structural
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from ._utils import PandasOrPolars, safe_log, to_numpy


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InflationDecompositionResult:
    """Output of an :class:`InflationDecomposer` fit.

    All time-series components are returned as Polars Series aligned to the
    input observations. In log space (when ``log_transform=True``), the
    components sum additively to the observed log-series.

    Attributes
    ----------
    trend:
        Smoothed level component (``mu_t`` in Harvey notation). This is the
        persistent structural level — the underlying cost trajectory stripped
        of cyclical and seasonal variation. Units are log-index values when
        ``log_transform=True``, original-scale otherwise.
    cycle:
        Stochastic cyclical component (``psi_t``). Mean-reverting deviation
        from the structural trend. Positive values indicate the series is
        currently running *above* the structural trend; negative values
        indicate it is *below*.
    seasonal:
        Seasonal component. All zeros when ``seasonal=None`` was passed at
        construction.
    irregular:
        Residual: ``observed - trend - cycle - seasonal``.
    structural_rate:
        Annualised structural trend rate as a decimal (e.g. ``0.085`` = 8.5 %
        per annum). Derived from the mean of the Kalman-smoothed slope state
        (``beta_t``) over the fitted sample. This is the trend to embed in
        forward projections when you believe the structural component persists.
    cyclical_position:
        Current cyclical deviation as a decimal at the end of the sample. For
        log-transformed data: ``exp(cycle[-1]) - 1``. A value of ``+0.05``
        means severity is 5 % above the structural trend at the latest period.
    cycle_period:
        Estimated cycle period in years, derived from the fitted frequency
        parameter ``lambda`` (radians per period). Returns ``float('nan')``
        when ``cycle=False``.
    total_trend_rate:
        Annualised total trend rate over the sample, computed via OLS on the
        observed series. Comparable to other fitters in this library.
    periods:
        Input period labels as a Polars Series.
    observations:
        Input observed values (log-transformed if ``log_transform=True``) as
        a Polars Series.
    log_transform:
        Whether the input was log-transformed before fitting.
    periods_per_year:
        As supplied to the constructor.
    aic:
        Akaike Information Criterion of the fitted state-space model.
    bic:
        Bayesian Information Criterion of the fitted state-space model.
    converged:
        Whether the MLE optimisation converged (``warnflag == 0``).
    n_obs:
        Number of observations.
    """

    trend: pl.Series
    cycle: pl.Series
    seasonal: pl.Series
    irregular: pl.Series
    structural_rate: float
    cyclical_position: float
    cycle_period: float
    total_trend_rate: float
    periods: pl.Series
    observations: pl.Series
    log_transform: bool
    periods_per_year: int
    aic: float
    bic: float
    converged: bool
    n_obs: int

    def summary(self) -> str:
        """Return a formatted text summary of the decomposition.

        Returns
        -------
        str
            Multi-line string suitable for printing to a console or logging.
        """
        cycle_dir = "above" if self.cyclical_position >= 0 else "below"
        cycle_period_str = (
            f"{self.cycle_period:.1f} years"
            if not (self.cycle_period != self.cycle_period)  # isnan check
            else "N/A (no cycle)"
        )
        lines = [
            "=== Claims Inflation Decomposition (Harvey Structural Model) ===",
            f"Observations          : {self.n_obs}",
            f"Periods per year      : {self.periods_per_year}",
            f"Log-transformed input : {self.log_transform}",
            "",
            "--- Trend ---",
            f"Structural trend (pa) : {self.structural_rate:.2%}",
            f"Total trend (pa)      : {self.total_trend_rate:.2%}",
            "",
            "--- Cycle ---",
            f"Estimated period      : {cycle_period_str}",
            f"Current position      : {self.cyclical_position:+.2%} ({cycle_dir} structural trend)",
            "",
            "--- Model fit ---",
            f"AIC                   : {self.aic:.1f}",
            f"BIC                   : {self.bic:.1f}",
            f"Converged             : {self.converged}",
        ]
        return "\n".join(lines)

    def decomposition_table(self) -> pl.DataFrame:
        """Return a Polars DataFrame with one column per component.

        Returns
        -------
        pl.DataFrame
            Columns: ``period``, ``observed``, ``trend``, ``cycle``,
            ``seasonal``, ``irregular``.
        """
        return pl.DataFrame(
            {
                "period": self.periods,
                "observed": self.observations,
                "trend": self.trend,
                "cycle": self.cycle,
                "seasonal": self.seasonal,
                "irregular": self.irregular,
            }
        )

    def plot(self) -> Any:
        """Return a three or four-panel matplotlib Figure showing each component.

        Panel 1 — Observed vs Trend (structural level).
        Panel 2 — Cyclical component.
        Panel 3 — Seasonal component (only shown when non-zero).
        Panel 4 — Irregular residual.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n = self.n_obs
        t = np.arange(n)
        period_labels = self.periods.cast(pl.String).to_list()

        obs_arr = self.observations.to_numpy()
        trend_arr = self.trend.to_numpy()
        cycle_arr = self.cycle.to_numpy()
        seas_arr = self.seasonal.to_numpy()
        irreg_arr = self.irregular.to_numpy()

        has_seasonal = np.any(np.abs(seas_arr) > 1e-10)
        n_panels = 4 if has_seasonal else 3

        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        fig.suptitle("Claims Inflation Decomposition — Harvey Structural Model", fontsize=11)

        tick_step = max(1, n // 8)
        tick_positions = list(range(0, n, tick_step))
        tick_labels = [period_labels[i] for i in tick_positions]

        # Panel 1: Observed vs Trend
        ax0 = axes[0]
        ax0.plot(t, obs_arr, color="steelblue", linewidth=1.5, label="Observed")
        ax0.plot(t, trend_arr, color="crimson", linewidth=2, linestyle="--", label="Structural trend")
        ax0.set_ylabel("Level")
        ax0.set_title("Observed vs Structural Trend")
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)

        # Panel 2: Cycle
        ax1 = axes[1]
        ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax1.fill_between(t, cycle_arr, 0, where=(cycle_arr >= 0), alpha=0.4, color="crimson", label="Above trend")
        ax1.fill_between(t, cycle_arr, 0, where=(cycle_arr < 0), alpha=0.4, color="steelblue", label="Below trend")
        ax1.plot(t, cycle_arr, color="darkred", linewidth=1.2)
        cp_str = f"{self.cycle_period:.1f} yrs" if not (self.cycle_period != self.cycle_period) else "N/A"
        ax1.set_ylabel("Cycle")
        ax1.set_title(f"Cyclical Component (period \u2248 {cp_str})")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        panel_idx = 2
        if has_seasonal:
            ax2 = axes[panel_idx]
            ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax2.bar(t, seas_arr, color="seagreen", alpha=0.6, label="Seasonal")
            ax2.set_ylabel("Seasonal")
            ax2.set_title("Seasonal Component")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            panel_idx += 1

        # Irregular
        ax_last = axes[panel_idx]
        ax_last.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax_last.bar(t, irreg_arr, color="grey", alpha=0.6, label="Irregular")
        ax_last.set_ylabel("Irregular")
        ax_last.set_title("Irregular (Residual Noise)")
        ax_last.legend(fontsize=8)
        ax_last.grid(True, alpha=0.3)
        ax_last.set_xticks(tick_positions)
        ax_last.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        cp_str = f"{self.cycle_period:.1f}" if not (self.cycle_period != self.cycle_period) else "nan"
        return (
            f"InflationDecompositionResult("
            f"n_obs={self.n_obs}, "
            f"structural_rate={self.structural_rate:.4f}, "
            f"cyclical_position={self.cyclical_position:+.4f}, "
            f"cycle_period={cp_str}yr, "
            f"aic={self.aic:.1f})"
        )


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


class InflationDecomposer:
    """Decompose insurance claims inflation into structural and cyclical components.

    Uses the Harvey (1989) structural time series model — a state-space
    formulation fitted by maximum likelihood via the Kalman filter — to separate
    a claims severity or loss ratio index into:

    - **Structural trend**: persistent level and slope driven by long-run cost
      escalators (vehicle complexity, labour shortages, legal inflation). This is
      the component pricing actuaries should embed in future projections.

    - **Stochastic cycle**: mean-reverting oscillation capturing the insurance
      market cycle, used car price cycles, credit hire volatility, and
      post-catastrophe bounce effects. Embedding this as a permanent trend leads
      to over-pricing when the cycle turns.

    - **Seasonal**: optional calendar seasonal effects (Q4 weather, seasonal
      repair demand).

    - **Irregular**: high-frequency observation noise.

    The model is estimated using ``statsmodels.tsa.statespace.UnobservedComponents``
    with ``level=True``, ``trend=True``, ``stochastic_level=True``,
    ``stochastic_trend=True``, ``cycle=True``, ``stochastic_cycle=True``,
    ``damped_cycle=True``. Components are extracted via the Kalman smoother
    using the ``res.level.smoothed``, ``res.trend.smoothed``,
    ``res.cycle.smoothed``, and ``res.seasonal.smoothed`` properties.

    Parameters
    ----------
    series:
        The observed inflation series. Accepts a pandas Series with a
        DatetimeIndex or PeriodIndex, a Polars Series, a list, or a numpy
        array. Typical inputs: a quarterly severity index (rebased to 100) or
        a loss ratio series. Must be strictly positive if
        ``log_transform=True``.
    periods:
        Optional period labels. If ``series`` is a pandas Series with a named
        index, those labels are used automatically. Otherwise supply an
        array-like of the same length as ``series``. If None, integer indices
        are used.
    cycle:
        Whether to include the stochastic cycle component. Set to ``False``
        for short series (<16 periods) where the cycle is unidentifiable.
        Default: ``True``.
    stochastic_cycle:
        Whether the cycle variance is stochastic (i.e., the cycle amplitude
        is allowed to vary over time). Default: ``True``.
    seasonal:
        Number of periods in a seasonal cycle. Use ``4`` for quarterly data
        with annual seasonality, ``12`` for monthly data. ``None`` suppresses
        the seasonal component. Default: ``None``.
    damped_cycle:
        Whether to include a damping factor (rho) in the cycle, enforcing
        mean reversion. Almost always ``True`` for insurance data; a
        non-damped cycle (rho=1) is non-stationary. Default: ``True``.
    cycle_period_bounds:
        Bounds on the cycle period in *years*, supplied as ``(lower, upper)``.
        Internally converted to periods by multiplying by ``periods_per_year``.
        The default ``(2, 12)`` allows cycles of 2–12 years, covering the
        typical insurance underwriting cycle (3–7 years). For quarterly data
        this becomes (8, 48) periods.
    log_transform:
        If ``True`` (the default), the series is log-transformed before
        fitting. Appropriate for price indices and loss ratios (strictly
        positive, log-normally distributed). Set to ``False`` only if the
        series is already on a log scale or you have a specific reason to
        model it on the original scale.
    periods_per_year:
        Number of periods per year. Use ``4`` for quarterly data (default) and
        ``12`` for monthly data. Affects annualisation of trend rates and the
        conversion of ``cycle_period_bounds`` to periods.
    fit_kwargs:
        Additional keyword arguments passed to
        ``UnobservedComponentsResults.fit()``. For example, use
        ``fit_kwargs={'method': 'powell'}`` if the default L-BFGS-B fails to
        converge.

    Examples
    --------
    Quarterly motor severity index, nine years of data:

    >>> import numpy as np
    >>> from insurance_trend import InflationDecomposer
    >>>
    >>> rng = np.random.default_rng(42)
    >>> n = 36  # nine years of quarterly data
    >>> t = np.arange(n, dtype=float)
    >>> structural = np.exp(0.07 / 4 * t)
    >>> cycle = 0.06 * np.sin(2 * np.pi * t / 24)  # 6-year cycle
    >>> noise = rng.normal(0, 0.01, n)
    >>> severity_index = 100.0 * structural * np.exp(cycle + noise)
    >>>
    >>> periods = [f"{y}Q{q}" for y in range(2016, 2026) for q in range(1, 5)][:n]
    >>> decomposer = InflationDecomposer(
    ...     series=severity_index,
    ...     periods=periods,
    ...     cycle=True,
    ...     cycle_period_bounds=(3, 10),
    ... )
    >>> result = decomposer.fit()
    >>> print(result.summary())

    Notes
    -----
    Minimum sample size: the Harvey model with a local linear trend plus
    stochastic cycle has 4–5 variance parameters to estimate. A minimum of 24
    observations (6 years of quarterly data) is recommended; 40+ is preferred.
    With fewer than 16 observations, use ``cycle=False``.
    """

    _MIN_OBS_WITH_CYCLE = 16
    _MIN_OBS_WITHOUT_CYCLE = 8

    def __init__(
        self,
        series: PandasOrPolars,
        periods: Optional[PandasOrPolars] = None,
        *,
        cycle: bool = True,
        stochastic_cycle: bool = True,
        seasonal: Optional[int] = None,
        damped_cycle: bool = True,
        cycle_period_bounds: tuple[float, float] = (2.0, 12.0),
        log_transform: bool = True,
        periods_per_year: int = 4,
        fit_kwargs: Optional[dict] = None,
    ) -> None:
        # ------------------------------------------------------------------ #
        # Validate periods_per_year
        # ------------------------------------------------------------------ #
        if periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be a positive integer, got {periods_per_year!r}."
            )
        self._periods_per_year = periods_per_year

        # ------------------------------------------------------------------ #
        # Extract period labels
        # ------------------------------------------------------------------ #
        if isinstance(series, pd.Series) and periods is None:
            if len(series.index) > 0:
                self._periods_raw = [str(p) for p in series.index]
            else:
                self._periods_raw = None
        elif periods is not None:
            if isinstance(periods, (pd.Series, pd.Index)):
                self._periods_raw = [str(p) for p in periods]
            elif isinstance(periods, pl.Series):
                self._periods_raw = periods.cast(pl.String).to_list()
            else:
                self._periods_raw = [str(p) for p in periods]
        else:
            self._periods_raw = None

        # ------------------------------------------------------------------ #
        # Convert series to numpy
        # ------------------------------------------------------------------ #
        self._series_raw = to_numpy(series, "series")
        n = len(self._series_raw)

        if self._periods_raw is None:
            self._periods_raw = [str(i) for i in range(n)]

        if len(self._periods_raw) != n:
            raise ValueError(
                f"periods length ({len(self._periods_raw)}) must match series length ({n})."
            )

        # ------------------------------------------------------------------ #
        # Validate strictly positive for log transform
        # ------------------------------------------------------------------ #
        if log_transform and np.any(self._series_raw <= 0):
            n_bad = int(np.sum(self._series_raw <= 0))
            raise ValueError(
                f"series contains {n_bad} non-positive value(s). "
                "log_transform=True requires all values to be strictly positive. "
                "Either fix the data or set log_transform=False."
            )

        # ------------------------------------------------------------------ #
        # Minimum sample size
        # ------------------------------------------------------------------ #
        min_obs = self._MIN_OBS_WITH_CYCLE if cycle else self._MIN_OBS_WITHOUT_CYCLE
        if n < min_obs:
            if cycle:
                raise ValueError(
                    f"series has only {n} observations; at least {min_obs} are required "
                    f"when cycle=True. For short series use cycle=False."
                )
            else:
                raise ValueError(
                    f"series has only {n} observations; at least {min_obs} are required."
                )

        # ------------------------------------------------------------------ #
        # Validate cycle_period_bounds
        # ------------------------------------------------------------------ #
        if len(cycle_period_bounds) != 2:
            raise ValueError("cycle_period_bounds must be a 2-tuple of (lower, upper) years.")
        lo, hi = cycle_period_bounds
        if lo <= 0 or hi <= 0:
            raise ValueError(
                f"cycle_period_bounds must be strictly positive, got ({lo}, {hi})."
            )
        if lo >= hi:
            raise ValueError(
                f"cycle_period_bounds lower ({lo}) must be less than upper ({hi})."
            )
        # Convert from years to periods
        self._cycle_period_bounds_periods = (
            lo * periods_per_year,
            hi * periods_per_year,
        )

        # ------------------------------------------------------------------ #
        # Validate seasonal
        # ------------------------------------------------------------------ #
        if seasonal is not None and seasonal < 2:
            raise ValueError(
                f"seasonal must be None or an integer >= 2, got {seasonal!r}."
            )

        # ------------------------------------------------------------------ #
        # Store parameters
        # ------------------------------------------------------------------ #
        self._cycle = cycle
        self._stochastic_cycle = stochastic_cycle
        self._seasonal = seasonal
        self._damped_cycle = damped_cycle
        self._log_transform = log_transform
        self._fit_kwargs: dict = fit_kwargs or {}

    # ---------------------------------------------------------------------- #
    # Public fit method
    # ---------------------------------------------------------------------- #

    def fit(self) -> InflationDecompositionResult:
        """Fit the Harvey structural model and return the decomposed components.

        Runs the Kalman filter and smoother via statsmodels
        ``UnobservedComponents.fit()``. All variance parameters are estimated
        by maximum likelihood (L-BFGS-B by default). Components are extracted
        via ``res.level.smoothed``, ``res.trend.smoothed``,
        ``res.cycle.smoothed``, and ``res.seasonal.smoothed``.

        Returns
        -------
        InflationDecompositionResult
            Dataclass containing all components, summary statistics, and helper
            methods (``summary()``, ``decomposition_table()``, ``plot()``).

        Raises
        ------
        ValueError
            If the input series fails validation.
        RuntimeError
            If statsmodels raises an unexpected error during fitting.
        """
        import warnings

        try:
            import statsmodels.api as sm
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError as exc:
            raise ImportError(
                "statsmodels >= 0.14 is required for InflationDecomposer. "
                "Install it with: pip install statsmodels"
            ) from exc

        n = len(self._series_raw)

        # ------------------------------------------------------------------ #
        # Apply log transform
        # ------------------------------------------------------------------ #
        if self._log_transform:
            y = safe_log(self._series_raw, "series")
        else:
            y = self._series_raw.astype(float)

        # ------------------------------------------------------------------ #
        # Build the UC model spec
        # ------------------------------------------------------------------ #
        model_kwargs: dict[str, Any] = {
            "level": True,
            "trend": True,
            "stochastic_level": True,
            "stochastic_trend": True,
            "irregular": True,
        }

        if self._cycle:
            model_kwargs["cycle"] = True
            model_kwargs["stochastic_cycle"] = self._stochastic_cycle
            model_kwargs["damped_cycle"] = self._damped_cycle
            model_kwargs["cycle_period_bounds"] = self._cycle_period_bounds_periods

        if self._seasonal is not None:
            model_kwargs["seasonal"] = self._seasonal
            model_kwargs["stochastic_seasonal"] = True

        try:
            model = UnobservedComponents(y, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to construct UnobservedComponents model: {exc}"
            ) from exc

        # ------------------------------------------------------------------ #
        # Fit by MLE
        # ------------------------------------------------------------------ #
        fit_kwargs = {"disp": False}
        fit_kwargs.update(self._fit_kwargs)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(**fit_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"InflationDecomposer MLE failed: {exc}. "
                "Try increasing the series length, setting cycle=False, "
                "or passing fit_kwargs={'method': 'powell'}."
            ) from exc

        # ------------------------------------------------------------------ #
        # Extract smoothed components via res.level.smoothed etc.
        # These are the canonical Kalman smoother component extractions in
        # statsmodels UnobservedComponents.
        # ------------------------------------------------------------------ #
        trend_arr = self._get_smoothed(res, "level", n)
        cycle_arr = self._get_smoothed(res, "cycle", n) if self._cycle else np.zeros(n)
        seasonal_arr = self._get_smoothed(res, "seasonal", n) if self._seasonal is not None else np.zeros(n)
        # Irregular: residual between observed and modelled components
        irregular_arr = y - trend_arr - cycle_arr - seasonal_arr

        # ------------------------------------------------------------------ #
        # Structural trend rate from smoothed slope state (beta)
        # ------------------------------------------------------------------ #
        structural_rate = self._compute_structural_rate(res, y, n)

        # ------------------------------------------------------------------ #
        # Cyclical position at end of sample
        # ------------------------------------------------------------------ #
        cyclical_position = self._compute_cyclical_position(cycle_arr)

        # ------------------------------------------------------------------ #
        # Estimated cycle period in years
        # ------------------------------------------------------------------ #
        cycle_period_years = self._compute_cycle_period(res)

        # ------------------------------------------------------------------ #
        # Total trend rate (OLS on observed — comparable to other fitters)
        # ------------------------------------------------------------------ #
        t_idx = np.arange(n, dtype=float)
        X_trend = sm.add_constant(t_idx)
        ols_res = sm.OLS(y, X_trend).fit()
        beta_total = float(ols_res.params[1])
        total_trend_rate = float(np.exp(beta_total * self._periods_per_year) - 1.0)

        # ------------------------------------------------------------------ #
        # Model fit statistics
        # ------------------------------------------------------------------ #
        aic = float(res.aic)
        bic = float(res.bic)
        converged = bool(
            res.mle_retvals.get("converged", True)
            if isinstance(res.mle_retvals, dict)
            else True
        )

        # ------------------------------------------------------------------ #
        # Package result
        # ------------------------------------------------------------------ #
        periods_series = pl.Series("period", [str(p) for p in self._periods_raw])

        return InflationDecompositionResult(
            trend=pl.Series("trend", trend_arr),
            cycle=pl.Series("cycle", cycle_arr),
            seasonal=pl.Series("seasonal", seasonal_arr),
            irregular=pl.Series("irregular", irregular_arr),
            structural_rate=structural_rate,
            cyclical_position=cyclical_position,
            cycle_period=cycle_period_years,
            total_trend_rate=total_trend_rate,
            periods=periods_series,
            observations=pl.Series("observed", y),
            log_transform=self._log_transform,
            periods_per_year=self._periods_per_year,
            aic=aic,
            bic=bic,
            converged=converged,
            n_obs=n,
        )

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _get_smoothed(res: Any, component: str, n: int) -> np.ndarray:
        """Extract the Kalman-smoothed component array from the results object.

        Uses ``res.<component>.smoothed`` which is the canonical API in
        ``statsmodels.tsa.statespace.UnobservedComponentsResults``.

        Parameters
        ----------
        res:
            Fitted ``UnobservedComponentsResults``.
        component:
            One of ``'level'``, ``'trend'``, ``'cycle'``, ``'seasonal'``.
        n:
            Expected number of observations.

        Returns
        -------
        numpy.ndarray of shape (n,), zero-filled if the component is absent.
        """
        try:
            bunch = getattr(res, component, None)
            if bunch is None:
                return np.zeros(n)
            smoothed = getattr(bunch, "smoothed", None)
            if smoothed is None:
                return np.zeros(n)
            arr = np.asarray(smoothed, dtype=float).ravel()[:n]
            # Replace NaN with zero (can occur at diffuse initialisation boundary)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            # Pad if shorter than n (shouldn't happen, but defensive)
            if len(arr) < n:
                arr = np.concatenate([arr, np.zeros(n - len(arr))])
            return arr
        except Exception:
            return np.zeros(n)

    def _compute_structural_rate(self, res: Any, y: np.ndarray, n: int) -> float:
        """Compute the annualised structural trend rate from the smoothed slope.

        Uses the mean of the Kalman-smoothed slope state (``beta_t``) from
        ``res.smoother_results.smoothed_state[1, :]``. The slope state is at
        index 1 in all Harvey local linear trend specifications (level at 0,
        slope at 1, then cycle/seasonal states follow).

        Falls back to OLS on the trend component if the state vector cannot
        be read.

        Parameters
        ----------
        res:
            Fitted ``UnobservedComponentsResults``.
        y:
            Log-transformed (or raw) observed series.
        n:
            Number of observations.

        Returns
        -------
        Annualised structural trend rate as a decimal.
        """
        try:
            smoothed_state = res.smoother_results.smoothed_state
            # Shape: (n_states, n_obs). Slope is always at index 1 for a
            # local linear trend (level=True, trend=True) spec.
            if smoothed_state.shape[0] >= 2:
                slope_state = smoothed_state[1, :]
                mean_slope = float(np.nanmean(slope_state))
                return float(np.exp(mean_slope * self._periods_per_year) - 1.0)
        except Exception:
            pass

        # Fallback: OLS on the extracted trend component
        try:
            import statsmodels.api as sm
            trend_arr = self._get_smoothed(res, "level", n)
            t_idx = np.arange(len(trend_arr), dtype=float)
            X = sm.add_constant(t_idx)
            ols_r = sm.OLS(trend_arr, X).fit()
            return float(np.exp(float(ols_r.params[1]) * self._periods_per_year) - 1.0)
        except Exception:
            pass

        # Last resort: total OLS
        try:
            import statsmodels.api as sm
            t_idx = np.arange(n, dtype=float)
            X = sm.add_constant(t_idx)
            ols_r = sm.OLS(y, X).fit()
            return float(np.exp(float(ols_r.params[1]) * self._periods_per_year) - 1.0)
        except Exception:
            return float("nan")

    def _compute_cyclical_position(self, cycle_arr: np.ndarray) -> float:
        """Compute the cyclical position at the end of the sample.

        For log-transformed data the cycle is additive in log space, so:
            cyclical_position = exp(cycle[-1]) - 1

        This gives the fraction by which the current observation is elevated
        (positive) or depressed (negative) relative to the structural trend.

        For untransformed data, returns ``cycle[-1]`` directly (the raw
        additive deviation from the trend level).
        """
        valid_idx = np.where(np.isfinite(cycle_arr) & (cycle_arr != 0))[0]
        if len(valid_idx) == 0:
            return 0.0
        last_cycle = float(cycle_arr[valid_idx[-1]])

        if self._log_transform:
            return float(np.exp(last_cycle) - 1.0)
        return last_cycle

    def _compute_cycle_period(self, res: Any) -> float:
        """Extract the estimated cycle period in years from the fitted model.

        statsmodels stores the cycle frequency (lambda, radians per period) as
        the parameter named ``'frequency.cycle'``. Period in periods is
        ``2*pi / lambda``; dividing by ``periods_per_year`` gives years.

        Returns ``float('nan')`` if no cycle was fitted or extraction fails.
        """
        if not self._cycle:
            return float("nan")

        try:
            param_names = res.model.param_names
            params = res.params
            for name, val in zip(param_names, params):
                if "frequency" in name.lower():
                    lambda_rad = float(val)
                    if lambda_rad > 1e-10:
                        return float(2.0 * np.pi / lambda_rad / self._periods_per_year)
        except Exception:
            pass

        # Fallback: midpoint of the bounds
        lo, hi = self._cycle_period_bounds_periods
        return float((lo + hi) / 2.0 / self._periods_per_year)
