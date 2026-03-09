"""Diagnostic plotting for insurance trend results.

All functions return matplotlib Figure objects — they do not call ``plt.show()``.
Callers are responsible for displaying or saving figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    from .result import LossCostTrendResult, TrendResult


def trend_diagnostic_plot(result: "TrendResult") -> "matplotlib.figure.Figure":
    """Three-panel diagnostic figure for a single-component TrendResult.

    Panel 1 — Actual vs Fitted values on the original scale.
    Panel 2 — Multiplicative residuals (actual/fitted - 1) with zero reference.
    Panel 3 — Forward projection fan with 90 % and 95 % confidence intervals.

    Parameters
    ----------
    result:
        A ``TrendResult`` instance returned by a fitter.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.subplots_adjust(hspace=0.45)

    actuals = result.actuals.to_numpy()
    fitted = result.fitted_values.to_numpy()
    residuals = result.residuals.to_numpy()
    n = len(actuals)
    t_hist = np.arange(n)

    periods_labels = result.periods.to_list() if len(result.periods) > 0 else list(range(n))

    # ------------------------------------------------------------------ #
    # Panel 1: Actual vs Fitted
    # ------------------------------------------------------------------ #
    ax1 = axes[0]
    ax1.plot(t_hist, actuals, "o-", color="#1f77b4", label="Actual", linewidth=1.5, markersize=4)
    ax1.plot(t_hist, fitted, "--", color="#d62728", label="Fitted", linewidth=1.5)

    # Mark changepoints
    for cp in result.changepoints:
        if 0 <= cp < n:
            ax1.axvline(cp, color="grey", linestyle=":", linewidth=1, alpha=0.7)

    ax1.set_title(
        f"Actual vs Fitted  |  Trend: {result.trend_rate:.2%} pa  "
        f"(R² = {result.r_squared:.3f})",
        fontsize=10,
    )
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    _label_axis_periods(ax1, t_hist, periods_labels)

    # ------------------------------------------------------------------ #
    # Panel 2: Residuals
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    ax2.bar(t_hist, residuals * 100, color="#ff7f0e", alpha=0.7, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Residuals (actual / fitted − 1)", fontsize=10)
    ax2.set_ylabel("Residual (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    _label_axis_periods(ax2, t_hist, periods_labels)

    # ------------------------------------------------------------------ #
    # Panel 3: Projection fan
    # ------------------------------------------------------------------ #
    ax3 = axes[2]

    if len(result.projection) > 0:
        proj = result.projection
        fwd_t = np.arange(n, n + len(proj))
        pt = proj["point"].to_numpy()
        lo = proj["lower"].to_numpy()
        hi = proj["upper"].to_numpy()

        # Historical fitted line, leading into projection
        ax3.plot(t_hist, fitted, "-", color="#d62728", linewidth=1.5, label="Fitted (historical)")
        ax3.plot(t_hist, actuals, "o", color="#1f77b4", markersize=4, label="Actual")

        # Projection fan
        ax3.plot(fwd_t, pt, "--", color="#d62728", linewidth=1.5, label="Projection")
        ax3.fill_between(fwd_t, lo, hi, alpha=0.2, color="#d62728", label="95% CI")

        # Bridge from last fitted to first projection
        bridge_t = [t_hist[-1], fwd_t[0]]
        bridge_y = [fitted[-1], pt[0]]
        ax3.plot(bridge_t, bridge_y, "--", color="#d62728", linewidth=1.5)

        ax3.axvline(n - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No projection available", transform=ax3.transAxes, ha="center")

    ax3.set_title("Forward Projection with Confidence Fan", fontsize=10)
    ax3.set_ylabel("Value")
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))

    fig.suptitle(
        f"Trend Diagnostic  |  Method: {result.method}  |  "
        f"95% CI: ({result.ci_lower:.2%}, {result.ci_upper:.2%})",
        fontsize=11,
        fontweight="bold",
    )

    return fig


def loss_cost_diagnostic_plot(result: "LossCostTrendResult") -> "matplotlib.figure.Figure":
    """Multi-panel diagnostic figure for a LossCostTrendResult.

    Shows frequency trend, severity trend, and combined loss cost projection
    side by side.

    Parameters
    ----------
    result:
        A ``LossCostTrendResult`` instance.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    _plot_component(axes[0, 0], result.frequency, "Frequency")
    _plot_component(axes[0, 1], result.severity, "Severity")
    _plot_projection(axes[1, 0], result.frequency, "Frequency Projection")
    _plot_projection(axes[1, 1], result.severity, "Severity Projection")

    si_str = (
        f"Superimposed inflation: {result.superimposed_inflation:.2%}"
        if result.superimposed_inflation is not None
        else "No external index"
    )

    fig.suptitle(
        f"Loss Cost Trend  |  Freq: {result.frequency.trend_rate:.2%}  "
        f"× Sev: {result.severity.trend_rate:.2%}  "
        f"= Combined: {result.combined_trend_rate:.2%}  |  {si_str}",
        fontsize=11,
        fontweight="bold",
    )
    return fig


# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #

def _label_axis_periods(ax, t: np.ndarray, labels: list) -> None:
    """Apply period labels to the x-axis if they are strings."""
    if labels and isinstance(labels[0], str):
        step = max(1, len(t) // 8)
        ticks = t[::step]
        tick_labels = [str(labels[i]) for i in range(0, len(labels), step)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)


def _plot_component(ax, result: "TrendResult", title: str) -> None:
    """Plot actual vs fitted for a single component."""
    import matplotlib.ticker as mticker

    actuals = result.actuals.to_numpy()
    fitted = result.fitted_values.to_numpy()
    n = len(actuals)
    t = np.arange(n)

    ax.plot(t, actuals, "o-", color="#1f77b4", label="Actual", linewidth=1.5, markersize=4)
    ax.plot(t, fitted, "--", color="#d62728", label="Fitted", linewidth=1.5)
    for cp in result.changepoints:
        if 0 <= cp < n:
            ax.axvline(cp, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_title(
        f"{title}  |  Trend: {result.trend_rate:.2%} pa  (R²={result.r_squared:.3f})",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))


def _plot_projection(ax, result: "TrendResult", title: str) -> None:
    """Plot the forward projection fan for a single component."""
    import matplotlib.ticker as mticker

    if len(result.projection) == 0:
        ax.text(0.5, 0.5, "No projection", transform=ax.transAxes, ha="center")
        ax.set_title(title, fontsize=9)
        return

    fitted = result.fitted_values.to_numpy()
    n = len(fitted)
    t_hist = np.arange(n)
    proj = result.projection
    fwd_t = np.arange(n, n + len(proj))
    pt = proj["point"].to_numpy()
    lo = proj["lower"].to_numpy()
    hi = proj["upper"].to_numpy()

    ax.plot(t_hist, fitted, "-", color="#d62728", linewidth=1.5)
    ax.plot(fwd_t, pt, "--", color="#d62728", linewidth=1.5, label="Point")
    ax.fill_between(fwd_t, lo, hi, alpha=0.2, color="#d62728", label="95% CI")
    ax.axvline(n - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
