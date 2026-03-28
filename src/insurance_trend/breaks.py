"""Structural break detection for insurance trend series.

Wraps the ``ruptures`` library with actuarial-friendly defaults and returns
break positions as integer indices into the input array.
"""

from __future__ import annotations

import warnings

import numpy as np


def detect_breakpoints(
    log_series: np.ndarray,
    penalty: float = 3.0,
    min_size: int = 3,
    max_breaks: int = 5,
) -> list[int]:
    """Detect structural break points in a log-transformed trend series.

    Uses the Pruned Exact Linear Time (PELT) algorithm with a radial basis
    function (RBF) cost applied to the *detrended* residuals of the input
    series. Detrending is critical: without it, RBF sees the steadily
    increasing values of a clean linear trend as a distributional shift and
    fires spurious breaks.

    The workflow is:
    1. Fit a global OLS linear trend: ``y_hat = a + b*t``.
    2. Compute residuals: ``e = log_series - y_hat``.
    3. Run PELT+RBF on the residuals.

    A clean linear trend has near-zero residuals, so PELT will not fire.  A
    genuine level shift appears as a step in the residuals and is reliably
    detected.

    RBF is still preferred over L2 (mean-shift) on residuals because
    insurance log-frequency residuals are typically not zero-mean within each
    segment — there is often a modest within-segment slope remaining after the
    global trend is removed.  RBF's sensitivity to distributional shape catches
    these shifts more reliably than L2.

    Parameters
    ----------
    log_series:
        Log-transformed series, e.g. ``log(frequency)`` or ``log(severity)``.
        Must be a 1-D numpy array with no NaN values.
    penalty:
        Penalty parameter for the PELT algorithm. Higher values produce fewer
        breaks. Default of 3.0 is a reasonable starting point for quarterly
        insurance data. For large step-changes (COVID lockdown magnitude,
        Ogden rate change) the break should fire at 3.0. Reduce to 1.5 to
        detect smaller shifts, or use ``changepoints=`` to impose known dates.
    min_size:
        Minimum number of observations in each segment. Prevents the algorithm
        from detecting breaks with too few data points on either side.
    max_breaks:
        Maximum number of breakpoints to return. If the algorithm detects more,
        only the first ``max_breaks`` are returned.

    Returns
    -------
    List of integer indices (0-based) where breaks occur. An empty list means
    no structural breaks were detected.

    Notes
    -----
    The ruptures library numbers breakpoints as the index of the *first*
    observation of the new segment. For example, a breakpoint of 8 means the
    series changes at index 8.

    The returned list excludes the implicit final breakpoint (length of series)
    that ruptures always appends.

    Why detrend before PELT?
        PELT with RBF is sensitive to changes in the local distribution, not
        just the mean. A clean linear ramp ``0.02 * t`` has steadily increasing
        values in the first half versus the second half — RBF reads this as a
        distributional shift and fires a spurious break even at moderate
        penalties. By removing the global linear trend first, the input to PELT
        is a residual series centred near zero, and a linear trend produces
        residuals that are near-zero everywhere. Only genuine level shifts or
        slope changes produce large residuals that trigger breaks.

    Why RBF over L2 on the residuals?
        PELT with ``model="l2"`` detects shifts in the *mean* level. For a
        log-frequency series with a +3% pa within-segment slope, the pre- and
        post-break means can overlap even when the step-change is large,
        causing the L2 model to miss the break entirely. The RBF kernel is
        sensitive to changes in the local distribution shape and reliably
        detects the kind of -35% level shifts seen in UK motor frequency
        during the 2020 COVID lockdown.
    """
    try:
        import ruptures as rpt
    except ImportError as exc:
        raise ImportError(
            "The 'ruptures' package is required for structural break detection. "
            "Install it with: pip install ruptures"
        ) from exc

    n = len(log_series)
    if n < 2 * min_size:
        return []

    # Detrend: remove a global OLS linear trend before running PELT.
    # Without this, RBF fires on the steadily increasing values of a linear
    # series because the left-half distribution differs from the right-half
    # distribution even when there is no structural break.
    t = np.arange(n, dtype=float)
    # OLS fit: [1, t] design matrix
    A = np.column_stack([np.ones(n), t])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_series, rcond=None)
    trend = A @ coeffs
    residuals = log_series - trend

    # PELT requires a 2-D array
    signal = residuals.reshape(-1, 1)

    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        raw_breaks = algo.predict(pen=penalty)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Breakpoint detection failed with ruptures: {exc}. "
            "Proceeding without structural break detection.",
            stacklevel=2,
        )
        return []

    # ruptures appends len(series) as a sentinel; drop it
    breaks = [b for b in raw_breaks if b < n]

    # Limit to max_breaks
    if len(breaks) > max_breaks:
        breaks = breaks[:max_breaks]

    return breaks


def split_segments(
    t: np.ndarray,
    y: np.ndarray,
    breakpoints: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split time and value arrays at the given breakpoints.

    Parameters
    ----------
    t:
        Integer time index array (0-based).
    y:
        Value array (same length as ``t``).
    breakpoints:
        Sorted list of integer breakpoint indices.

    Returns
    -------
    List of (t_segment, y_segment) tuples, one per segment.
    """
    if not breakpoints:
        return [(t, y)]

    # Ensure sorted and add sentinel
    bps = sorted(breakpoints) + [len(t)]
    segments = []
    start = 0
    for bp in bps:
        segments.append((t[start:bp], y[start:bp]))
        start = bp
    return [(seg_t, seg_y) for seg_t, seg_y in segments if len(seg_t) > 0]
