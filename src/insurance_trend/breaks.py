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

    Uses the Pruned Exact Linear Time (PELT) algorithm with an L2 cost function.
    Returns integer indices at which the series changes slope.

    Parameters
    ----------
    log_series:
        Log-transformed series, e.g. ``log(frequency)`` or ``log(severity)``.
        Must be a 1-D numpy array with no NaN values.
    penalty:
        Penalty parameter for the PELT algorithm. Higher values produce fewer
        breaks. Default of 3.0 is a reasonable starting point for quarterly
        insurance data; reduce to 1.5 to increase sensitivity.
    min_size:
        Minimum number of observations in each segment. Prevents the algorithm
        from detecting breaks with too few data points on either side.
    max_breaks:
        Maximum number of breakpoints to return. If the algorithm detects more,
        only the top ``max_breaks`` (by signal contrast) are returned.

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

    # PELT requires a 2-D array
    signal = log_series.reshape(-1, 1)

    try:
        algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
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
