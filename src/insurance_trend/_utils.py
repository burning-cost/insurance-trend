"""Shared utilities for insurance-trend."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import polars as pl


# Type aliases accepted as inputs
PandasOrPolars = Union[pd.Series, pl.Series, list, np.ndarray]


def to_numpy(x: PandasOrPolars, name: str = "input") -> np.ndarray:
    """Convert any supported array-like to a NumPy array.

    Accepts pandas Series, Polars Series, lists, and numpy arrays.
    """
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, na_value=np.nan)
    if isinstance(x, pl.Series):
        return x.to_numpy()
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)
    raise TypeError(
        f"{name} must be a pandas Series, Polars Series, list, or numpy array; "
        f"got {type(x).__name__}"
    )


def to_polars_series(x: PandasOrPolars, name: str = "series") -> pl.Series:
    """Convert any supported array-like to a Polars Series."""
    if isinstance(x, pl.Series):
        return x
    if isinstance(x, pd.Series):
        return pl.from_pandas(x.rename(name))
    arr = to_numpy(x, name=name)
    return pl.Series(name=name, values=arr)


def periods_to_index(periods: PandasOrPolars) -> np.ndarray:
    """Convert a period array to integer time indices (0, 1, 2, ...).

    Accepts numeric indices, pandas PeriodIndex/DatetimeIndex, or string
    representations of quarters (e.g. '2020Q1', '2020-Q1').
    """
    if isinstance(periods, (pd.Series, pd.Index)):
        vals = periods.values
    elif isinstance(periods, pl.Series):
        vals = periods.to_numpy()
    else:
        vals = np.asarray(periods)

    # If already numeric, return as integer index
    if np.issubdtype(vals.dtype, np.number):
        return np.arange(len(vals), dtype=float)

    # Otherwise treat as ordered categories, return ordinal positions
    return np.arange(len(vals), dtype=float)


def quarter_dummies(n: int, periods: np.ndarray | None = None) -> np.ndarray:
    """Build quarterly seasonal dummy matrix (Q1, Q2, Q3 dummies; Q4 is the base).

    Parameters
    ----------
    n:
        Number of observations.
    periods:
        Optional 0-based integer period indices. If None, assumes data start at Q1.

    Returns
    -------
    Array of shape (n, 3) with columns for Q1, Q2, Q3 indicators.
    """
    if periods is None:
        periods = np.arange(n)
    # Quarter cycles from 0 (Q1) to 3 (Q4)
    quarters = (periods % 4).astype(int)
    dummies = np.zeros((n, 3), dtype=float)
    for q in range(3):  # Q1=0, Q2=1, Q3=2
        dummies[:, q] = (quarters == q).astype(float)
    return dummies


def annual_trend_rate(beta: float, periods_per_year: int = 4) -> float:
    """Convert a per-period log-linear slope to an annual trend rate.

    Parameters
    ----------
    beta:
        Estimated slope in log space, per period.
    periods_per_year:
        Number of periods per year (4 for quarterly, 12 for monthly).

    Returns
    -------
    Annual trend rate as a decimal, e.g. 0.085 for 8.5 % per annum.
    """
    return float(np.exp(beta * periods_per_year) - 1)


def validate_lengths(**arrays: PandasOrPolars) -> int:
    """Validate that all provided arrays have the same length. Returns the length."""
    lengths = {}
    for name, arr in arrays.items():
        if isinstance(arr, (pd.Series, pl.Series)):
            lengths[name] = len(arr)
        else:
            lengths[name] = len(np.asarray(arr))
    unique = set(lengths.values())
    if len(unique) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(f"All input arrays must have the same length; got {detail}")
    return unique.pop()


def safe_log(y: np.ndarray, label: str = "values") -> np.ndarray:
    """Take natural log of y, raising a clear error if non-positive values present."""
    if np.any(y <= 0):
        bad = np.sum(y <= 0)
        raise ValueError(
            f"{label} contains {bad} non-positive value(s). "
            "Log-linear trend fitting requires strictly positive values."
        )
    return np.log(y.astype(float))
