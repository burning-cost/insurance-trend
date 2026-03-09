"""insurance-trend: Loss cost trend analysis for UK personal lines insurance pricing.

The library provides four main classes:

- :class:`FrequencyTrendFitter` — log-linear trend fitting for claim frequency
  (claims per unit of exposure).
- :class:`SeverityTrendFitter` — log-linear trend fitting for claim severity
  (average cost per claim), with optional external index deflation and
  superimposed inflation calculation.
- :class:`LossCostTrendFitter` — combined frequency × severity analysis with
  :meth:`~LossCostTrendFitter.decompose` and projected loss cost.
- :class:`ExternalIndex` — ONS API fetcher and CSV loader, with a catalogue of
  UK insurance-relevant ONS series codes.

Results are returned as :class:`TrendResult` or :class:`LossCostTrendResult`
dataclasses. All outputs use Polars DataFrames/Series. Both pandas and Polars
inputs are accepted.

Quick start::

    from insurance_trend import LossCostTrendFitter

    fitter = LossCostTrendFitter(
        periods=['2020Q1', '2020Q2', '2021Q1', '2021Q2',
                 '2022Q1', '2022Q2', '2023Q1', '2023Q2'],
        claim_counts=[110, 115, 108, 112, 105, 109, 102, 107],
        earned_exposure=[1000, 1010, 1005, 1008, 1002, 1006, 1001, 1004],
        total_paid=[550000, 580000, 562000, 573000, 545000, 558000, 538000, 552000],
    )
    result = fitter.fit()
    print(result.summary())
    print(result.decompose())
"""

from .frequency import FrequencyTrendFitter
from .index import ExternalIndex
from .loss_cost import LossCostTrendFitter
from .result import LossCostTrendResult, TrendResult
from .severity import SeverityTrendFitter

__version__ = "0.1.0"

__all__ = [
    "FrequencyTrendFitter",
    "SeverityTrendFitter",
    "LossCostTrendFitter",
    "ExternalIndex",
    "TrendResult",
    "LossCostTrendResult",
    "__version__",
]
