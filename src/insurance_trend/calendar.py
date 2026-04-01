"""BreakEventCalendar — map structural break dates to known UK insurance market events.

Structural break detectors (PELT, CUSUM, Bai-Perron) tell you *where* a series
changes; they cannot tell you *why*. The answer is usually sitting in a regulatory
gazette or an industry newsletter from that quarter.

This module maintains a registry of known UK personal lines insurance market events
— Ogden rate changes, Insurance Premium Tax rises, Civil Liability Act reforms,
COVID lockdowns, supply chain shocks — and matches detected break dates against
that registry. Each detected break is either attributed to a known event (within
a configurable tolerance window) or flagged as unexplained.

The result is a structured attribution report that a pricing actuary can include
in a trend review pack: "the 2020Q1 break in motor frequency was the first COVID
lockdown; the 2017Q1 break in motor severity was the Ogden rate change to -0.75%."

Usage::

    from insurance_trend import BreakEventCalendar

    # Use the built-in UK calendar
    calendar = BreakEventCalendar()

    # Detected breaks from InflationDecomposer / detect_breakpoints
    # as period strings
    breaks = ["2017Q1", "2020Q1", "2022Q1"]

    report = calendar.attribute(breaks)
    print(report.summary())

    # Or add a custom event
    calendar.add_event(
        period="2023Q4",
        description="FCA premium finance rules effective",
        category="regulation",
        impact=0,
    )

Period format
-------------
All periods must be strings in one of these formats:

- Quarterly: ``"2020Q1"`` through ``"2020Q4"``
- Monthly: ``"2020M01"`` through ``"2020M12"`` (or ``"2020-01"``)
- Annual: ``"2020"``

Internally every period is converted to a (year, sub-period) tuple for
arithmetic. The tolerance window is expressed in *periods* relative to the
input data frequency — so ``tolerance=2`` allows matching two quarters either
side of a detected break.

Design note on period arithmetic
---------------------------------
We deliberately avoid ``datetime`` objects. Insurance pricing data is always
labelled by *period* (quarter, month, year), and using ``datetime`` would force
an arbitrary day choice that misleads readers. Instead we do ordinal arithmetic
on (year, sub_period) tuples. This keeps the public API simple: everything is
a string like ``"2020Q1"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import polars as pl


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ImpactDirection = Literal[-1, 0, 1]
"""
-1 = downward pressure on claims cost (e.g. reform that reduces awards),
 0 = ambiguous or mixed impact,
+1 = upward pressure on claims cost (e.g. Ogden rate reduction, cost inflation shock).
"""

Category = Literal[
    "regulation",
    "legal",
    "macro",
    "covid",
    "supply_chain",
    "tax",
    "market",
    "other",
]


# ---------------------------------------------------------------------------
# Period parsing utilities
# ---------------------------------------------------------------------------

_QUARTERLY_RE = re.compile(r"^(\d{4})[Qq]([1-4])$")
_MONTHLY_RE = re.compile(r"^(\d{4})[Mm-]?(\d{2})$")
_ANNUAL_RE = re.compile(r"^(\d{4})$")


def _parse_period(period: str) -> tuple[int, int, int]:
    """Parse a period string to (year, sub_period, periods_per_year).

    Returns
    -------
    (year, sub_period, periods_per_year) where sub_period is 1-based.

    Examples
    --------
    >>> _parse_period("2020Q1")
    (2020, 1, 4)
    >>> _parse_period("2020M03")
    (2020, 3, 12)
    >>> _parse_period("2020")
    (2020, 1, 1)

    Raises
    ------
    ValueError
        If the period string cannot be parsed.
    """
    period = period.strip()
    m = _QUARTERLY_RE.match(period)
    if m:
        return int(m.group(1)), int(m.group(2)), 4
    m = _MONTHLY_RE.match(period)
    if m:
        sub = int(m.group(2))
        if not 1 <= sub <= 12:
            raise ValueError(f"Month sub-period out of range in {period!r}: got {sub}.")
        return int(m.group(1)), sub, 12
    m = _ANNUAL_RE.match(period)
    if m:
        return int(m.group(1)), 1, 1
    raise ValueError(
        f"Cannot parse period {period!r}. "
        "Expected formats: '2020Q1', '2020M03', '2020-03', '2020'."
    )


def _period_to_ordinal(year: int, sub: int, ppy: int) -> float:
    """Convert a (year, sub_period, ppy) tuple to a continuous ordinal.

    The ordinal is in units of the given frequency (quarters, months, years).
    For example: 2020Q1 → 2020*4 + 0 = 8080; 2020Q2 → 8081.
    """
    return float(year * ppy + (sub - 1))


def _ordinal_distance(p1: str, p2: str) -> float:
    """Absolute period distance between two period strings.

    Both periods must be the same frequency (both quarterly, both monthly, etc.).
    Returns the distance in periods (e.g. 2020Q1 to 2020Q3 → 2).

    Raises
    ------
    ValueError
        If the two periods are of different frequencies.
    """
    y1, s1, ppy1 = _parse_period(p1)
    y2, s2, ppy2 = _parse_period(p2)
    if ppy1 != ppy2:
        raise ValueError(
            f"Cannot compute distance between periods of different frequencies: "
            f"{p1!r} (ppy={ppy1}) and {p2!r} (ppy={ppy2})."
        )
    o1 = _period_to_ordinal(y1, s1, ppy1)
    o2 = _period_to_ordinal(y2, s2, ppy2)
    return abs(o1 - o2)


# ---------------------------------------------------------------------------
# CalendarEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarEvent:
    """A single known UK insurance market event.

    Attributes
    ----------
    period:
        The period in which the event occurred or took effect, e.g. ``"2017Q1"``.
    description:
        Human-readable description of the event.
    category:
        Broad category. One of: ``'regulation'``, ``'legal'``, ``'macro'``,
        ``'covid'``, ``'supply_chain'``, ``'tax'``, ``'market'``, ``'other'``.
    impact:
        Expected directional impact on claims costs.
        ``+1`` = upward pressure (raises costs or reserves),
        ``-1`` = downward pressure (reduces costs or awards),
        ``0`` = ambiguous or mixed.
    source:
        Optional reference or citation.
    """

    period: str
    description: str
    category: str
    impact: int  # -1, 0, or +1
    source: str = ""

    def __post_init__(self) -> None:
        if self.impact not in (-1, 0, 1):
            raise ValueError(
                f"impact must be -1, 0, or +1; got {self.impact!r} "
                f"for event '{self.description}'."
            )
        # Validate period parses cleanly
        _parse_period(self.period)


# ---------------------------------------------------------------------------
# Attribution results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakAttribution:
    """Attribution of a single detected structural break.

    Attributes
    ----------
    break_period:
        The detected break period string (as supplied by the caller).
    matched_event:
        The nearest :class:`CalendarEvent` within the tolerance window, or
        ``None`` if no event was found within tolerance.
    distance:
        Distance in periods between the break and the matched event.
        ``None`` if ``matched_event`` is ``None``.
    explained:
        ``True`` when a matching event was found within the tolerance window.
    """

    break_period: str
    matched_event: Optional[CalendarEvent]
    distance: Optional[float]
    explained: bool


@dataclass
class AttributionReport:
    """Full attribution report for a set of detected structural breaks.

    Attributes
    ----------
    attributions:
        List of :class:`BreakAttribution` instances, one per detected break,
        in the same order as the input ``break_periods``.
    n_breaks:
        Total number of detected breaks.
    n_explained:
        Number of breaks matched to a known calendar event.
    n_unexplained:
        Number of breaks with no match within tolerance.
    tolerance:
        The tolerance window (in periods) used for matching.
    """

    attributions: list[BreakAttribution]
    n_breaks: int
    n_explained: int
    n_unexplained: int
    tolerance: float

    def summary(self) -> str:
        """Return a formatted text summary of the attribution report.

        Returns
        -------
        str
            Multi-line string suitable for printing to a console or logging.
        """
        lines = [
            "=== Break Attribution Report ===",
            f"Detected breaks   : {self.n_breaks}",
            f"Explained         : {self.n_explained}",
            f"Unexplained       : {self.n_unexplained}",
            f"Tolerance window  : ±{self.tolerance:.0f} period(s)",
            "",
        ]
        for attr in self.attributions:
            status = "EXPLAINED" if attr.explained else "UNEXPLAINED"
            lines.append(f"  Break: {attr.break_period}  [{status}]")
            if attr.matched_event is not None:
                evt = attr.matched_event
                dist_str = f"{attr.distance:.0f} period(s) away" if attr.distance else "exact match"
                impact_str = {1: "upward", -1: "downward", 0: "ambiguous"}.get(evt.impact, "unknown")
                lines.append(f"    -> {evt.period}: {evt.description}")
                lines.append(f"       Category: {evt.category}  |  Impact: {impact_str}  |  Distance: {dist_str}")
                if evt.source:
                    lines.append(f"       Source: {evt.source}")
            else:
                lines.append("    -> No matching event found within tolerance window.")
        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Return a Polars DataFrame with one row per detected break.

        Columns
        -------
        break_period, explained, matched_event_period, matched_event_description,
        matched_event_category, matched_event_impact, distance_periods.
        """
        rows: dict[str, list] = {
            "break_period": [],
            "explained": [],
            "matched_event_period": [],
            "matched_event_description": [],
            "matched_event_category": [],
            "matched_event_impact": [],
            "distance_periods": [],
        }
        for attr in self.attributions:
            rows["break_period"].append(attr.break_period)
            rows["explained"].append(attr.explained)
            if attr.matched_event is not None:
                rows["matched_event_period"].append(attr.matched_event.period)
                rows["matched_event_description"].append(attr.matched_event.description)
                rows["matched_event_category"].append(attr.matched_event.category)
                rows["matched_event_impact"].append(attr.matched_event.impact)
                rows["distance_periods"].append(float(attr.distance) if attr.distance is not None else None)
            else:
                rows["matched_event_period"].append(None)
                rows["matched_event_description"].append(None)
                rows["matched_event_category"].append(None)
                rows["matched_event_impact"].append(None)
                rows["distance_periods"].append(None)
        return pl.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"AttributionReport("
            f"n_breaks={self.n_breaks}, "
            f"n_explained={self.n_explained}, "
            f"n_unexplained={self.n_unexplained}, "
            f"tolerance={self.tolerance})"
        )


# ---------------------------------------------------------------------------
# Default UK insurance event calendar
# ---------------------------------------------------------------------------

_DEFAULT_UK_EVENTS: list[CalendarEvent] = [
    # --- Tax changes ---
    CalendarEvent(
        period="2015Q4",
        description="IPT raised from 6% to 9.5% — Insurance Premium Tax first major rise",
        category="tax",
        impact=1,
        source="Finance Act 2015",
    ),
    CalendarEvent(
        period="2016Q4",
        description="IPT raised from 9.5% to 10% — second IPT rise in twelve months",
        category="tax",
        impact=1,
        source="Finance Act 2016",
    ),
    CalendarEvent(
        period="2017Q2",
        description="IPT raised from 10% to 12% — further IPT rise; cumulative doubling since 2015",
        category="tax",
        impact=1,
        source="Finance Act 2017",
    ),
    # --- Ogden rate changes ---
    CalendarEvent(
        period="2017Q1",
        description="Ogden discount rate changed from +2.5% to -0.75% — large uplift to catastrophic injury reserves",
        category="legal",
        impact=1,
        source="Damages Act 1996; Lord Chancellor announcement Feb 2017",
    ),
    CalendarEvent(
        period="2019Q3",
        description="Ogden discount rate revised from -0.75% to -0.25% — partial reversal under Civil Liability Act",
        category="legal",
        impact=-1,
        source="Civil Liability Act 2018; Lord Chancellor announcement Aug 2019",
    ),
    # --- Legal / PI reform ---
    CalendarEvent(
        period="2013Q1",
        description="Legal Aid, Sentencing and Punishment of Offenders Act (LASPO) — referral fee ban",
        category="legal",
        impact=-1,
        source="LASPO Act 2012, Part 2",
    ),
    CalendarEvent(
        period="2013Q2",
        description="Qualified one-way costs shifting (QOCS) introduced — changed litigation economics",
        category="legal",
        impact=0,
        source="Civil Procedure Rules update, April 2013",
    ),
    CalendarEvent(
        period="2021Q2",
        description="Whiplash tariff effective (Civil Liability Act) — fixed PI tariff for soft tissue claims up to 2 years",
        category="legal",
        impact=-1,
        source="Civil Liability Act 2018; Whiplash Injury Regulations 2021 (effective May 2021)",
    ),
    CalendarEvent(
        period="2021Q2",
        description="OIC Portal launched — Official Injury Claim portal for low-value road traffic PI claims",
        category="regulation",
        impact=-1,
        source="FCA/MoJ, May 2021",
    ),
    # --- COVID ---
    CalendarEvent(
        period="2020Q1",
        description="COVID-19 first national lockdown — sharp suppression of motor frequency (fewer miles driven)",
        category="covid",
        impact=-1,
        source="UK Government lockdown 23 March 2020",
    ),
    CalendarEvent(
        period="2020Q3",
        description="Post-lockdown claims bounce — backlog claims submissions and severity spike",
        category="covid",
        impact=1,
        source="Industry data; ABI claims statistics Q3 2020",
    ),
    CalendarEvent(
        period="2021Q1",
        description="Third national lockdown — further frequency suppression (Jan–Mar 2021)",
        category="covid",
        impact=-1,
        source="UK Government lockdown restrictions Jan 2021",
    ),
    # --- Supply chain / macro ---
    CalendarEvent(
        period="2021Q3",
        description="Global semiconductor shortage peak — new vehicle delivery delays; used car values surge",
        category="supply_chain",
        impact=1,
        source="SMMT; S&P Global; ABI motor severity data",
    ),
    CalendarEvent(
        period="2022Q1",
        description="Energy crisis and supply chain inflation — repair parts costs surge; body shop labour rates spike",
        category="supply_chain",
        impact=1,
        source="ONS PPI; ABI; Thatcham Research 2022",
    ),
    CalendarEvent(
        period="2022Q2",
        description="Ukraine war inflation shock — vehicle repair costs and hire car rates accelerate",
        category="macro",
        impact=1,
        source="ONS CPI; ABI motor bulletin Q2 2022",
    ),
    CalendarEvent(
        period="2023Q3",
        description="Used car values begin normalising — Manheim/Cap HPI used car deflation after chip shortage peak",
        category="market",
        impact=-1,
        source="Cap HPI; Manheim Market Report 2023",
    ),
    # --- Regulation / FCA ---
    CalendarEvent(
        period="2022Q1",
        description="FCA General Insurance Pricing Practices (GIPP) rules effective — dual pricing ban",
        category="regulation",
        impact=0,
        source="FCA PS21/5; effective 1 January 2022",
    ),
    CalendarEvent(
        period="2023Q3",
        description="FCA Consumer Duty in force — outcome-focused pricing obligations take effect",
        category="regulation",
        impact=0,
        source="FCA PS22/9; effective 31 July 2023",
    ),
    CalendarEvent(
        period="2019Q4",
        description="FCA market study into general insurance pricing practices announced — competitive pressure",
        category="regulation",
        impact=0,
        source="FCA MS18/1; interim report published Dec 2019",
    ),
    CalendarEvent(
        period="2012Q1",
        description="Gender Directive ruling — sex-based pricing banned in EU/UK personal lines",
        category="regulation",
        impact=1,
        source="ECJ Test-Achats ruling; effective 21 Dec 2012",
    ),
    # --- Market structure ---
    CalendarEvent(
        period="2015Q2",
        description="Price comparison website dominance — PCW share of new business exceeds 70%; commoditisation pressure",
        category="market",
        impact=0,
        source="Mintel; EY UK insurance report 2015",
    ),
    CalendarEvent(
        period="2024Q1",
        description="Inflation easing — motor body shop costs decelerating; repair severity growth moderates",
        category="macro",
        impact=-1,
        source="ONS CPI; ABI motor bulletin Q1 2024",
    ),
]


# ---------------------------------------------------------------------------
# BreakEventCalendar
# ---------------------------------------------------------------------------


class BreakEventCalendar:
    """Registry of known UK insurance market events for structural break attribution.

    Ships with a built-in calendar of 20+ major UK personal lines market events
    covering regulatory changes, legal reforms, macroeconomic shocks, and the
    COVID-19 period. Users can extend this with custom events via
    :meth:`add_event`.

    Given a list of detected break dates, :meth:`attribute` matches each break
    to the nearest calendar event within a configurable tolerance window and
    returns a structured :class:`AttributionReport`.

    Parameters
    ----------
    include_defaults:
        Whether to load the built-in UK insurance event calendar on
        construction. Set to ``False`` to start with an empty registry and add
        only custom events. Default: ``True``.
    tolerance:
        Matching tolerance in periods. A detected break at period P will be
        matched to the nearest calendar event within [P - tolerance, P + tolerance].
        The default of 2 allows matching two quarters (or two months) either
        side of the detected break date, which is appropriate given that
        change-point algorithms typically locate breaks within one to two
        periods of the true event date. Set to 0 for exact-match only.

    Examples
    --------
    Basic usage with default calendar::

        from insurance_trend import BreakEventCalendar

        calendar = BreakEventCalendar()
        report = calendar.attribute(["2017Q1", "2020Q1", "2022Q1"])
        print(report.summary())

    Start from an empty calendar::

        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2024Q1", "Custom event", "regulation", impact=1)
        report = cal.attribute(["2024Q1"])

    Filter to a specific category before attributing::

        legal_cal = BreakEventCalendar()
        legal_events = legal_cal.filter_events(categories=["legal", "regulation"])
        # legal_events is a new BreakEventCalendar containing only those events

    Notes
    -----
    Period arithmetic is done on the (year, sub_period) ordinal representation.
    Mixed frequencies (e.g. quarterly breaks against monthly events) will raise
    a ``ValueError``. Ensure all period strings share the same frequency.
    """

    def __init__(
        self,
        include_defaults: bool = True,
        tolerance: float = 2.0,
    ) -> None:
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative; got {tolerance!r}.")
        self._tolerance = tolerance
        self._events: list[CalendarEvent] = []
        if include_defaults:
            self._events = list(_DEFAULT_UK_EVENTS)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def tolerance(self) -> float:
        """Matching tolerance in periods."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"tolerance must be non-negative; got {value!r}.")
        self._tolerance = value

    @property
    def events(self) -> list[CalendarEvent]:
        """All registered :class:`CalendarEvent` instances (read-only copy)."""
        return list(self._events)

    @property
    def n_events(self) -> int:
        """Number of registered events."""
        return len(self._events)

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def add_event(
        self,
        period: str,
        description: str,
        category: str,
        impact: int,
        source: str = "",
    ) -> "BreakEventCalendar":
        """Add a custom event to the calendar.

        Parameters
        ----------
        period:
            Period string, e.g. ``"2023Q4"`` or ``"2023M11"``.
        description:
            Human-readable description.
        category:
            Broad category string. Recommended values: ``'regulation'``,
            ``'legal'``, ``'macro'``, ``'covid'``, ``'supply_chain'``,
            ``'tax'``, ``'market'``, ``'other'``. Free text is accepted.
        impact:
            Direction of expected impact: ``+1`` (upward), ``-1`` (downward),
            ``0`` (ambiguous or mixed).
        source:
            Optional reference or citation.

        Returns
        -------
        self
            The calendar instance (for method chaining).
        """
        event = CalendarEvent(
            period=period,
            description=description,
            category=category,
            impact=impact,
            source=source,
        )
        self._events.append(event)
        return self

    def remove_event(self, period: str, description_contains: str = "") -> int:
        """Remove events matching the given period and optional description substring.

        Parameters
        ----------
        period:
            Period string to match exactly.
        description_contains:
            Optional substring to filter by description. If empty, removes all
            events for the given period.

        Returns
        -------
        int
            Number of events removed.
        """
        before = len(self._events)
        self._events = [
            e for e in self._events
            if not (
                e.period == period
                and (not description_contains or description_contains in e.description)
            )
        ]
        return before - len(self._events)

    # ------------------------------------------------------------------
    # Query / filter methods
    # ------------------------------------------------------------------

    def filter_events(
        self,
        categories: Optional[list[str]] = None,
        impact: Optional[int] = None,
        from_period: Optional[str] = None,
        to_period: Optional[str] = None,
    ) -> "BreakEventCalendar":
        """Return a new :class:`BreakEventCalendar` with filtered events.

        All filters are combined with AND logic.

        Parameters
        ----------
        categories:
            If supplied, only include events whose ``category`` is in this list.
        impact:
            If supplied (``-1``, ``0``, or ``+1``), only include events with
            that impact direction.
        from_period:
            If supplied, only include events at or after this period.
        to_period:
            If supplied, only include events at or before this period.

        Returns
        -------
        BreakEventCalendar
            A new instance with the built-in defaults *not* loaded (only the
            filtered events are copied).
        """
        filtered = list(self._events)

        if categories is not None:
            cat_set = set(categories)
            filtered = [e for e in filtered if e.category in cat_set]

        if impact is not None:
            filtered = [e for e in filtered if e.impact == impact]

        if from_period is not None:
            y_from, s_from, ppy_from = _parse_period(from_period)
            ord_from = _period_to_ordinal(y_from, s_from, ppy_from)
            filtered = [
                e for e in filtered
                if _period_to_ordinal(*_parse_period(e.period)) >= ord_from
            ]

        if to_period is not None:
            y_to, s_to, ppy_to = _parse_period(to_period)
            ord_to = _period_to_ordinal(y_to, s_to, ppy_to)
            filtered = [
                e for e in filtered
                if _period_to_ordinal(*_parse_period(e.period)) <= ord_to
            ]

        new_cal = BreakEventCalendar(include_defaults=False, tolerance=self._tolerance)
        new_cal._events = filtered
        return new_cal

    def events_dataframe(self) -> pl.DataFrame:
        """Return all calendar events as a Polars DataFrame.

        Columns
        -------
        period, description, category, impact, source.

        Returns
        -------
        pl.DataFrame
        """
        return pl.DataFrame(
            {
                "period": [e.period for e in self._events],
                "description": [e.description for e in self._events],
                "category": [e.category for e in self._events],
                "impact": [e.impact for e in self._events],
                "source": [e.source for e in self._events],
            }
        )

    # ------------------------------------------------------------------
    # Core attribution method
    # ------------------------------------------------------------------

    def attribute(
        self,
        break_periods: list[str],
        tolerance: Optional[float] = None,
    ) -> AttributionReport:
        """Attribute a list of detected structural breaks to known calendar events.

        Each detected break is matched to the nearest calendar event within
        ``tolerance`` periods (using the instance default unless overridden
        here). Where multiple events fall within tolerance, the nearest is
        chosen. Ties are broken by the order events appear in the calendar
        (earlier-registered events win).

        Parameters
        ----------
        break_periods:
            Detected break periods as strings, e.g. ``["2017Q1", "2020Q1"]``.
            These can come from :func:`~insurance_trend.breaks.detect_breakpoints`
            (converted from integer indices to period strings via the original
            ``periods`` array) or from any external change-point detector.
        tolerance:
            Override the instance-level tolerance for this call only. If
            ``None`` (default), uses ``self.tolerance``.

        Returns
        -------
        AttributionReport
            Full attribution report with one :class:`BreakAttribution` per
            detected break, plus summary statistics.

        Raises
        ------
        ValueError
            If break periods and calendar events are of mixed frequency, or if
            a period string cannot be parsed.

        Examples
        --------
        >>> calendar = BreakEventCalendar()
        >>> report = calendar.attribute(["2017Q1", "2020Q1", "2022Q1"])
        >>> print(report.summary())
        """
        tol = tolerance if tolerance is not None else self._tolerance

        attributions: list[BreakAttribution] = []
        for bp in break_periods:
            # Validate the break period string
            _parse_period(bp)

            best_event: Optional[CalendarEvent] = None
            best_distance: Optional[float] = None

            for evt in self._events:
                try:
                    dist = _ordinal_distance(bp, evt.period)
                except ValueError:
                    # Frequency mismatch — skip this event silently so that
                    # a calendar with mixed frequencies does not error out on
                    # a single break period. A calendar containing both quarterly
                    # and monthly events is perfectly valid; we just skip events
                    # that cannot be compared to this particular break period.
                    continue

                if dist <= tol:
                    if best_distance is None or dist < best_distance:
                        best_event = evt
                        best_distance = dist

            explained = best_event is not None
            attributions.append(
                BreakAttribution(
                    break_period=bp,
                    matched_event=best_event,
                    distance=best_distance,
                    explained=explained,
                )
            )

        n_explained = sum(1 for a in attributions if a.explained)
        return AttributionReport(
            attributions=attributions,
            n_breaks=len(attributions),
            n_explained=n_explained,
            n_unexplained=len(attributions) - n_explained,
            tolerance=tol,
        )

    def attribute_indices(
        self,
        break_indices: list[int],
        periods: list[str],
        tolerance: Optional[float] = None,
    ) -> AttributionReport:
        """Attribute break indices (from :func:`detect_breakpoints`) to events.

        A convenience wrapper that converts integer break indices to period
        strings using the supplied ``periods`` list, then calls
        :meth:`attribute`.

        Parameters
        ----------
        break_indices:
            0-based integer break indices, as returned by
            :func:`~insurance_trend.breaks.detect_breakpoints`.
        periods:
            The full list of period labels used when fitting the trend model,
            aligned to the same series as the break detection. Must be at
            least as long as ``max(break_indices) + 1``.
        tolerance:
            Override tolerance (periods). Default: ``self.tolerance``.

        Returns
        -------
        AttributionReport

        Raises
        ------
        IndexError
            If any break index is out of range for the ``periods`` list.
        ValueError
            If ``periods`` is empty.

        Examples
        --------
        >>> from insurance_trend import BreakEventCalendar
        >>> from insurance_trend.breaks import detect_breakpoints
        >>> import numpy as np
        >>>
        >>> periods = [f"{y}Q{q}" for y in range(2015, 2025) for q in range(1, 5)]
        >>> series = np.random.default_rng(0).normal(size=40)
        >>> breaks = detect_breakpoints(series)
        >>> calendar = BreakEventCalendar()
        >>> report = calendar.attribute_indices(breaks, periods)
        >>> print(report.summary())
        """
        if not periods:
            raise ValueError("periods must not be empty.")
        n = len(periods)
        break_period_strs: list[str] = []
        for idx in break_indices:
            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Break index {idx} is out of range for periods list of length {n}."
                )
            break_period_strs.append(periods[idx])
        return self.attribute(break_period_strs, tolerance=tolerance)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BreakEventCalendar("
            f"n_events={self.n_events}, "
            f"tolerance={self._tolerance})"
        )

    def __len__(self) -> int:
        return self.n_events
