"""Tests for BreakEventCalendar and supporting classes.

Coverage:
- Period parsing: quarterly, monthly, annual, edge cases, error paths
- _ordinal_distance: correct arithmetic, frequency mismatch error
- CalendarEvent: construction, frozen dataclass, invalid impact
- BreakEventCalendar construction: defaults loaded, empty, invalid tolerance
- add_event / remove_event mutations and chaining
- filter_events: by category, impact, from/to period, combined
- events_dataframe: columns and row count
- attribute: exact match, within tolerance, outside tolerance, empty breaks,
  multiple breaks, multiple events in window (nearest wins), unexplained breaks
- attribute_indices: correct index→period mapping, out-of-range error,
  empty periods error
- AttributionReport: summary, to_dataframe, repr
- BreakAttribution: explained flag, None matched_event
- Default calendar: event count, known events present
- Tolerance property setter validation
- Mixed frequency handling (skipped gracefully)
- End-to-end: detect_breakpoints integration (integer indices → period strings)
"""

from __future__ import annotations

import pytest

from insurance_trend.calendar import (
    AttributionReport,
    BreakAttribution,
    BreakEventCalendar,
    CalendarEvent,
    _DEFAULT_UK_EVENTS,
    _ordinal_distance,
    _parse_period,
    _period_to_ordinal,
)


# ---------------------------------------------------------------------------
# _parse_period tests
# ---------------------------------------------------------------------------


class TestParsePeriod:
    def test_quarterly_formats(self):
        assert _parse_period("2020Q1") == (2020, 1, 4)
        assert _parse_period("2020Q4") == (2020, 4, 4)
        assert _parse_period("1999q2") == (1999, 2, 4)

    def test_monthly_hyphen(self):
        assert _parse_period("2020-01") == (2020, 1, 12)
        assert _parse_period("2020-12") == (2020, 12, 12)

    def test_monthly_M_format(self):
        assert _parse_period("2020M03") == (2020, 3, 12)
        assert _parse_period("2020m11") == (2020, 11, 12)

    def test_annual(self):
        assert _parse_period("2021") == (2021, 1, 1)
        assert _parse_period("1990") == (1990, 1, 1)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse period"):
            _parse_period("2020-Q1")

    def test_month_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Month sub-period out of range"):
            _parse_period("2020M13")

    def test_zero_month_raises(self):
        with pytest.raises(ValueError, match="Month sub-period out of range"):
            _parse_period("2020M00")

    def test_whitespace_stripped(self):
        assert _parse_period("  2020Q1  ") == (2020, 1, 4)


# ---------------------------------------------------------------------------
# _period_to_ordinal tests
# ---------------------------------------------------------------------------


class TestPeriodToOrdinal:
    def test_quarterly_ordinals(self):
        o1 = _period_to_ordinal(2020, 1, 4)
        o2 = _period_to_ordinal(2020, 2, 4)
        assert o2 - o1 == pytest.approx(1.0)

    def test_annual_step(self):
        o1 = _period_to_ordinal(2020, 1, 4)
        o2 = _period_to_ordinal(2021, 1, 4)
        assert o2 - o1 == pytest.approx(4.0)

    def test_monthly_step(self):
        o1 = _period_to_ordinal(2020, 1, 12)
        o2 = _period_to_ordinal(2020, 3, 12)
        assert o2 - o1 == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _ordinal_distance tests
# ---------------------------------------------------------------------------


class TestOrdinalDistance:
    def test_exact_match(self):
        assert _ordinal_distance("2020Q1", "2020Q1") == pytest.approx(0.0)

    def test_one_quarter_apart(self):
        assert _ordinal_distance("2020Q1", "2020Q2") == pytest.approx(1.0)

    def test_cross_year(self):
        assert _ordinal_distance("2019Q4", "2020Q1") == pytest.approx(1.0)

    def test_symmetric(self):
        assert _ordinal_distance("2020Q3", "2017Q1") == _ordinal_distance("2017Q1", "2020Q3")

    def test_three_years_quarterly(self):
        assert _ordinal_distance("2020Q1", "2017Q1") == pytest.approx(12.0)

    def test_monthly_distance(self):
        assert _ordinal_distance("2020M01", "2020M07") == pytest.approx(6.0)

    def test_annual_distance(self):
        assert _ordinal_distance("2020", "2023") == pytest.approx(3.0)

    def test_frequency_mismatch_raises(self):
        with pytest.raises(ValueError, match="different frequencies"):
            _ordinal_distance("2020Q1", "2020M01")


# ---------------------------------------------------------------------------
# CalendarEvent tests
# ---------------------------------------------------------------------------


class TestCalendarEvent:
    def test_valid_construction(self):
        evt = CalendarEvent(
            period="2020Q1",
            description="Test event",
            category="covid",
            impact=-1,
            source="Test source",
        )
        assert evt.period == "2020Q1"
        assert evt.category == "covid"
        assert evt.impact == -1

    def test_zero_impact_allowed(self):
        evt = CalendarEvent(period="2020Q1", description="Ambiguous", category="other", impact=0)
        assert evt.impact == 0

    def test_invalid_impact_raises(self):
        with pytest.raises(ValueError, match="impact must be -1, 0, or \\+1"):
            CalendarEvent(period="2020Q1", description="Bad", category="other", impact=2)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="Cannot parse period"):
            CalendarEvent(period="not-a-period", description="Bad", category="other", impact=0)

    def test_frozen_dataclass(self):
        evt = CalendarEvent(period="2020Q1", description="Test", category="other", impact=0)
        with pytest.raises(Exception):
            evt.period = "2021Q1"  # type: ignore[misc]

    def test_source_optional(self):
        evt = CalendarEvent(period="2020Q1", description="Test", category="other", impact=0)
        assert evt.source == ""


# ---------------------------------------------------------------------------
# BreakEventCalendar construction tests
# ---------------------------------------------------------------------------


class TestBreakEventCalendarConstruction:
    def test_defaults_loaded(self):
        cal = BreakEventCalendar()
        assert cal.n_events >= 20
        assert cal.tolerance == 2.0

    def test_empty_calendar(self):
        cal = BreakEventCalendar(include_defaults=False)
        assert cal.n_events == 0

    def test_custom_tolerance(self):
        cal = BreakEventCalendar(tolerance=1.0)
        assert cal.tolerance == 1.0

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            BreakEventCalendar(tolerance=-1.0)

    def test_zero_tolerance_allowed(self):
        cal = BreakEventCalendar(tolerance=0)
        assert cal.tolerance == 0.0

    def test_repr(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=2.0)
        r = repr(cal)
        assert "BreakEventCalendar" in r
        assert "n_events=0" in r

    def test_len(self):
        cal = BreakEventCalendar(include_defaults=False)
        assert len(cal) == 0
        cal.add_event("2020Q1", "Test", "other", 0)
        assert len(cal) == 1


# ---------------------------------------------------------------------------
# add_event / remove_event tests
# ---------------------------------------------------------------------------


class TestMutation:
    def test_add_event_increases_count(self):
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2020Q1", "Test", "covid", -1)
        assert cal.n_events == 1

    def test_add_event_returns_self_for_chaining(self):
        cal = BreakEventCalendar(include_defaults=False)
        result = cal.add_event("2020Q1", "A", "other", 0).add_event("2021Q1", "B", "other", 1)
        assert result is cal
        assert cal.n_events == 2

    def test_add_event_invalid_impact(self):
        cal = BreakEventCalendar(include_defaults=False)
        with pytest.raises(ValueError):
            cal.add_event("2020Q1", "Bad", "other", 5)

    def test_add_event_invalid_period(self):
        cal = BreakEventCalendar(include_defaults=False)
        with pytest.raises(ValueError):
            cal.add_event("not-valid", "Bad", "other", 0)

    def test_remove_event_exact_period(self):
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2020Q1", "Event A", "covid", -1)
        cal.add_event("2021Q1", "Event B", "macro", 1)
        removed = cal.remove_event("2020Q1")
        assert removed == 1
        assert cal.n_events == 1
        assert cal.events[0].period == "2021Q1"

    def test_remove_event_with_description_filter(self):
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2020Q1", "Ogden rate change", "legal", 1)
        cal.add_event("2020Q1", "IPT rise", "tax", 1)
        removed = cal.remove_event("2020Q1", description_contains="Ogden")
        assert removed == 1
        assert cal.n_events == 1
        assert "IPT" in cal.events[0].description

    def test_remove_nonexistent_returns_zero(self):
        cal = BreakEventCalendar(include_defaults=False)
        assert cal.remove_event("2099Q1") == 0

    def test_events_returns_copy(self):
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2020Q1", "Test", "other", 0)
        events_copy = cal.events
        events_copy.append(CalendarEvent("2021Q1", "Extra", "other", 0))
        assert cal.n_events == 1  # original unaffected


# ---------------------------------------------------------------------------
# filter_events tests
# ---------------------------------------------------------------------------


class TestFilterEvents:
    def setup_method(self):
        self.cal = BreakEventCalendar(include_defaults=False)
        self.cal.add_event("2017Q1", "Ogden -0.75%", "legal", 1)
        self.cal.add_event("2019Q3", "Ogden -0.25%", "legal", -1)
        self.cal.add_event("2020Q1", "COVID lockdown", "covid", -1)
        self.cal.add_event("2022Q1", "GIPP rules", "regulation", 0)
        self.cal.add_event("2015Q4", "IPT to 9.5%", "tax", 1)

    def test_filter_by_category(self):
        filtered = self.cal.filter_events(categories=["legal"])
        assert filtered.n_events == 2
        assert all(e.category == "legal" for e in filtered.events)

    def test_filter_by_multiple_categories(self):
        filtered = self.cal.filter_events(categories=["legal", "covid"])
        assert filtered.n_events == 3

    def test_filter_by_impact_positive(self):
        filtered = self.cal.filter_events(impact=1)
        assert filtered.n_events == 2
        assert all(e.impact == 1 for e in filtered.events)

    def test_filter_by_impact_negative(self):
        filtered = self.cal.filter_events(impact=-1)
        assert filtered.n_events == 2

    def test_filter_by_impact_zero(self):
        filtered = self.cal.filter_events(impact=0)
        assert filtered.n_events == 1

    def test_filter_from_period(self):
        filtered = self.cal.filter_events(from_period="2019Q1")
        periods = [e.period for e in filtered.events]
        assert "2015Q4" not in periods
        assert "2017Q1" not in periods
        assert "2019Q3" in periods
        assert "2020Q1" in periods

    def test_filter_to_period(self):
        filtered = self.cal.filter_events(to_period="2019Q4")
        periods = [e.period for e in filtered.events]
        assert "2020Q1" not in periods
        assert "2022Q1" not in periods
        assert "2017Q1" in periods

    def test_combined_filters(self):
        filtered = self.cal.filter_events(
            categories=["legal"],
            from_period="2018Q1",
        )
        assert filtered.n_events == 1
        assert filtered.events[0].period == "2019Q3"

    def test_filter_returns_new_instance(self):
        filtered = self.cal.filter_events(categories=["legal"])
        assert filtered is not self.cal

    def test_filter_inherits_tolerance(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=3.0)
        cal.add_event("2020Q1", "Test", "other", 0)
        filtered = cal.filter_events()
        assert filtered.tolerance == 3.0

    def test_filter_empty_result(self):
        filtered = self.cal.filter_events(categories=["supply_chain"])
        assert filtered.n_events == 0


# ---------------------------------------------------------------------------
# events_dataframe tests
# ---------------------------------------------------------------------------


class TestEventsDataframe:
    def test_columns_present(self):
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2020Q1", "Test event", "covid", -1, "Some source")
        df = cal.events_dataframe()
        assert set(df.columns) == {"period", "description", "category", "impact", "source"}

    def test_row_count_matches(self):
        cal = BreakEventCalendar()
        df = cal.events_dataframe()
        assert len(df) == cal.n_events

    def test_empty_calendar_empty_df(self):
        cal = BreakEventCalendar(include_defaults=False)
        df = cal.events_dataframe()
        assert len(df) == 0

    def test_values_correct(self):
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2017Q1", "Ogden rate change", "legal", 1, "Damages Act")
        df = cal.events_dataframe()
        assert df["period"][0] == "2017Q1"
        assert df["description"][0] == "Ogden rate change"
        assert df["impact"][0] == 1


# ---------------------------------------------------------------------------
# attribute tests
# ---------------------------------------------------------------------------


class TestAttribute:
    def test_exact_match(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=2)
        cal.add_event("2020Q1", "COVID lockdown", "covid", -1)
        report = cal.attribute(["2020Q1"])
        assert report.n_breaks == 1
        assert report.n_explained == 1
        assert report.attributions[0].explained is True
        assert report.attributions[0].distance == pytest.approx(0.0)

    def test_within_tolerance(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=2)
        cal.add_event("2020Q1", "COVID lockdown", "covid", -1)
        # Break detected one quarter late — still within 2 periods
        report = cal.attribute(["2020Q2"])
        assert report.n_explained == 1
        assert report.attributions[0].distance == pytest.approx(1.0)

    def test_outside_tolerance(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=2)
        cal.add_event("2020Q1", "COVID lockdown", "covid", -1)
        # 4 quarters away — outside tolerance of 2
        report = cal.attribute(["2021Q1"])
        assert report.n_explained == 0
        assert report.attributions[0].explained is False
        assert report.attributions[0].matched_event is None

    def test_nearest_event_wins(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=4)
        cal.add_event("2020Q1", "Event A", "covid", -1)
        cal.add_event("2020Q3", "Event B", "macro", 1)
        # Break at 2020Q2: distance to A = 1, distance to B = 1 — ties broken by registration order
        # Break at 2020Q4: distance to A = 3, distance to B = 1 — B wins
        report = cal.attribute(["2020Q4"])
        assert report.attributions[0].matched_event.description == "Event B"

    def test_strictly_nearer_event_wins(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=5)
        cal.add_event("2017Q1", "Ogden", "legal", 1)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        # Break at 2020Q2 — closer to COVID (1) than Ogden (13)
        report = cal.attribute(["2020Q2"])
        assert report.attributions[0].matched_event.description == "COVID"

    def test_empty_breaks_list(self):
        cal = BreakEventCalendar()
        report = cal.attribute([])
        assert report.n_breaks == 0
        assert report.n_explained == 0
        assert report.n_unexplained == 0

    def test_multiple_breaks(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=1)
        cal.add_event("2017Q1", "Ogden", "legal", 1)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        report = cal.attribute(["2017Q1", "2020Q1", "2015Q1"])
        assert report.n_breaks == 3
        assert report.n_explained == 2
        assert report.n_unexplained == 1

    def test_tolerance_override_at_call_site(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=0)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        # With instance tolerance=0: 2020Q2 is outside
        r0 = cal.attribute(["2020Q2"])
        assert r0.n_explained == 0
        # With override tolerance=2: 2020Q2 is within
        r2 = cal.attribute(["2020Q2"], tolerance=2)
        assert r2.n_explained == 1

    def test_invalid_break_period_raises(self):
        cal = BreakEventCalendar(include_defaults=False)
        with pytest.raises(ValueError, match="Cannot parse period"):
            cal.attribute(["not-a-period"])

    def test_report_tolerance_matches_used(self):
        cal = BreakEventCalendar(tolerance=3.0)
        report = cal.attribute([])
        assert report.tolerance == 3.0

    def test_report_tolerance_override(self):
        cal = BreakEventCalendar(tolerance=3.0)
        report = cal.attribute([], tolerance=1.0)
        assert report.tolerance == 1.0

    def test_no_events_in_calendar_all_unexplained(self):
        cal = BreakEventCalendar(include_defaults=False)
        report = cal.attribute(["2020Q1", "2017Q1"])
        assert report.n_explained == 0
        assert report.n_unexplained == 2

    def test_zero_tolerance_exact_only(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=0)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        r_exact = cal.attribute(["2020Q1"])
        r_near = cal.attribute(["2020Q2"])
        assert r_exact.n_explained == 1
        assert r_near.n_explained == 0


# ---------------------------------------------------------------------------
# attribute_indices tests
# ---------------------------------------------------------------------------


class TestAttributeIndices:
    def _make_periods(self, n: int = 40, start_year: int = 2015) -> list[str]:
        labels = []
        for year in range(start_year, start_year + 20):
            for q in range(1, 5):
                labels.append(f"{year}Q{q}")
                if len(labels) == n:
                    return labels
        return labels

    def test_basic_mapping(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=1)
        cal.add_event("2017Q1", "Ogden", "legal", 1)
        periods = self._make_periods(40)
        # "2017Q1" is index 8 (2015Q1=0, 2015Q2=1, ..., 2016Q4=7, 2017Q1=8)
        report = cal.attribute_indices([8], periods)
        assert report.attributions[0].break_period == "2017Q1"
        assert report.n_explained == 1

    def test_out_of_range_raises(self):
        cal = BreakEventCalendar(include_defaults=False)
        periods = self._make_periods(20)
        with pytest.raises(IndexError):
            cal.attribute_indices([99], periods)

    def test_negative_index_raises(self):
        cal = BreakEventCalendar(include_defaults=False)
        periods = self._make_periods(20)
        with pytest.raises(IndexError):
            cal.attribute_indices([-1], periods)

    def test_empty_periods_raises(self):
        cal = BreakEventCalendar(include_defaults=False)
        with pytest.raises(ValueError, match="periods must not be empty"):
            cal.attribute_indices([0], [])

    def test_empty_breaks_returns_empty_report(self):
        cal = BreakEventCalendar()
        periods = self._make_periods(20)
        report = cal.attribute_indices([], periods)
        assert report.n_breaks == 0

    def test_multiple_indices(self):
        cal = BreakEventCalendar(include_defaults=False, tolerance=1)
        cal.add_event("2017Q1", "Ogden", "legal", 1)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        periods = self._make_periods(40)
        idx_ogden = periods.index("2017Q1")
        idx_covid = periods.index("2020Q1")
        report = cal.attribute_indices([idx_ogden, idx_covid], periods)
        assert report.n_explained == 2


# ---------------------------------------------------------------------------
# AttributionReport tests
# ---------------------------------------------------------------------------


class TestAttributionReport:
    def _make_report(self) -> AttributionReport:
        cal = BreakEventCalendar(include_defaults=False, tolerance=2)
        cal.add_event("2017Q1", "Ogden -0.75%", "legal", 1, "Lord Chancellor 2017")
        cal.add_event("2020Q1", "COVID lockdown", "covid", -1)
        return cal.attribute(["2017Q1", "2022Q3"])

    def test_summary_contains_break_periods(self):
        report = self._make_report()
        s = report.summary()
        assert "2017Q1" in s
        assert "2022Q3" in s

    def test_summary_contains_explained_unexplained(self):
        report = self._make_report()
        s = report.summary()
        assert "EXPLAINED" in s
        assert "UNEXPLAINED" in s

    def test_summary_contains_event_description(self):
        report = self._make_report()
        s = report.summary()
        assert "Ogden" in s

    def test_summary_contains_source(self):
        report = self._make_report()
        s = report.summary()
        assert "Lord Chancellor" in s

    def test_to_dataframe_columns(self):
        report = self._make_report()
        df = report.to_dataframe()
        expected = {
            "break_period", "explained", "matched_event_period",
            "matched_event_description", "matched_event_category",
            "matched_event_impact", "distance_periods",
        }
        assert set(df.columns) == expected

    def test_to_dataframe_row_count(self):
        report = self._make_report()
        df = report.to_dataframe()
        assert len(df) == 2

    def test_to_dataframe_nulls_for_unexplained(self):
        report = self._make_report()
        df = report.to_dataframe()
        # Second row (2022Q3) is unexplained — event columns should be null
        unexplained_row = df.filter(df["break_period"] == "2022Q3")
        assert unexplained_row["matched_event_period"][0] is None

    def test_repr(self):
        report = self._make_report()
        r = repr(report)
        assert "AttributionReport" in r
        assert "n_breaks=2" in r

    def test_counts_correct(self):
        report = self._make_report()
        assert report.n_breaks == 2
        assert report.n_explained == 1
        assert report.n_unexplained == 1


# ---------------------------------------------------------------------------
# Default calendar content tests
# ---------------------------------------------------------------------------


class TestDefaultCalendar:
    def test_minimum_event_count(self):
        """Must have at least 20 events to cover the required UK insurance history."""
        cal = BreakEventCalendar()
        assert cal.n_events >= 20

    def test_ogden_2017_present(self):
        cal = BreakEventCalendar()
        ogden_events = [e for e in cal.events if "2017Q1" == e.period and "Ogden" in e.description]
        assert len(ogden_events) >= 1, "Ogden -0.75% (2017Q1) event should be in default calendar"

    def test_ogden_2019_present(self):
        cal = BreakEventCalendar()
        ogden_events = [e for e in cal.events if "2019Q3" == e.period and "Ogden" in e.description]
        assert len(ogden_events) >= 1, "Ogden -0.25% (2019Q3) event should be in default calendar"

    def test_covid_lockdown_present(self):
        cal = BreakEventCalendar()
        covid_events = [e for e in cal.events if e.category == "covid"]
        assert len(covid_events) >= 2, "At least two COVID-related events expected"

    def test_ipt_rise_2015_present(self):
        cal = BreakEventCalendar()
        ipt_events = [e for e in cal.events if e.category == "tax" and "2015" in e.period]
        assert len(ipt_events) >= 1, "IPT 2015 rise should be in default calendar"

    def test_gipp_2022_present(self):
        cal = BreakEventCalendar()
        gipp_events = [e for e in cal.events if "GIPP" in e.description or "Pricing Practices" in e.description]
        assert len(gipp_events) >= 1, "GIPP 2022 event should be in default calendar"

    def test_whiplash_reform_2021_present(self):
        cal = BreakEventCalendar()
        wi_events = [e for e in cal.events if "whiplash" in e.description.lower() or "Whiplash" in e.description]
        assert len(wi_events) >= 1, "Whiplash tariff reform (2021) should be in default calendar"

    def test_all_events_have_valid_impact(self):
        cal = BreakEventCalendar()
        for evt in cal.events:
            assert evt.impact in (-1, 0, 1), f"Invalid impact for event {evt.period}: {evt.impact}"

    def test_all_periods_parseable(self):
        cal = BreakEventCalendar()
        for evt in cal.events:
            _parse_period(evt.period)  # Should not raise

    def test_all_events_are_quarterly(self):
        """Default calendar should use quarterly period strings for consistency."""
        cal = BreakEventCalendar()
        for evt in cal.events:
            y, s, ppy = _parse_period(evt.period)
            assert ppy == 4, (
                f"Default calendar event {evt.period!r} is not quarterly (ppy={ppy}). "
                "Mixed-frequency default calendar is unexpected."
            )

    def test_categories_cover_expected_types(self):
        cal = BreakEventCalendar()
        cats = {e.category for e in cal.events}
        expected_cats = {"regulation", "legal", "macro", "covid", "supply_chain", "tax"}
        assert expected_cats.issubset(cats), (
            f"Default calendar missing expected categories. Got: {cats}"
        )

    def test_known_break_attribution(self):
        """Ogden 2017 and COVID 2020 should be attributed with default tolerance."""
        cal = BreakEventCalendar()
        report = cal.attribute(["2017Q1", "2020Q1"])
        assert report.n_explained == 2

    def test_known_break_near_tolerance(self):
        """A break one period off a known event should still be attributed (tol=2)."""
        cal = BreakEventCalendar(tolerance=2)
        report = cal.attribute(["2017Q2"])  # one quarter after Ogden 2017Q1
        assert report.n_explained == 1

    def test_far_future_unexplained(self):
        """A break in 2035 should be unexplained."""
        cal = BreakEventCalendar()
        report = cal.attribute(["2035Q1"])
        assert report.n_unexplained == 1


# ---------------------------------------------------------------------------
# Tolerance property setter
# ---------------------------------------------------------------------------


class TestToleranceSetter:
    def test_set_valid_tolerance(self):
        cal = BreakEventCalendar()
        cal.tolerance = 3.0
        assert cal.tolerance == 3.0

    def test_set_negative_tolerance_raises(self):
        cal = BreakEventCalendar()
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            cal.tolerance = -0.5


# ---------------------------------------------------------------------------
# Mixed frequency handling
# ---------------------------------------------------------------------------


class TestMixedFrequency:
    def test_mixed_frequency_skipped_gracefully(self):
        """Monthly events should be silently skipped when break period is quarterly."""
        cal = BreakEventCalendar(include_defaults=False, tolerance=2)
        cal.add_event("2020M03", "Monthly event", "covid", -1)
        cal.add_event("2020Q1", "Quarterly event", "covid", -1)
        # 2020Q1 break should match the quarterly event, not crash on the monthly one
        report = cal.attribute(["2020Q1"])
        assert report.n_explained == 1
        assert report.attributions[0].matched_event.description == "Quarterly event"

    def test_all_mixed_frequency_results_in_unexplained(self):
        """If all calendar events are monthly and break is quarterly, break is unexplained."""
        cal = BreakEventCalendar(include_defaults=False, tolerance=2)
        cal.add_event("2020M03", "Monthly event", "covid", -1)
        report = cal.attribute(["2020Q1"])
        assert report.n_unexplained == 1


# ---------------------------------------------------------------------------
# End-to-end integration with detect_breakpoints
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_attribute_indices_end_to_end(self):
        """Simulate a full workflow: detect breaks → attribute to events."""
        import numpy as np
        from insurance_trend.breaks import detect_breakpoints

        # Build a synthetic series with a step change at period 8 (2017Q1)
        rng = np.random.default_rng(99)
        n = 40
        log_series = np.zeros(n)
        log_series[:8] = 0.02 * np.arange(8)
        log_series[8:] = 0.5 + 0.02 * np.arange(32)  # large step at index 8
        log_series += rng.normal(0, 0.01, n)

        periods = [f"{y}Q{q}" for y in range(2015, 2025) for q in range(1, 5)]

        breaks = detect_breakpoints(log_series, penalty=1.5)

        cal = BreakEventCalendar(tolerance=3)
        report = cal.attribute_indices(breaks, periods)

        # We should get a report back without errors
        assert isinstance(report, AttributionReport)
        assert report.n_breaks == len(breaks)
        # At least the summary should be a non-empty string
        assert len(report.summary()) > 0

    def test_full_pipeline_with_known_events(self):
        """Known break at 2017Q1 in a series should attribute to Ogden event."""
        cal = BreakEventCalendar(tolerance=2)
        periods = [f"{y}Q{q}" for y in range(2015, 2026) for q in range(1, 5)]
        # Index of 2017Q1 in that list
        idx_2017q1 = periods.index("2017Q1")

        report = cal.attribute_indices([idx_2017q1], periods)
        assert report.n_explained == 1
        assert "Ogden" in report.attributions[0].matched_event.description

    def test_summary_printable(self):
        """summary() should return a non-trivial string."""
        cal = BreakEventCalendar()
        report = cal.attribute(["2017Q1", "2020Q1", "2025Q3"])
        s = report.summary()
        assert isinstance(s, str)
        assert len(s.splitlines()) > 5
