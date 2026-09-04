"""Tests for RollingAverage."""

import pytest

from kinetirun.metrics import RollingAverage


def test_value_is_none_before_any_input():
    assert RollingAverage().value is None


def test_averages_values():
    average = RollingAverage()
    for value in (10.0, 20.0, 30.0):
        average.add(value)

    assert average.value == pytest.approx(20.0)


def test_window_evicts_oldest():
    average = RollingAverage(window=2)
    average.add(100.0)
    average.add(10.0)
    average.add(20.0)

    # The 100 is gone, so only 10 and 20 count.
    assert average.value == pytest.approx(15.0)


def test_reset_clears():
    average = RollingAverage()
    average.add(5.0)
    average.reset()

    assert average.value is None


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError):
        RollingAverage(window=0)
