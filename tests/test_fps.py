"""Tests for FpsCounter.

No webcam involved. This is the pattern the whole project depends on: logic is
separated from hardware, so it can be tested by feeding it fake data. Every
movement detector in Phase 7 and beyond will be tested exactly like this, with
synthetic pose sequences instead of synthetic timestamps.
"""

import pytest

from kinetirun.camera import FpsCounter


def test_fps_is_unknown_before_two_frames():
    """A rate needs an interval, and one point in time is not an interval."""
    counter = FpsCounter()

    assert counter.fps is None

    counter.tick(0.0)
    assert counter.fps is None

    counter.tick(0.5)
    assert counter.fps is not None


def test_perfectly_regular_frames_give_exact_fps():
    """10 frames spaced 0.1s apart is exactly 10 FPS."""
    counter = FpsCounter()

    for i in range(10):
        counter.tick(i * 0.1)

    # approx, because 0.1 has no exact binary representation.
    assert counter.fps == pytest.approx(10.0)


def test_window_discards_old_frames():
    """Only the last `window` timestamps should influence the result.

    This is the test that proves the deque's maxlen is doing its job: the slow
    frames are pushed out entirely, so they must not drag the average down.
    """
    counter = FpsCounter(window=3)

    # Three slow frames: 1 FPS.
    counter.tick(0.0)
    counter.tick(1.0)
    counter.tick(2.0)
    assert counter.fps == pytest.approx(1.0)

    # Three fast frames evict all of the slow ones: 100 FPS.
    counter.tick(2.01)
    counter.tick(2.02)
    counter.tick(2.03)
    assert counter.fps == pytest.approx(100.0)


def test_identical_timestamps_do_not_crash():
    """Two frames at the same instant would otherwise divide by zero."""
    counter = FpsCounter()

    counter.tick(1.0)
    counter.tick(1.0)

    assert counter.fps is None


def test_reset_clears_history():
    counter = FpsCounter()

    counter.tick(0.0)
    counter.tick(0.1)
    assert counter.fps is not None

    counter.reset()
    assert counter.fps is None


def test_invalid_window_is_rejected():
    """A window of 1 can never produce a rate, so it is a programming error."""
    with pytest.raises(ValueError):
        FpsCounter(window=1)
