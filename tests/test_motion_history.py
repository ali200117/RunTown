"""Tests for MotionHistory.

These are the questions a squat detector will ask in Phase 7, asked directly.
"""

import pytest

from kinetirun.tracking import BodyPose, MotionHistory

from test_body_pose import build_pose


def hip_y(pose: BodyPose) -> float:
    """Hips in world space. Larger means lower, since y grows downward."""
    return pose.hip_center.y


def fill(history: MotionHistory, values, fps: float = 30.0) -> MotionHistory:
    """Feed a scripted sequence of hip heights at a steady frame rate."""
    for i, value in enumerate(values):
        history.append(build_pose(hip_y=value, timestamp=i / fps))
    return history


# ---- the window ----------------------------------------------------------


def test_starts_empty():
    history = MotionHistory()

    assert len(history) == 0
    assert history.latest is None
    assert history.span == 0.0


def test_old_poses_are_dropped():
    history = fill(MotionHistory(duration=1.0), [0.0] * 90)  # 3 seconds at 30 FPS

    assert history.span <= 1.0
    assert len(history) < 90


def test_losing_the_pose_clears_everything():
    """Velocity must never be computed across a gap where the user was absent."""
    history = fill(MotionHistory(), [0.0] * 30)
    history.append(None)

    assert len(history) == 0


def test_invalid_duration_is_rejected():
    with pytest.raises(ValueError):
        MotionHistory(duration=0)


# ---- looking backwards in time -------------------------------------------


def test_value_at_interpolates_between_frames():
    """The answer must not depend on whether a frame landed exactly then."""
    # 0.0, 0.1, 0.2 ... at 10 FPS, so time and value advance together.
    history = fill(MotionHistory(), [i * 0.1 for i in range(11)], fps=10.0)

    # 0.25 s ago, half way between two samples.
    assert history.value_at(0.25, hip_y) == pytest.approx(0.75)


def test_value_at_returns_none_beyond_the_window():
    history = fill(MotionHistory(duration=2.0), [0.0] * 10)

    assert history.value_at(5.0, hip_y) is None


def test_displacement_measures_change_over_time():
    history = fill(MotionHistory(), [i * 0.01 for i in range(31)])

    # 0.5 s at 30 FPS is 15 frames, each worth 0.01.
    assert history.displacement(0.5, hip_y) == pytest.approx(0.15, abs=0.01)


def test_velocity_is_change_per_second():
    """Hips descending at a steady 0.3 m/s."""
    history = fill(MotionHistory(), [i * 0.01 for i in range(31)])

    assert history.velocity(hip_y) == pytest.approx(0.3, abs=0.05)


def test_velocity_is_zero_when_still():
    history = fill(MotionHistory(), [0.5] * 30)

    assert history.velocity(hip_y) == pytest.approx(0.0)


def test_velocity_sign_shows_direction():
    descending = fill(MotionHistory(), [i * 0.01 for i in range(31)])
    ascending = fill(MotionHistory(), [0.3 - i * 0.01 for i in range(31)])

    assert descending.velocity(hip_y) > 0  # y grows downward
    assert ascending.velocity(hip_y) < 0


# ---- finding the turning point -------------------------------------------


def test_extreme_finds_the_bottom_of_a_squat_and_when_it_happened():
    """The timestamp is the useful half: it separates a completed movement
    from one still in progress."""
    down = [i * 0.02 for i in range(16)]      # descend to 0.30
    up = [0.30 - i * 0.02 for i in range(1, 16)]
    history = fill(MotionHistory(duration=5.0), down + up)

    value, timestamp = history.extreme(hip_y, largest=True)

    assert value == pytest.approx(0.30)
    assert timestamp == pytest.approx(15 / 30.0)


def test_extreme_can_be_limited_to_a_recent_window():
    history = fill(MotionHistory(duration=5.0), [0.9] + [0.1] * 30)

    value, _ = history.extreme(hip_y, seconds=0.5, largest=True)

    assert value == pytest.approx(0.1)


def test_time_since_measures_age():
    history = fill(MotionHistory(duration=5.0), [0.0] * 31)

    assert history.time_since(0.0) == pytest.approx(1.0, abs=0.05)
