"""Tests for PoseSmoother."""

import pytest

from kinetirun.tracking import BodyPose, Point3D, PoseSmoother
from kinetirun.vision.landmarks import Landmark

from test_body_pose import build_pose


def hip_x(pose: BodyPose) -> float:
    return pose.image[Landmark.LEFT_HIP].x


def shift(pose: BodyPose, dx: float, timestamp: float) -> BodyPose:
    moved = tuple(Point3D(p.x + dx, p.y, p.z, p.visibility) for p in pose.image)
    return BodyPose(timestamp=timestamp, world=pose.world, image=moved)


def test_first_pose_passes_through():
    """Nothing to blend with yet, so it must not be dragged toward zero."""
    pose = build_pose()

    assert PoseSmoother().smooth(pose) is pose


def test_smoothing_lags_behind_a_step_change():
    smoother = PoseSmoother(alpha=0.5)
    start = build_pose()
    smoother.smooth(start)

    jumped = shift(start, dx=1.0, timestamp=0.1)
    result = smoother.smooth(jumped)

    # Half way there after one frame, by definition of alpha=0.5.
    assert hip_x(result) == pytest.approx(hip_x(start) + 0.5)


def test_repeated_frames_converge_on_the_true_value():
    smoother = PoseSmoother(alpha=0.5)
    start = build_pose()
    smoother.smooth(start)

    target = shift(start, dx=1.0, timestamp=0.1)
    for _ in range(10):
        result = smoother.smooth(target)

    assert hip_x(result) == pytest.approx(hip_x(target), abs=0.01)


def test_alpha_of_one_is_a_passthrough():
    smoother = PoseSmoother(alpha=1.0)
    start = build_pose()
    smoother.smooth(start)

    jumped = shift(start, dx=1.0, timestamp=0.1)

    assert hip_x(smoother.smooth(jumped)) == pytest.approx(hip_x(jumped))


def test_losing_the_pose_clears_the_history():
    """Blending across an absence would drag the skeleton and fake a velocity."""
    smoother = PoseSmoother(alpha=0.5)
    smoother.smooth(build_pose())
    assert smoother.smooth(None) is None

    returning = shift(build_pose(), dx=5.0, timestamp=1.0)

    assert smoother.smooth(returning) is returning


def test_timestamp_is_never_smoothed():
    smoother = PoseSmoother(alpha=0.2)
    smoother.smooth(build_pose(timestamp=0.0))

    result = smoother.smooth(build_pose(timestamp=7.5))

    assert result.timestamp == 7.5


def test_visibility_is_not_smoothed():
    """Confidence is not a position: averaging would let good frames prop up a
    joint the model has actually lost."""
    smoother = PoseSmoother(alpha=0.5)
    smoother.smooth(build_pose(visibility=1.0))

    result = smoother.smooth(build_pose(visibility=0.0, timestamp=0.1))

    assert result.image[Landmark.LEFT_HIP].visibility == 0.0


@pytest.mark.parametrize("alpha", [0.0, -0.5, 1.5])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        PoseSmoother(alpha=alpha)
