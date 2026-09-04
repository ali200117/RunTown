"""Tests for calibration. No camera, no model - just scripted poses."""

import pytest

from kinetirun.calibration import Baseline, CalibrationState, Calibrator
from kinetirun.tracking import BodyPose, Point3D
from kinetirun.vision.landmarks import Landmark

from test_body_pose import build_pose


def shifted(pose: BodyPose, dx: float = 0.0, dy: float = 0.0, timestamp: float = 0.0) -> BodyPose:
    """Move a whole pose across the frame - what a side step or jump looks like."""
    moved = tuple(
        Point3D(p.x + dx, p.y + dy, p.z, p.visibility) for p in pose.image
    )
    return BodyPose(timestamp=timestamp, world=pose.world, image=moved)


def feed(calibrator: Calibrator, count: int = 40, fps: float = 30.0, **pose_kwargs):
    """Feed `count` identical, still poses at a steady frame rate."""
    for i in range(count):
        pose = build_pose(timestamp=i / fps, **pose_kwargs)
        calibrator.update(pose)
    return calibrator


# ---- state machine -------------------------------------------------------


def test_starts_waiting():
    assert Calibrator().state is CalibrationState.WAITING


def test_completes_after_standing_still():
    calibrator = feed(Calibrator(duration=1.0, min_samples=10))

    assert calibrator.state is CalibrationState.COMPLETE
    assert calibrator.baseline is not None
    assert calibrator.progress == 1.0


def test_no_pose_keeps_it_waiting():
    calibrator = Calibrator()
    for _ in range(50):
        calibrator.update(None)

    assert calibrator.state is CalibrationState.WAITING
    assert calibrator.baseline is None


def test_unreliable_poses_are_ignored():
    """MediaPipe always returns coordinates, including for joints it cannot
    see. Calibrating on those would bake fiction into every threshold."""
    calibrator = Calibrator(duration=1.0, min_samples=10)
    feed(calibrator, count=40, visibility=0.1)

    assert calibrator.state is CalibrationState.WAITING


def test_too_few_samples_does_not_complete():
    """A low frame rate must not produce a baseline built from four frames."""
    calibrator = Calibrator(duration=1.0, min_samples=30)
    feed(calibrator, count=10, fps=5.0)  # 2 seconds of time, but only 10 samples

    assert calibrator.state is not CalibrationState.COMPLETE


def test_moving_resets_progress():
    calibrator = Calibrator(duration=1.0, min_samples=10)
    base = build_pose()

    for i in range(40):
        calibrator.update(shifted(base, timestamp=i / 30.0))
    assert calibrator.state is CalibrationState.COMPLETE

    calibrator.reset()
    # Drift far sideways part-way through: progress must not survive it.
    # 20 samples remain after the reset - under the 1.0 second requirement.
    for i in range(40):
        drift = 0.0 if i < 20 else 0.5
        calibrator.update(shifted(base, dx=drift, timestamp=i / 30.0))

    assert calibrator.state is not CalibrationState.COMPLETE


def test_reset_clears_the_baseline():
    calibrator = feed(Calibrator(duration=1.0, min_samples=10))
    calibrator.reset()

    assert calibrator.state is CalibrationState.WAITING
    assert calibrator.baseline is None


def test_outliers_do_not_move_the_baseline():
    """One frame where MediaPipe loses the plot should not shift the ruler."""
    calibrator = Calibrator(duration=1.0, min_samples=10)

    for i in range(40):
        # A single absurd shoulder width in the middle of the run.
        half_width = 5.0 if i == 20 else 0.20
        calibrator.update(build_pose(shoulder_half_width=half_width, timestamp=i / 30.0))

    assert calibrator.baseline is not None
    assert calibrator.baseline.shoulder_width == pytest.approx(0.40)


# ---- the ratios detectors will actually use ------------------------------


def test_hip_drop_is_zero_when_standing():
    calibrator = feed(Calibrator(duration=1.0, min_samples=10))

    assert calibrator.baseline.hip_drop(build_pose()) == pytest.approx(0.0)


def test_hip_drop_grows_as_you_descend():
    baseline = feed(Calibrator(duration=1.0, min_samples=10)).baseline

    shallow = baseline.hip_drop(build_pose(hip_y=0.10))
    deep = baseline.hip_drop(build_pose(hip_y=0.30))

    assert 0.0 < shallow < deep


def test_lateral_offset_is_measured_in_shoulder_widths():
    baseline = feed(Calibrator(duration=1.0, min_samples=10)).baseline

    # Shoulder width in image units is 0.40 here, so half a shoulder width
    # sideways is 0.20.
    moved = shifted(build_pose(), dx=0.20)

    assert baseline.lateral_offset(moved) == pytest.approx(0.5)


def test_vertical_offset_is_negative_when_rising():
    """Image y grows downward, so a jump produces a negative offset."""
    baseline = feed(Calibrator(duration=1.0, min_samples=10)).baseline

    jumped = shifted(build_pose(), dy=-0.20)

    assert baseline.vertical_offset(jumped) == pytest.approx(-0.5)


# ---- persistence ---------------------------------------------------------


def test_baseline_survives_a_round_trip(tmp_path):
    baseline = feed(Calibrator(duration=1.0, min_samples=10)).baseline
    path = tmp_path / "calibration.json"

    baseline.save(path)

    assert Baseline.load(path) == baseline
