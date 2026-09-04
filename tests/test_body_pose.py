"""Tests for BodyPose.

The point of Phase 4 in one file: body measurements are now verified against
poses built by hand, with no camera, no model and no lighting. Every movement
detector from Phase 7 onward gets tested exactly this way, with sequences of
these instead of single poses.
"""

import math
from types import SimpleNamespace

import pytest

from kinetirun.tracking import BodyPose, Point3D, angle_between
from kinetirun.vision.landmarks import Landmark


def build_pose(
    hip_y: float = 0.0,
    knee_y: float = 0.45,
    knee_x_offset: float = 0.0,
    ankle_y: float = 0.90,
    shoulder_y: float = -0.50,
    shoulder_half_width: float = 0.20,
    timestamp: float = 0.0,
    visibility: float = 1.0,
) -> BodyPose:
    """Build a symmetric, front-facing pose.

    Defaults describe someone standing upright with straight legs. Remember the
    axis convention: y grows DOWNWARD, so ankles (0.90) are below hips (0.0)
    and shoulders (-0.50) are above them.
    """
    points = [Point3D(0.0, 0.0, 0.0, visibility) for _ in range(33)]

    def place(landmark: Landmark, x: float, y: float) -> None:
        points[landmark] = Point3D(x, y, 0.0, visibility)

    place(Landmark.LEFT_SHOULDER, -shoulder_half_width, shoulder_y)
    place(Landmark.RIGHT_SHOULDER, shoulder_half_width, shoulder_y)
    place(Landmark.LEFT_HIP, -0.10, hip_y)
    place(Landmark.RIGHT_HIP, 0.10, hip_y)
    place(Landmark.LEFT_KNEE, -0.10 + knee_x_offset, knee_y)
    place(Landmark.RIGHT_KNEE, 0.10 + knee_x_offset, knee_y)
    place(Landmark.LEFT_ANKLE, -0.10, ankle_y)
    place(Landmark.RIGHT_ANKLE, 0.10, ankle_y)

    frozen = tuple(points)
    return BodyPose(timestamp=timestamp, world=frozen, image=frozen)


# ---- geometry primitives -------------------------------------------------


def test_angle_of_a_straight_line_is_180_degrees():
    a = Point3D(0.0, 0.0, 0.0, 1.0)
    vertex = Point3D(0.0, 1.0, 0.0, 1.0)
    b = Point3D(0.0, 2.0, 0.0, 1.0)

    assert angle_between(a, vertex, b) == pytest.approx(180.0)


def test_right_angle_is_90_degrees():
    a = Point3D(0.0, 0.0, 0.0, 1.0)
    vertex = Point3D(0.0, 1.0, 0.0, 1.0)
    b = Point3D(1.0, 1.0, 0.0, 1.0)

    assert angle_between(a, vertex, b) == pytest.approx(90.0)


def test_degenerate_angle_does_not_crash():
    """Two coincident points give a zero-length vector and no defined angle."""
    point = Point3D(1.0, 1.0, 1.0, 1.0)

    assert angle_between(point, point, point) == 0.0


def test_midpoint_takes_the_worst_visibility():
    """A midpoint built from one guessed joint is itself a guess."""
    confident = Point3D(0.0, 0.0, 0.0, 0.99)
    guessed = Point3D(2.0, 0.0, 0.0, 0.10)

    middle = confident.midpoint(guessed)

    assert middle.x == pytest.approx(1.0)
    assert middle.visibility == pytest.approx(0.10)


# ---- body measurements ---------------------------------------------------


def test_shoulder_width_is_measured_in_metres():
    pose = build_pose(shoulder_half_width=0.21)

    assert pose.shoulder_width == pytest.approx(0.42)


def test_standing_legs_are_nearly_straight():
    pose = build_pose()

    assert pose.mean_knee_angle == pytest.approx(180.0, abs=1.0)


def test_bending_the_knees_reduces_the_knee_angle():
    """The core squat signal: knees forward and hips lower means a smaller angle."""
    standing = build_pose()
    squatting = build_pose(knee_y=0.35, knee_x_offset=0.25, hip_y=0.20)

    assert squatting.mean_knee_angle < standing.mean_knee_angle
    assert squatting.mean_knee_angle < 140.0


def test_hip_height_shrinks_when_descending():
    """The second squat signal, and the reason the sign convention matters."""
    standing = build_pose(hip_y=0.0)
    lowered = build_pose(hip_y=0.25)  # y grows downward, so this is LOWER

    assert standing.hip_height_above_ankles == pytest.approx(0.90)
    assert lowered.hip_height_above_ankles < standing.hip_height_above_ankles


def test_torso_height_is_shoulder_to_hip_distance():
    pose = build_pose(shoulder_y=-0.55)

    assert pose.torso_height == pytest.approx(0.55)


# ---- confidence ----------------------------------------------------------


def test_lower_body_visibility_reports_the_weakest_joint():
    points = list(build_pose().world)
    points[Landmark.LEFT_ANKLE] = Point3D(-0.10, 0.90, 0.0, 0.12)
    pose = BodyPose(timestamp=0.0, world=tuple(points), image=tuple(points))

    assert pose.lower_body_visibility == pytest.approx(0.12)
    assert not pose.is_reliable()


def test_a_fully_visible_pose_is_reliable():
    assert build_pose(visibility=0.95).is_reliable()


# ---- conversion boundary -------------------------------------------------


def test_no_detection_converts_to_none():
    empty = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])

    assert BodyPose.from_landmarker_result(empty, timestamp=1.0) is None


def test_conversion_is_duck_typed():
    """The converter never imports mediapipe, so plain objects are enough.

    That is what keeps the dependency arrow pointing one way - and what lets
    this test exist at all.
    """
    fake = [SimpleNamespace(x=0.1 * i, y=0.2, z=0.0, visibility=0.9) for i in range(33)]
    result = SimpleNamespace(pose_landmarks=[fake], pose_world_landmarks=[fake])

    pose = BodyPose.from_landmarker_result(result, timestamp=12.5)

    assert pose is not None
    assert pose.timestamp == 12.5
    assert len(pose.world) == 33
    assert pose.world[5].x == pytest.approx(0.5)


def test_pose_is_immutable():
    """Motion history keeps references to old poses; silent mutation there
    would be extremely hard to debug."""
    pose = build_pose()

    with pytest.raises(Exception):
        pose.timestamp = 99.0
