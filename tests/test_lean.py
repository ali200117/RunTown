"""Tests for LeanDetector.

The gesture is a sideways bow: feet planted, torso tilting over. Angles are in
degrees relative to the calibrated standing posture.
"""

import math

import pytest

from kinetirun.calibration import Calibrator
from kinetirun.game import SUBWAY_SURFERS, GameKey
from kinetirun.input import InputDispatcher, RecordingAdapter
from kinetirun.movement import LeanConfig, LeanDetector, LeanState, MovementType
from kinetirun.tracking import BodyPose, MotionHistory, Point3D
from kinetirun.vision.landmarks import Landmark

from test_body_pose import build_pose
from test_calibration import feed

FPS = 30.0

# Landmarks above the hips: these are what swing when you bow sideways.
UPPER_BODY = {
    Landmark.LEFT_SHOULDER,
    Landmark.RIGHT_SHOULDER,
    Landmark.LEFT_ELBOW,
    Landmark.RIGHT_ELBOW,
    Landmark.LEFT_WRIST,
    Landmark.RIGHT_WRIST,
    Landmark.NOSE,
}


@pytest.fixture
def baseline():
    return feed(Calibrator(duration=1.0, min_samples=10)).baseline


def lean_pose(angle_degrees: float, timestamp: float, visibility: float = 1.0):
    """A pose whose torso is tilted `angle_degrees` from vertical.

    build_pose puts the hips at y=0 and the shoulders at y=-0.50, so the torso
    is 0.50 long. Tilting it by an angle moves the shoulders sideways by
    0.50 * tan(angle) - positive is toward the screen's right.
    """
    base = build_pose(timestamp=timestamp, visibility=visibility)
    torso_length = 0.50
    shift = torso_length * math.tan(math.radians(angle_degrees))

    image = tuple(
        Point3D(
            point.x + (shift if index in UPPER_BODY else 0.0),
            point.y,
            point.z,
            point.visibility,
        )
        for index, point in enumerate(base.image)
    )
    return BodyPose(timestamp=timestamp, world=base.world, image=image)


def ramp(start: float, end: float, seconds: float) -> list[float]:
    frames = max(1, int(seconds * FPS))
    return [start + (end - start) * (i + 1) / frames for i in range(frames)]


def lean_profile(
    angle: float = 22.0,
    over: float = 0.35,
    hold: float = 0.2,
    back: float = 0.35,
    settle: float = 0.4,
) -> list[float]:
    """A full lean: upright, tilt over, hold, straighten, upright."""
    return (
        [0.0] * int(0.4 * FPS)
        + ramp(0.0, angle, over)
        + [angle] * int(hold * FPS)
        + ramp(angle, 0.0, back)
        + [0.0] * int(settle * FPS)
    )


def simulate(angles, baseline, detector=None, visibility: float = 1.0, dispatcher=None):
    detector = detector or LeanDetector()
    history = MotionHistory()
    events = []

    for i, angle in enumerate(angles):
        pose = lean_pose(angle, timestamp=i / FPS, visibility=visibility)
        history.append(pose)
        event = detector.update(pose, baseline, history)
        if event is not None:
            events.append(event)
            if dispatcher is not None:
                dispatcher.dispatch([event])

    return detector, events


# ---- the geometry --------------------------------------------------------


def test_standing_upright_is_zero_degrees():
    assert build_pose().torso_tilt == pytest.approx(0.0)


def test_the_tilt_angle_matches_the_pose():
    assert lean_pose(20.0, timestamp=0.0).torso_tilt == pytest.approx(20.0, abs=0.5)


def test_tilt_is_signed():
    assert lean_pose(-20.0, timestamp=0.0).torso_tilt < 0


def test_tilt_is_measured_against_the_calibrated_posture(baseline):
    """Nobody stands perfectly straight and no camera is perfectly level.
    Measuring against the baseline cancels both errors out."""
    assert baseline.lean_angle(lean_pose(18.0, timestamp=0.0)) == pytest.approx(
        18.0, abs=0.5
    )


# ---- the happy path ------------------------------------------------------


def test_leaning_right_emits_a_right_event(baseline):
    _, events = simulate(lean_profile(), baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.SIDE_RIGHT_COMPLETED


def test_leaning_left_emits_a_left_event(baseline):
    """Frames are mirrored, so a negative tilt is the user's own left."""
    _, events = simulate([-value for value in lean_profile()], baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.SIDE_LEFT_COMPLETED


def test_the_event_carries_useful_metadata(baseline):
    _, events = simulate(lean_profile(angle=26.0), baseline)
    event = events[0]

    assert event.displacement == pytest.approx(26.0, abs=2.0)
    assert event.duration > 0.2
    assert 0.0 < event.quality <= 1.0


def test_a_deeper_lean_scores_higher(baseline):
    _, shallow = simulate(lean_profile(angle=16.0), baseline)
    _, deep = simulate(lean_profile(angle=30.0), baseline)

    assert deep[0].quality > shallow[0].quality


def test_leaning_both_ways_emits_both_events(baseline):
    right = lean_profile()
    left = [-value for value in lean_profile()]

    _, events = simulate(right + left, baseline)

    assert [event.type for event in events] == [
        MovementType.SIDE_RIGHT_COMPLETED,
        MovementType.SIDE_LEFT_COMPLETED,
    ]


def test_the_detector_returns_to_upright(baseline):
    detector, _ = simulate(lean_profile(), baseline)

    assert detector.state is LeanState.UPRIGHT


# ---- the things that must NOT count --------------------------------------


def test_a_small_tilt_is_rejected(baseline):
    """Normal postural sway must not steer the game."""
    _, events = simulate(lean_profile(angle=8.0), baseline)

    assert events == []


def test_repeated_small_tilts_never_accumulate(baseline):
    swaying = []
    for _ in range(6):
        swaying += ramp(0.0, 8.0, 0.25) + ramp(8.0, 0.0, 0.25)

    _, events = simulate([0.0] * 12 + swaying, baseline)

    assert events == []


def test_drifting_over_slowly_is_rejected(baseline):
    """Same angle, but far too slow to be a deliberate gesture."""
    _, events = simulate(lean_profile(angle=25.0, over=2.5, back=2.5), baseline)

    assert events == []


def test_not_straightening_up_never_completes(baseline):
    """Staying leaned over is not a completed movement."""
    detector, events = simulate(
        [0.0] * 12 + ramp(0.0, 25.0, 0.35) + [25.0] * int(1.5 * FPS), baseline
    )

    assert events == []
    assert detector.state is not LeanState.UPRIGHT


def test_holding_the_lean_too_long_is_abandoned(baseline):
    detector = LeanDetector(LeanConfig(max_duration=0.8))

    _, events = simulate(lean_profile(hold=2.0), baseline, detector=detector)

    assert events == []


def test_standing_still_produces_nothing(baseline):
    _, events = simulate([0.0] * 120, baseline)

    assert events == []


def test_jitter_produces_nothing(baseline):
    noise = [1.5 * (i % 3 - 1) for i in range(120)]

    _, events = simulate(noise, baseline)

    assert events == []


# ---- confidence, recovery and reporting ----------------------------------


def test_an_untracked_body_produces_nothing(baseline):
    _, events = simulate(lean_profile(), baseline, visibility=0.1)

    assert events == []


def test_it_re_arms_after_a_failed_attempt(baseline):
    detector, _ = simulate(lean_profile(angle=8.0), baseline)
    _, events = simulate(lean_profile(), baseline, detector=detector)

    assert len(events) == 1


def test_a_rejected_lean_reports_why(baseline):
    """The tuning readout: "it does not work" becomes a number to change."""
    detector, _ = simulate(lean_profile(angle=8.0), baseline)

    assert "not far enough" in detector.last_rejection


def test_thresholds_are_configurable(baseline):
    strict = LeanDetector(LeanConfig(min_angle=35.0))

    _, events = simulate(lean_profile(angle=22.0), baseline, detector=strict)

    assert events == []


# ---- end to end ----------------------------------------------------------


def test_a_lean_becomes_an_arrow_key(baseline):
    """Downstream cannot tell a lean from a step: same event, same key."""
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter, enabled=True)

    simulate(lean_profile(), baseline, dispatcher=dispatcher)

    assert adapter.presses == [GameKey.RIGHT]
