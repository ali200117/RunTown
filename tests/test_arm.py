"""Tests for ArmDetector.

The gesture: raise one arm out to the side, then lower it. The signal is that
arm's ELEVATION in degrees - 0 hanging, 90 straight out, 180 straight up -
with the right arm counting positive and the left negative.
"""

import math

import pytest

from kinetirun.calibration import Calibrator
from kinetirun.game import SUBWAY_SURFERS, GameKey
from kinetirun.input import InputDispatcher, RecordingAdapter
from kinetirun.movement import (
    ArmConfig,
    ArmDetector,
    ArmState,
    MovementEngine,
    MovementType,
)
from kinetirun.tracking import BodyPose, MotionHistory, Point3D
from kinetirun.vision.landmarks import Landmark

from test_body_pose import build_pose
from test_calibration import feed

FPS = 30.0
ARM_LENGTH = 0.35  # in normalized image units, shoulder to wrist

LEGS = {
    Landmark.LEFT_KNEE,
    Landmark.RIGHT_KNEE,
    Landmark.LEFT_ANKLE,
    Landmark.RIGHT_ANKLE,
}


def arm_pose(
    angle: float,
    timestamp: float,
    visibility: float = 1.0,
    legs_visible: bool = True,
    both_arms: bool = False,
):
    """A pose with one arm raised `angle` degrees. Positive raises the RIGHT arm.

    The wrist is placed on a circle around the shoulder: straight down at 0,
    straight out to the side at 90.
    """
    base = build_pose(timestamp=timestamp, visibility=visibility)
    points = list(base.image)

    def place_wrist(side_sign: int, elevation: float) -> None:
        # side_sign is the USER's side: +1 right, -1 left. MediaPipe's LEFT_*
        # landmarks are the user's right side, because frames are mirrored
        # before the model sees them.
        shoulder_index = (
            Landmark.LEFT_SHOULDER if side_sign > 0 else Landmark.RIGHT_SHOULDER
        )
        wrist_index = Landmark.LEFT_WRIST if side_sign > 0 else Landmark.RIGHT_WRIST
        shoulder = points[shoulder_index]

        radians = math.radians(elevation)
        points[wrist_index] = Point3D(
            shoulder.x + side_sign * ARM_LENGTH * math.sin(radians),
            shoulder.y + ARM_LENGTH * math.cos(radians),
            0.0,
            points[wrist_index].visibility,
        )

    # Both arms hang by default; the gesture raises one of them.
    place_wrist(1, abs(angle) if (angle > 0 or both_arms) else 0.0)
    place_wrist(-1, abs(angle) if (angle < 0 or both_arms) else 0.0)

    if not legs_visible:
        for index in LEGS:
            point = points[index]
            points[index] = Point3D(point.x, point.y, point.z, 0.05)

    return BodyPose(timestamp=timestamp, world=base.world, image=tuple(points))


@pytest.fixture
def baseline():
    """Calibrated from arms-hanging poses, which is how you stand for it."""
    calibrator = Calibrator(duration=1.0, min_samples=10)
    for i in range(40):
        calibrator.update(arm_pose(0.0, timestamp=i / FPS))
    return calibrator.baseline


def ramp(start: float, end: float, seconds: float) -> list[float]:
    frames = max(1, int(seconds * FPS))
    return [start + (end - start) * (i + 1) / frames for i in range(frames)]


def swipe_profile(
    angle: float = 85.0,
    up: float = 0.3,
    hold: float = 0.1,
    down: float = 0.3,
    settle: float = 0.3,
) -> list[float]:
    """A full swipe: arms down, raise, brief hold, lower, arms down."""
    return (
        [0.0] * int(0.4 * FPS)
        + ramp(0.0, angle, up)
        + [angle] * max(1, int(hold * FPS))
        + ramp(angle, 0.0, down)
        + [0.0] * int(settle * FPS)
    )


def simulate(
    angles,
    baseline,
    detector=None,
    visibility: float = 1.0,
    legs_visible: bool = True,
    dispatcher=None,
):
    detector = detector or ArmDetector()
    history = MotionHistory()
    events = []

    for i, angle in enumerate(angles):
        pose = arm_pose(
            angle, timestamp=i / FPS, visibility=visibility, legs_visible=legs_visible
        )
        history.append(pose)
        event = detector.update(pose, baseline, history)
        if event is not None:
            events.append(event)
            if dispatcher is not None:
                dispatcher.dispatch([event])

    return detector, events


# ---- the signal ----------------------------------------------------------


def test_a_hanging_arm_is_zero_degrees():
    assert arm_pose(0.0, timestamp=0.0).arm_elevation("left") == pytest.approx(
        0.0, abs=1.0
    )


def test_a_horizontal_arm_is_ninety_degrees():
    """arm_elevation takes MEDIAPIPE side names, so "left" here is the arm the
    user raised on their own right."""
    assert arm_pose(90.0, timestamp=0.0).arm_elevation("left") == pytest.approx(
        90.0, abs=1.0
    )


def test_mediapipe_side_labels_are_mirrored():
    """The bug this convention caused: raising the user's right arm must not
    read as their left. Frames are mirrored, so MediaPipe's RIGHT_* landmarks
    belong to the user's LEFT side."""
    user_raised_their_right = arm_pose(90.0, timestamp=0.0)

    assert user_raised_their_right.arm_elevation("left") > 80.0
    assert user_raised_their_right.arm_elevation("right") < 10.0
    assert user_raised_their_right.arm_raise > 0


def test_raising_the_right_arm_is_positive():
    assert arm_pose(80.0, timestamp=0.0).arm_raise > 0


def test_raising_the_left_arm_is_negative():
    assert arm_pose(-80.0, timestamp=0.0).arm_raise < 0


def test_raising_both_arms_cancels_out():
    """Ambiguous gestures must say nothing rather than guess a direction."""
    both = arm_pose(80.0, timestamp=0.0, both_arms=True)

    assert both.arm_raise == pytest.approx(0.0, abs=1.0)


def test_the_signal_is_measured_against_the_calibrated_rest_pose(baseline):
    assert baseline.arm_reach(arm_pose(70.0, timestamp=0.0)) == pytest.approx(
        70.0, abs=2.0
    )


# ---- the happy path ------------------------------------------------------


def test_raising_the_right_arm_emits_a_right_event(baseline):
    _, events = simulate(swipe_profile(), baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.SIDE_RIGHT_COMPLETED


def test_raising_the_left_arm_emits_a_left_event(baseline):
    _, events = simulate([-value for value in swipe_profile()], baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.SIDE_LEFT_COMPLETED


def test_it_works_with_the_legs_out_of_frame(baseline):
    """The reason this mode exists: step and lean both need the lower body,
    which is often not in frame in a small room."""
    _, events = simulate(swipe_profile(), baseline, legs_visible=False)

    assert len(events) == 1


def test_a_swipe_is_quick(baseline):
    """It has to be fast enough to react with in a game."""
    _, events = simulate(swipe_profile(), baseline)

    assert events[0].duration < 1.0


def test_raising_the_arm_higher_scores_higher(baseline):
    _, low = simulate(swipe_profile(angle=60.0), baseline)
    _, high = simulate(swipe_profile(angle=95.0), baseline)

    assert high[0].quality > low[0].quality


def test_using_each_arm_emits_each_direction(baseline):
    right = swipe_profile()
    left = [-value for value in swipe_profile()]

    _, events = simulate(right + left, baseline)

    assert [event.type for event in events] == [
        MovementType.SIDE_RIGHT_COMPLETED,
        MovementType.SIDE_LEFT_COMPLETED,
    ]


def test_the_detector_returns_to_rest(baseline):
    detector, _ = simulate(swipe_profile(), baseline)

    assert detector.state is ArmState.REST


# ---- the things that must NOT count --------------------------------------


def test_a_half_raised_arm_is_rejected(baseline):
    _, events = simulate(swipe_profile(angle=35.0), baseline)

    assert events == []


def test_raising_the_arm_slowly_is_rejected(baseline):
    """Stretching or reaching for something must not steer the game."""
    _, events = simulate(swipe_profile(angle=85.0, up=2.5, down=2.5), baseline)

    assert events == []


def test_repeated_small_movements_never_accumulate(baseline):
    fidgeting = []
    for _ in range(6):
        fidgeting += ramp(0.0, 30.0, 0.2) + ramp(30.0, 0.0, 0.2)

    _, events = simulate([0.0] * 12 + fidgeting, baseline)

    assert events == []


def test_keeping_the_arm_up_never_completes(baseline):
    detector, events = simulate(
        [0.0] * 12 + ramp(0.0, 85.0, 0.3) + [85.0] * int(1.2 * FPS), baseline
    )

    assert events == []
    assert detector.state is not ArmState.REST


def test_holding_too_long_is_abandoned(baseline):
    detector = ArmDetector(ArmConfig(max_duration=0.5))

    _, events = simulate(swipe_profile(hold=1.5), baseline, detector=detector)

    assert events == []


def test_arms_at_rest_produce_nothing(baseline):
    _, events = simulate([0.0] * 120, baseline)

    assert events == []


def test_jitter_produces_nothing(baseline):
    noise = [4.0 * (i % 3 - 1) for i in range(120)]

    _, events = simulate(noise, baseline)

    assert events == []


# ---- confidence, recovery and reporting ----------------------------------


def test_untracked_arms_produce_nothing(baseline):
    _, events = simulate(swipe_profile(), baseline, visibility=0.1)

    assert events == []


def test_it_re_arms_after_a_failed_attempt(baseline):
    detector, _ = simulate(swipe_profile(angle=35.0), baseline)
    _, events = simulate(swipe_profile(), baseline, detector=detector)

    assert len(events) == 1


def test_a_rejected_swipe_reports_why(baseline):
    detector, _ = simulate(swipe_profile(angle=35.0), baseline)

    assert "not raised enough" in detector.last_rejection


def test_thresholds_are_configurable(baseline):
    strict = ArmDetector(ArmConfig(min_angle=150.0))

    _, events = simulate(swipe_profile(angle=85.0), baseline, detector=strict)

    assert events == []


# ---- integration ---------------------------------------------------------


def test_arm_is_the_default_sideways_mode():
    assert MovementEngine().side.__class__.__name__ == "ArmDetector"


def test_a_swipe_becomes_an_arrow_key(baseline):
    """Downstream cannot tell a swipe from a step: same event, same key."""
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter, enabled=True)

    simulate(swipe_profile(), baseline, dispatcher=dispatcher)

    assert adapter.presses == [GameKey.RIGHT]
