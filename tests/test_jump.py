"""Tests for JumpDetector.

Lift is upward hip displacement in shoulder widths, positive up. build_pose
gives a shoulder width of 0.40 in image units, so a lift of 0.25 shoulder
widths is a shift of 0.10 up the frame.
"""

import pytest

from kinetirun.calibration import Calibrator
from kinetirun.movement import JumpConfig, JumpDetector, JumpState, MovementType
from kinetirun.tracking import BodyPose, MotionHistory, Point3D
from kinetirun.vision.landmarks import Landmark

from test_body_pose import build_pose
from test_calibration import feed

FPS = 30.0
SHOULDER_WIDTH_IMAGE = 0.40

FOOT_LANDMARKS = {
    Landmark.LEFT_ANKLE,
    Landmark.RIGHT_ANKLE,
    Landmark.LEFT_HEEL,
    Landmark.RIGHT_HEEL,
    Landmark.LEFT_FOOT_INDEX,
    Landmark.RIGHT_FOOT_INDEX,
}


@pytest.fixture
def baseline():
    return feed(Calibrator(duration=1.0, min_samples=10)).baseline


def jump_pose(lift: float, foot_ratio: float, timestamp: float, visibility: float = 1.0):
    """A pose raised by `lift` shoulder widths.

    foot_ratio says how much the FEET took part: 1.0 is a real jump, 0.0 is
    rising onto the toes with the ankles staying put.
    """
    base = build_pose(timestamp=timestamp, visibility=visibility)
    # Image y grows downward, so rising means subtracting.
    body_shift = -lift * SHOULDER_WIDTH_IMAGE
    foot_shift = body_shift * foot_ratio

    image = tuple(
        Point3D(
            point.x,
            point.y + (foot_shift if index in FOOT_LANDMARKS else body_shift),
            point.z,
            point.visibility,
        )
        for index, point in enumerate(base.image)
    )
    return BodyPose(timestamp=timestamp, world=base.world, image=image)


def ramp(start: float, end: float, seconds: float) -> list[float]:
    frames = max(1, int(seconds * FPS))
    return [start + (end - start) * (i + 1) / frames for i in range(frames)]


def jump_profile(
    height: float = 0.28,
    rise: float = 0.18,
    hang: float = 0.07,
    fall: float = 0.18,
    settle: float = 0.4,
) -> list[float]:
    """A full jump: grounded, fast rise, brief hang, fall, grounded."""
    return (
        [0.0] * int(0.4 * FPS)
        + ramp(0.0, height, rise)
        + [height] * max(1, int(hang * FPS))
        + ramp(height, 0.0, fall)
        + [0.0] * int(settle * FPS)
    )


def simulate(lifts, baseline, detector=None, foot_ratio: float = 1.0, visibility: float = 1.0):
    detector = detector or JumpDetector()
    history = MotionHistory()
    events = []

    for i, lift in enumerate(lifts):
        pose = jump_pose(lift, foot_ratio, timestamp=i / FPS, visibility=visibility)
        history.append(pose)
        event = detector.update(pose, baseline, history)
        if event is not None:
            events.append(event)

    return detector, events


# ---- the happy path ------------------------------------------------------


def test_a_good_jump_emits_one_event(baseline):
    _, events = simulate(jump_profile(), baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.JUMP_COMPLETED


def test_the_event_carries_useful_metadata(baseline):
    _, events = simulate(jump_profile(height=0.32), baseline)
    event = events[0]

    assert event.displacement == pytest.approx(0.32, abs=0.03)
    assert 0.15 <= event.duration <= 1.0
    assert 0.0 < event.quality <= 1.0


def test_a_higher_jump_scores_higher(baseline):
    _, low = simulate(jump_profile(height=0.20), baseline)
    _, high = simulate(jump_profile(height=0.40), baseline)

    assert high[0].quality > low[0].quality


def test_the_detector_returns_to_grounded(baseline):
    detector, _ = simulate(jump_profile(), baseline)

    assert detector.state is JumpState.GROUNDED


def test_two_jumps_emit_two_events(baseline):
    _, events = simulate(jump_profile() + jump_profile(), baseline)

    assert len(events) == 2


# ---- the things that must NOT count --------------------------------------


def test_rising_onto_the_toes_is_rejected(baseline):
    """Hips rise far enough and fast enough, but the feet never leave the
    ground. The independent foot signal is what catches this."""
    _, events = simulate(jump_profile(height=0.30), baseline, foot_ratio=0.0)

    assert events == []


def test_a_small_hop_is_rejected(baseline):
    _, events = simulate(jump_profile(height=0.08), baseline)

    assert events == []


def test_standing_up_slowly_is_not_a_jump(baseline):
    """Same displacement, but nowhere near the speed. Speed is the signal that
    separates a jump from any other upward movement."""
    _, events = simulate(jump_profile(height=0.30, rise=1.2, fall=1.2), baseline)

    assert events == []


def test_standing_still_produces_nothing(baseline):
    _, events = simulate([0.0] * 120, baseline)

    assert events == []


def test_jitter_produces_nothing(baseline):
    noise = [0.02 * (i % 3 - 1) for i in range(120)]

    _, events = simulate(noise, baseline)

    assert events == []


def test_staying_up_never_completes(baseline):
    """Climbing onto something and staying there is not a jump."""
    detector, events = simulate(
        [0.0] * 12 + ramp(0.0, 0.30, 0.18) + [0.30] * int(1.0 * FPS), baseline
    )

    assert events == []
    assert detector.state is not JumpState.GROUNDED


def test_hanging_too_long_is_abandoned(baseline):
    detector = JumpDetector(JumpConfig(max_duration=0.5))

    _, events = simulate(jump_profile(hang=1.5), baseline, detector=detector)

    assert events == []


def test_a_squat_does_not_register_as_a_jump(baseline):
    """Standing back up from a squat is an upward movement that ends at the
    baseline. It must never reach the required height above it."""
    squat_then_stand = (
        [0.0] * 12 + ramp(0.0, -0.35, 0.5) + [-0.35] * 6 + ramp(-0.35, 0.0, 0.35)
    )

    _, events = simulate(squat_then_stand + [0.0] * 20, baseline)

    assert events == []


# ---- confidence and recovery ---------------------------------------------


def test_an_untracked_body_produces_nothing(baseline):
    _, events = simulate(jump_profile(), baseline, visibility=0.1)

    assert events == []


def test_it_re_arms_after_a_failed_attempt(baseline):
    detector, _ = simulate(jump_profile(height=0.08), baseline)
    _, events = simulate(jump_profile(), baseline, detector=detector)

    assert len(events) == 1


# ---- configurability -----------------------------------------------------


def test_the_foot_rule_can_be_disabled(baseline):
    lenient = JumpDetector(JumpConfig(min_foot_lift=0.0))

    _, events = simulate(jump_profile(height=0.30), baseline, detector=lenient, foot_ratio=0.0)

    assert len(events) == 1


def test_thresholds_are_configurable(baseline):
    strict = JumpDetector(JumpConfig(min_height=0.60))

    _, events = simulate(jump_profile(height=0.28), baseline, detector=strict)

    assert events == []
