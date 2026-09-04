"""Tests for SideDetector.

Offsets are in shoulder widths. build_pose gives a shoulder width of 0.40 in
image units, so a shift of 0.28 across the frame is 0.7 shoulder widths.
"""

import pytest

from kinetirun.calibration import Calibrator
from kinetirun.movement import MovementType, SideConfig, SideDetector, SideState
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


def side_pose(offset: float, foot_ratio: float, timestamp: float, visibility: float = 1.0):
    """A pose displaced sideways by `offset` shoulder widths.

    foot_ratio says how much of that the FEET took part in: 1.0 is a real step,
    0.0 is leaning the upper body while the feet stay planted.
    """
    base = build_pose(timestamp=timestamp, visibility=visibility)
    body_shift = offset * SHOULDER_WIDTH_IMAGE
    foot_shift = body_shift * foot_ratio

    image = tuple(
        Point3D(
            point.x + (foot_shift if index in FOOT_LANDMARKS else body_shift),
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


def step_profile(
    offset: float = 0.75,
    out: float = 0.4,
    hold: float = 0.2,
    back: float = 0.4,
    settle: float = 0.4,
) -> list[float]:
    """A full side step: centred, out, hold, back, centred."""
    return (
        [0.0] * int(0.4 * FPS)
        + ramp(0.0, offset, out)
        + [offset] * int(hold * FPS)
        + ramp(offset, 0.0, back)
        + [0.0] * int(settle * FPS)
    )


def simulate(offsets, baseline, detector=None, foot_ratio: float = 1.0, visibility: float = 1.0):
    detector = detector or SideDetector()
    history = MotionHistory()
    events = []

    for i, offset in enumerate(offsets):
        pose = side_pose(offset, foot_ratio, timestamp=i / FPS, visibility=visibility)
        history.append(pose)
        event = detector.update(pose, baseline, history)
        if event is not None:
            events.append(event)

    return detector, events


# ---- the happy path ------------------------------------------------------


def test_a_step_right_emits_a_right_event(baseline):
    _, events = simulate(step_profile(), baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.SIDE_RIGHT_COMPLETED


def test_a_step_left_emits_a_left_event(baseline):
    """Frames are mirrored, so a negative offset is the user's own left."""
    _, events = simulate([-value for value in step_profile()], baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.SIDE_LEFT_COMPLETED


def test_the_event_carries_useful_metadata(baseline):
    _, events = simulate(step_profile(offset=0.80), baseline)
    event = events[0]

    assert event.displacement == pytest.approx(0.80, abs=0.05)
    assert event.duration > 0.25
    assert 0.0 < event.quality <= 1.0


def test_a_bigger_step_scores_higher(baseline):
    _, small = simulate(step_profile(offset=0.58), baseline)
    _, large = simulate(step_profile(offset=1.00), baseline)

    assert large[0].quality > small[0].quality


def test_stepping_both_ways_emits_both_events(baseline):
    right = step_profile()
    left = [-value for value in step_profile()]

    _, events = simulate(right + left, baseline)

    assert [event.type for event in events] == [
        MovementType.SIDE_RIGHT_COMPLETED,
        MovementType.SIDE_LEFT_COMPLETED,
    ]


def test_the_detector_returns_to_center(baseline):
    detector, _ = simulate(step_profile(), baseline)

    assert detector.state is SideState.CENTER


# ---- the things that must NOT count --------------------------------------


def test_leaning_is_rejected(baseline):
    """The requirement this phase exists for: tilting the upper body moves the
    hips without moving the feet, and is not a side step."""
    _, events = simulate(step_profile(offset=0.90), baseline, foot_ratio=0.0)

    assert events == []


def test_a_small_sway_is_rejected(baseline):
    _, events = simulate(step_profile(offset=0.35), baseline)

    assert events == []


def test_repeated_small_sways_never_accumulate(baseline):
    swaying = []
    for _ in range(6):
        swaying += ramp(0.0, 0.35, 0.25) + ramp(0.35, 0.0, 0.25)

    _, events = simulate([0.0] * 12 + swaying, baseline)

    assert events == []


def test_not_returning_to_center_never_completes(baseline):
    """Stepping out and staying there is not a completed movement."""
    detector, events = simulate(
        [0.0] * 12 + ramp(0.0, 0.75, 0.4) + [0.75] * int(1.5 * FPS), baseline
    )

    assert events == []
    assert detector.state is not SideState.CENTER


def test_staying_out_too_long_is_abandoned(baseline):
    detector = SideDetector(SideConfig(max_duration=0.8))

    _, events = simulate(step_profile(hold=2.0), baseline, detector=detector)

    assert events == []


def test_standing_still_produces_nothing(baseline):
    _, events = simulate([0.0] * 120, baseline)

    assert events == []


def test_jitter_while_centred_produces_nothing(baseline):
    noise = [0.05 * (i % 3 - 1) for i in range(120)]

    _, events = simulate(noise, baseline)

    assert events == []


# ---- confidence and recovery ---------------------------------------------


def test_an_untracked_body_produces_nothing(baseline):
    _, events = simulate(step_profile(), baseline, visibility=0.1)

    assert events == []


def test_it_re_arms_after_a_failed_attempt(baseline):
    detector, _ = simulate(step_profile(offset=0.35), baseline)
    _, events = simulate(step_profile(), baseline, detector=detector)

    assert len(events) == 1


# ---- configurability -----------------------------------------------------


def test_the_foot_rule_can_be_disabled(baseline):
    """Feet often fall outside the frame in a small room, so the anti-lean rule
    has to be optional rather than a hard assumption."""
    lenient = SideDetector(SideConfig(min_foot_travel=0.0))

    _, events = simulate(step_profile(offset=0.90), baseline, detector=lenient, foot_ratio=0.0)

    assert len(events) == 1


def test_thresholds_are_configurable(baseline):
    strict = SideDetector(SideConfig(min_offset=1.2))

    _, events = simulate(step_profile(offset=0.75), baseline, detector=strict)

    assert events == []
