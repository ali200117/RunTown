"""Tests for SquatDetector.

The payoff for keeping the movement engine free of OpenCV and MediaPipe: a
complete squat can be scripted as a list of depths and pushed through the real
detector, with no camera, no model and no lighting.

Depth is `hip_drop`: 0.0 standing, 0.35 a solid squat.
"""

import pytest

from kinetirun.calibration import Calibrator
from kinetirun.movement import MovementType, SquatConfig, SquatDetector, SquatState
from kinetirun.tracking import MotionHistory

from test_body_pose import build_pose
from test_calibration import feed

FPS = 30.0
STANDING_HIP_HEIGHT = 0.90  # metres, from build_pose defaults


@pytest.fixture
def baseline():
    return feed(Calibrator(duration=1.0, min_samples=10)).baseline


def ramp(start: float, end: float, seconds: float) -> list[float]:
    """A linear change in depth, sampled at FPS."""
    frames = max(1, int(seconds * FPS))
    return [start + (end - start) * (i + 1) / frames for i in range(frames)]


def squat_profile(
    depth: float = 0.35,
    down: float = 0.5,
    hold: float = 0.25,
    up: float = 0.5,
    settle: float = 0.5,
) -> list[float]:
    """A full squat: stand, descend, hold, rise, stand again."""
    return (
        [0.0] * int(0.5 * FPS)
        + ramp(0.0, depth, down)
        + [depth] * int(hold * FPS)
        + ramp(depth, 0.0, up)
        + [0.0] * int(settle * FPS)
    )


def simulate(
    depths,
    baseline,
    detector=None,
    knee_ratio: float = 0.7,
    visibility: float = 1.0,
):
    """Push a depth sequence through the real detector.

    knee_ratio controls how far the knees travel forward per unit of depth.
    Setting it to 0 produces someone lowering their hips by bending at the
    waist with straight legs - which must not count as a squat.
    """
    detector = detector or SquatDetector()
    history = MotionHistory()
    events = []

    for i, depth in enumerate(depths):
        pose = build_pose(
            hip_y=depth * STANDING_HIP_HEIGHT,
            knee_x_offset=depth * knee_ratio,
            timestamp=i / FPS,
            visibility=visibility,
        )
        history.append(pose)
        event = detector.update(pose, baseline, history)
        if event is not None:
            events.append(event)

    return detector, events


# ---- the happy path ------------------------------------------------------


def test_a_good_squat_emits_one_event(baseline):
    _, events = simulate(squat_profile(), baseline)

    assert len(events) == 1
    assert events[0].type is MovementType.SQUAT_COMPLETED


def test_the_event_carries_useful_metadata(baseline):
    _, events = simulate(squat_profile(depth=0.40), baseline)
    event = events[0]

    assert event.displacement == pytest.approx(0.40, abs=0.02)
    # The clock starts when the descent is detected, a few frames after the
    # hips begin to move, so this is shorter than the scripted 1.25 s.
    assert 0.9 <= event.duration <= 1.6
    assert 0.0 < event.quality <= 1.0


def test_a_deeper_squat_scores_higher(baseline):
    _, shallow = simulate(squat_profile(depth=0.29), baseline)
    _, deep = simulate(squat_profile(depth=0.45), baseline)

    assert deep[0].quality > shallow[0].quality


def test_the_detector_returns_to_ready(baseline):
    detector, _ = simulate(squat_profile(), baseline)

    assert detector.state is SquatState.READY


def test_two_squats_emit_two_events(baseline):
    detector, events = simulate(squat_profile() + squat_profile(), baseline)

    assert len(events) == 2


# ---- the things that must NOT count --------------------------------------


def test_a_shallow_dip_is_rejected(baseline):
    """The whole point of the project: a small bob is not a squat."""
    _, events = simulate(squat_profile(depth=0.18), baseline)

    assert events == []


def test_repeated_half_squats_never_accumulate(baseline):
    """Bouncing must not eventually be accepted. This is what WAIT_FOR_NEUTRAL
    is for, and why an arbitrary cooldown is not needed."""
    bobbing = []
    for _ in range(6):
        bobbing += ramp(0.0, 0.20, 0.25) + ramp(0.20, 0.0, 0.25)

    _, events = simulate([0.0] * 15 + bobbing, baseline)

    assert events == []


def test_a_bounce_is_too_fast(baseline):
    """Deep enough, but over in a fraction of a second."""
    _, events = simulate(
        squat_profile(depth=0.40, down=0.12, hold=0.0, up=0.12), baseline
    )

    assert events == []


def test_bending_at_the_waist_is_rejected(baseline):
    """Hips drop far enough, but the knees never bend. The second, independent
    signal is what catches this - hip depth alone would be fooled."""
    _, events = simulate(squat_profile(depth=0.40), baseline, knee_ratio=0.0)

    assert events == []


def test_staying_down_never_completes(baseline):
    """A squat is not finished until the user stands back up."""
    detector, events = simulate(
        [0.0] * 15 + ramp(0.0, 0.35, 0.5) + [0.35] * int(2 * FPS), baseline
    )

    assert events == []
    assert detector.state is not SquatState.READY


def test_squatting_for_too_long_is_abandoned(baseline):
    """Standing around bent over is not a movement."""
    config = SquatConfig(max_duration=1.0)
    detector, events = simulate(
        squat_profile(hold=2.0), baseline, detector=SquatDetector(config)
    )

    assert events == []


def test_standing_still_produces_nothing(baseline):
    _, events = simulate([0.0] * 120, baseline)

    assert events == []


def test_jitter_while_standing_produces_nothing(baseline):
    """Residual noise after smoothing must not trip the descent trigger."""
    noise = [0.02 * (i % 3 - 1) for i in range(120)]

    _, events = simulate(noise, baseline)

    assert events == []


# ---- confidence and recovery ---------------------------------------------


def test_an_untracked_body_produces_nothing(baseline):
    """MediaPipe returns coordinates for joints it cannot see. Acting on those
    would mean measuring a fabricated leg."""
    _, events = simulate(squat_profile(), baseline, visibility=0.1)

    assert events == []


def test_losing_tracking_mid_squat_abandons_the_attempt(baseline):
    detector = SquatDetector()
    history = MotionHistory()
    baseline_ = baseline

    # Descend normally.
    depths = [0.0] * 15 + ramp(0.0, 0.35, 0.5)
    for i, depth in enumerate(depths):
        pose = build_pose(
            hip_y=depth * STANDING_HIP_HEIGHT,
            knee_x_offset=depth * 0.7,
            timestamp=i / FPS,
        )
        history.append(pose)
        detector.update(pose, baseline_, history)

    # Tracking drops out, then the user reappears standing.
    detector.update(None, baseline_, history)

    assert detector.state is SquatState.WAIT_FOR_NEUTRAL


def test_it_re_arms_after_a_failed_attempt(baseline):
    """A rejected squat must not leave the detector stuck."""
    detector, _ = simulate(squat_profile(depth=0.18), baseline)
    _, events = simulate(squat_profile(), baseline, detector=detector)

    assert len(events) == 1


# ---- configurability -----------------------------------------------------


def test_thresholds_are_configurable(baseline):
    """Requiring a harder squat is a config change, not a code change."""
    strict = SquatDetector(SquatConfig(min_depth=0.45))

    _, events = simulate(squat_profile(depth=0.35), baseline, detector=strict)

    assert events == []
