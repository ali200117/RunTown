"""Tests for the movement engine, game profile and input dispatcher.

The last test in this file is the one that matters most: a scripted squat goes
in as poses and a DOWN key comes out, through the real engine, the real profile
and the real dispatcher. Nothing is mocked except the final keystroke.
"""

import pytest

from kinetirun.calibration import Calibrator
from kinetirun.game import SUBWAY_SURFERS, GameKey, GameProfile
from kinetirun.input import InputDispatcher, RecordingAdapter
from kinetirun.movement import MovementEngine, MovementEvent, MovementType
from kinetirun.tracking import MotionHistory

from test_calibration import feed
from test_jump import jump_pose, jump_profile
from test_side import side_pose, step_profile
from test_squat import STANDING_HIP_HEIGHT, squat_profile
from test_body_pose import build_pose

FPS = 30.0


@pytest.fixture
def baseline():
    return feed(Calibrator(duration=1.0, min_samples=10)).baseline


def event(movement: MovementType) -> MovementEvent:
    return MovementEvent(
        type=movement, timestamp=1.0, duration=0.8, displacement=0.35, quality=0.9
    )


# ---- game profile --------------------------------------------------------


def test_subway_maps_all_four_movements():
    assert SUBWAY_SURFERS.key_for(MovementType.SIDE_LEFT_COMPLETED) is GameKey.LEFT
    assert SUBWAY_SURFERS.key_for(MovementType.SIDE_RIGHT_COMPLETED) is GameKey.RIGHT
    assert SUBWAY_SURFERS.key_for(MovementType.JUMP_COMPLETED) is GameKey.UP
    assert SUBWAY_SURFERS.key_for(MovementType.SQUAT_COMPLETED) is GameKey.DOWN


def test_an_unmapped_movement_returns_none():
    """A game with no crouch should simply ignore squats, not crash."""
    partial = GameProfile(
        name="Jump only",
        mapping={MovementType.JUMP_COMPLETED: GameKey.SPACE},
    )

    assert partial.key_for(MovementType.SQUAT_COMPLETED) is None


def test_a_new_game_is_a_new_mapping_not_new_code():
    """The whole point of the seam: another game changes data, not detectors."""
    temple_run = GameProfile(
        name="Temple Run",
        mapping={
            MovementType.JUMP_COMPLETED: GameKey.SPACE,
            MovementType.SQUAT_COMPLETED: GameKey.DOWN,
        },
    )

    assert temple_run.key_for(MovementType.JUMP_COMPLETED) is GameKey.SPACE


# ---- dispatcher ----------------------------------------------------------


def test_nothing_is_sent_while_disabled():
    """Disabled is the default. Keys go to whichever window has focus, so a
    session must never start typing without being asked."""
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter)

    dispatcher.dispatch([event(MovementType.JUMP_COMPLETED)])

    assert adapter.presses == []
    assert dispatcher.status == "OFF"


def test_enabling_lets_input_through():
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter)

    dispatcher.toggle()
    dispatcher.dispatch([event(MovementType.JUMP_COMPLETED)])

    assert adapter.presses == [GameKey.UP]


def test_each_movement_maps_to_its_key():
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter, enabled=True)

    dispatcher.dispatch(
        [
            event(MovementType.SQUAT_COMPLETED),
            event(MovementType.SIDE_LEFT_COMPLETED),
            event(MovementType.SIDE_RIGHT_COMPLETED),
            event(MovementType.JUMP_COMPLETED),
        ]
    )

    assert adapter.presses == [GameKey.DOWN, GameKey.LEFT, GameKey.RIGHT, GameKey.UP]


def test_unmapped_events_are_skipped_silently():
    adapter = RecordingAdapter()
    profile = GameProfile(name="Jump only", mapping={MovementType.JUMP_COMPLETED: GameKey.UP})
    dispatcher = InputDispatcher(profile, adapter, enabled=True)

    pressed = dispatcher.dispatch(
        [event(MovementType.SQUAT_COMPLETED), event(MovementType.JUMP_COMPLETED)]
    )

    assert pressed == [GameKey.UP]


def test_toggle_flips_both_ways():
    dispatcher = InputDispatcher(SUBWAY_SURFERS, RecordingAdapter())

    assert dispatcher.toggle() is True
    assert dispatcher.toggle() is False


# ---- movement engine -----------------------------------------------------


def run_engine(poses, baseline, engine=None, dispatcher=None):
    """Push poses through the engine, and optionally on to a dispatcher."""
    engine = engine or MovementEngine()
    history = MotionHistory()
    events = []

    for pose in poses:
        history.append(pose)
        frame_events = engine.update(pose, baseline, history)
        events += frame_events
        if dispatcher is not None:
            dispatcher.dispatch(frame_events)

    return engine, events


def squat_poses():
    return [
        build_pose(
            hip_y=depth * STANDING_HIP_HEIGHT,
            knee_x_offset=depth * 0.7,
            timestamp=i / FPS,
        )
        for i, depth in enumerate(squat_profile())
    ]


def test_the_engine_rejects_an_unknown_sideways_mode():
    with pytest.raises(ValueError):
        MovementEngine(sideways="sidestep")


def test_lean_and_step_modes_emit_the_same_event_type():
    """The point of the seam: the game layer cannot tell which one is in use."""
    assert MovementEngine(sideways="lean").side.__class__.__name__ == "LeanDetector"
    assert MovementEngine(sideways="step").side.__class__.__name__ == "SideDetector"


def test_the_engine_runs_every_detector(baseline):
    engine, _ = run_engine(squat_poses(), baseline)

    assert set(engine.states) == {"SQUAT", "SIDE", "JUMP"}


def test_the_engine_counts_movements(baseline):
    engine, events = run_engine(squat_poses(), baseline)

    assert len(events) == 1
    assert engine.counts["SQUAT_COMPLETED"] == 1
    assert engine.last_event is events[0]


def test_a_squat_does_not_trigger_the_other_detectors(baseline):
    """Detectors are independent, so a clean squat must not also look like a
    jump or a side step to the ones that were not involved."""
    _, events = run_engine(squat_poses(), baseline)

    assert [e.type for e in events] == [MovementType.SQUAT_COMPLETED]


def test_resetting_keeps_the_session_tally(baseline):
    """Recalibrating mid-session should not wipe what the user has done."""
    engine, _ = run_engine(squat_poses(), baseline)

    engine.reset()

    assert engine.counts["SQUAT_COMPLETED"] == 1


# ---- end to end ----------------------------------------------------------


def test_a_squat_becomes_a_down_arrow(baseline):
    """Poses in, keystroke out, through the real engine, profile and dispatcher."""
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter, enabled=True)

    run_engine(squat_poses(), baseline, dispatcher=dispatcher)

    assert adapter.presses == [GameKey.DOWN]


def test_a_side_step_becomes_an_arrow_key(baseline):
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter, enabled=True)

    poses = [
        side_pose(offset, foot_ratio=1.0, timestamp=i / FPS)
        for i, offset in enumerate(step_profile())
    ]
    # The engine defaults to lean mode, so a step has to be asked for.
    run_engine(poses, baseline, engine=MovementEngine(sideways="step"), dispatcher=dispatcher)

    assert adapter.presses == [GameKey.RIGHT]


def test_a_jump_becomes_an_up_arrow(baseline):
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter, enabled=True)

    poses = [
        jump_pose(lift, foot_ratio=1.0, timestamp=i / FPS)
        for i, lift in enumerate(jump_profile())
    ]
    run_engine(poses, baseline, dispatcher=dispatcher)

    assert adapter.presses == [GameKey.UP]


def test_a_rejected_movement_sends_nothing(baseline):
    """The validation upstream is what protects the game from noise: a shallow
    dip never becomes an event, so it can never become a keystroke."""
    adapter = RecordingAdapter()
    dispatcher = InputDispatcher(SUBWAY_SURFERS, adapter, enabled=True)

    poses = [
        build_pose(hip_y=depth * STANDING_HIP_HEIGHT, knee_x_offset=depth * 0.7, timestamp=i / FPS)
        for i, depth in enumerate(squat_profile(depth=0.15))
    ]
    run_engine(poses, baseline, dispatcher=dispatcher)

    assert adapter.presses == []
