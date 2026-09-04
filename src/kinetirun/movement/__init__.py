"""Movement engine.

Turns poses into completed-movement events. Knows about the body and about
time; knows nothing about games, keyboards or MediaPipe.
"""

from kinetirun.movement.config import JumpConfig, SideConfig, SquatConfig
from kinetirun.movement.events import MovementEvent, MovementType
from kinetirun.movement.jump import JumpDetector, JumpState
from kinetirun.movement.side import SideDetector, SideState
from kinetirun.movement.squat import SquatDetector, SquatState

__all__ = [
    "MovementEvent",
    "MovementType",
    "JumpConfig",
    "JumpDetector",
    "JumpState",
    "SideConfig",
    "SideDetector",
    "SideState",
    "SquatConfig",
    "SquatDetector",
    "SquatState",
]
