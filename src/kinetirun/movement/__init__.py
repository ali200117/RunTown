"""Movement engine.

Turns poses into completed-movement events. Knows about the body and about
time; knows nothing about games, keyboards or MediaPipe.
"""

from kinetirun.movement.arm import ArmDetector, ArmState
from kinetirun.movement.config import (
    ArmConfig,
    JumpConfig,
    LeanConfig,
    SideConfig,
    SquatConfig,
)
from kinetirun.movement.engine import MovementEngine
from kinetirun.movement.events import MovementEvent, MovementType
from kinetirun.movement.jump import JumpDetector, JumpState
from kinetirun.movement.lean import LeanDetector, LeanState
from kinetirun.movement.side import SideDetector, SideState
from kinetirun.movement.squat import SquatDetector, SquatState

__all__ = [
    "MovementEngine",
    "MovementEvent",
    "MovementType",
    "ArmConfig",
    "ArmDetector",
    "ArmState",
    "JumpConfig",
    "JumpDetector",
    "JumpState",
    "LeanConfig",
    "LeanDetector",
    "LeanState",
    "SideConfig",
    "SideDetector",
    "SideState",
    "SquatConfig",
    "SquatDetector",
    "SquatState",
]
