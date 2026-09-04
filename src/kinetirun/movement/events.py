"""Movement events.

The output of the movement engine, and the boundary the game layer sits behind.
An event says WHAT the body did - never which key to press. SubwayProfile in
Phase 11 maps SQUAT_COMPLETED to the down arrow; the squat detector has no idea
that a down arrow exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MovementType(Enum):
    SQUAT_COMPLETED = "SQUAT_COMPLETED"
    SIDE_LEFT_COMPLETED = "SIDE_LEFT_COMPLETED"
    SIDE_RIGHT_COMPLETED = "SIDE_RIGHT_COMPLETED"
    JUMP_COMPLETED = "JUMP_COMPLETED"


@dataclass(frozen=True)
class MovementEvent:
    """A completed physical movement.

    The metadata beyond `type` is not needed to play a game, but it is what
    makes the fitness side of this project possible later: depth, duration and
    quality are exactly the numbers a session summary would report.
    """

    type: MovementType
    timestamp: float      # when the movement completed, in perf_counter seconds
    duration: float       # seconds from start to completion
    displacement: float   # peak movement, in body-relative units
    quality: float        # 0..1, how well the movement was executed
