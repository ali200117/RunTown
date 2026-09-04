"""Game profiles: movement events to game keys.

This is the seam that keeps KinetiRun from being a Subway Surfers program. The
movement engine emits SQUAT_COMPLETED; this layer decides that Subway Surfers
wants a down arrow for that. Neither side knows about the other.

A profile is data, not behaviour, so adding a game is a new mapping rather than
a new class hierarchy. If a game ever needs something a mapping cannot express
- a held key, a combination - that is the point to introduce a richer type,
not before.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from kinetirun.movement import MovementType


class GameKey(Enum):
    """Keys a profile can ask for.

    Deliberately abstract rather than raw key names: the input adapter decides
    what "UP" means to the operating system. Swapping PyAutoGUI for a gamepad
    emulator later would not touch a single profile.
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    SPACE = "space"


@dataclass(frozen=True)
class GameProfile:
    """Which key each movement should produce, for one game."""

    name: str
    mapping: Mapping[MovementType, GameKey]

    def key_for(self, movement: MovementType) -> Optional[GameKey]:
        """The key for a movement, or None if this game ignores it.

        None is a legitimate answer, not an error: a game that has no crouch
        should simply not react to squats.
        """
        return self.mapping.get(movement)


SUBWAY_SURFERS = GameProfile(
    name="Subway Surfers",
    mapping={
        MovementType.SIDE_LEFT_COMPLETED: GameKey.LEFT,
        MovementType.SIDE_RIGHT_COMPLETED: GameKey.RIGHT,
        MovementType.JUMP_COMPLETED: GameKey.UP,
        MovementType.SQUAT_COMPLETED: GameKey.DOWN,
    },
)
