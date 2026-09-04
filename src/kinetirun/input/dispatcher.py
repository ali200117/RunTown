"""Turning movement events into key presses.

The last link in the chain, and the narrowest: look the movement up in the
profile, hand the key to the adapter. All the intelligence lives upstream - by
the time an event reaches here it has already been validated by a state
machine, so there is nothing left to second-guess.
"""

from __future__ import annotations

from typing import List, Optional

from kinetirun.game.profile import GameKey, GameProfile
from kinetirun.input.adapters import InputAdapter
from kinetirun.movement import MovementEvent


class InputDispatcher:
    """Maps movement events onto key presses, when enabled."""

    def __init__(
        self,
        profile: GameProfile,
        adapter: InputAdapter,
        enabled: bool = False,
    ) -> None:
        """
        Args:
            enabled: OFF by default, on purpose. Key presses go to whichever
                window has focus, so a session that starts sending input
                without being asked would type into the user's editor. Turning
                it on is always a deliberate act.
        """
        self.profile = profile
        self.adapter = adapter
        self.enabled = enabled

    def dispatch(self, events: List[MovementEvent]) -> List[GameKey]:
        """Send input for each event. Returns the keys actually pressed."""
        if not self.enabled:
            return []

        pressed = []
        for event in events:
            key = self.profile.key_for(event.type)
            # An unmapped movement is normal, not an error: a game with no
            # crouch simply ignores squats.
            if key is None:
                continue
            self.adapter.press(key)
            pressed.append(key)
        return pressed

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    @property
    def status(self) -> str:
        return "ON" if self.enabled else "OFF"
