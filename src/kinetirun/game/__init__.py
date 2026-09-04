"""Game layer.

Maps generic movement events onto the keys a specific game expects. Depends on
the movement layer; nothing in the movement layer depends on this.
"""

from kinetirun.game.profile import SUBWAY_SURFERS, GameKey, GameProfile

__all__ = ["GameKey", "GameProfile", "SUBWAY_SURFERS"]
