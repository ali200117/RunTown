"""Sideways lean detection.

    UPRIGHT -> LEANING -> AT_TARGET -> RETURNING -> (event) -> UPRIGHT

    Any state can fall out to WAIT_FOR_NEUTRAL, which returns to UPRIGHT only
    once the user is standing straight again.

The alternative to SideDetector, for rooms with no space to step sideways. The
gesture is a whole-body bow to the side rather than a step: the feet stay put
and the torso tilts.

It emits exactly the same SIDE_LEFT_COMPLETED / SIDE_RIGHT_COMPLETED events, so
nothing downstream changes - the game profile and the input adapter cannot tell
which one produced the event. That is the seam working as intended.

Note the deliberate tradeoff: SideDetector rejects leaning as cheating, and
this detector is built on leaning. Swapping in this one removes the guarantee
that the movement was a full-body displacement. The remaining defences are the
angle threshold, the speed threshold, the required return to upright, and the
duration limits.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from kinetirun.calibration import Baseline
from kinetirun.movement.config import LeanConfig
from kinetirun.movement.events import MovementEvent, MovementType
from kinetirun.tracking import BodyPose, MotionHistory


class LeanState(Enum):
    UPRIGHT = "UPRIGHT"
    LEANING = "LEANING"
    AT_TARGET = "AT_TARGET"
    RETURNING = "RETURNING"
    WAIT_FOR_NEUTRAL = "WAIT_FOR_NEUTRAL"


class LeanDetector:
    """Detects complete sideways leans, left and right."""

    def __init__(self, config: Optional[LeanConfig] = None) -> None:
        self.config = config or LeanConfig()
        self._state = LeanState.UPRIGHT
        self._started_at: Optional[float] = None
        self._direction = 0        # -1 left, +1 right, 0 none in progress
        self._peak_angle = 0.0     # signed degrees
        self._rejection: Optional[str] = None

    # ---- driving --------------------------------------------------------

    def update(
        self,
        pose: Optional[BodyPose],
        baseline: Baseline,
        history: MotionHistory,
    ) -> Optional[MovementEvent]:
        """Feed one frame. Returns an event only on the frame that completes a lean."""
        config = self.config

        # Only the torso matters here, so the full lower-body check would be
        # too strict - but shoulders and hips must be trustworthy, since the
        # angle is computed entirely from them.
        if pose is None or not pose.is_reliable(config.min_visibility):
            self._abandon("body not tracked")
            return None

        angle = baseline.lean_angle(pose)
        rate = history.velocity(baseline.lean_angle, config.velocity_window)
        if rate is None:
            return None  # not enough history yet

        if self._started_at is not None:
            if abs(angle) > abs(self._peak_angle):
                self._peak_angle = angle

            if pose.timestamp - self._started_at > config.max_duration:
                self._abandon("took too long")
                return None

        if self._state is LeanState.UPRIGHT:
            self._on_upright(pose, angle, rate)
        elif self._state is LeanState.LEANING:
            self._on_leaning(rate)
        elif self._state is LeanState.AT_TARGET:
            self._on_at_target(rate)
        elif self._state is LeanState.RETURNING:
            return self._on_returning(pose, angle)
        elif self._state is LeanState.WAIT_FOR_NEUTRAL:
            self._on_wait(angle)

        return None

    def reset(self) -> None:
        self._state = LeanState.UPRIGHT
        self._rejection = None
        self._clear_attempt()

    # ---- states ---------------------------------------------------------

    def _on_upright(self, pose: BodyPose, angle: float, rate: float) -> None:
        """Standing straight. A lean starts on deliberate tilting speed."""
        if abs(angle) <= self.config.neutral_angle:
            return
        if abs(rate) < self.config.start_rate:
            return
        # Only count tilting away from upright, not straightening back up.
        if (angle > 0) != (rate > 0):
            return

        self._state = LeanState.LEANING
        self._started_at = pose.timestamp
        self._direction = 1 if angle > 0 else -1
        self._peak_angle = angle

    def _on_leaning(self, rate: float) -> None:
        """Tilting over. Ends when the tilting speed dies away."""
        if abs(rate) > self.config.settle_rate and (rate > 0) == (self._direction > 0):
            return  # still going over

        if abs(self._peak_angle) >= self.config.min_angle:
            self._state = LeanState.AT_TARGET
        else:
            self._rejection = (
                f"not far enough: {abs(self._peak_angle):.0f} < "
                f"{self.config.min_angle:.0f} deg"
            )
            self._state = LeanState.WAIT_FOR_NEUTRAL

    def _on_at_target(self, rate: float) -> None:
        """Held over. Waiting for the return toward upright."""
        heading_back = -self._direction
        if abs(rate) > self.config.return_rate and (rate > 0) == (heading_back > 0):
            self._state = LeanState.RETURNING

    def _on_returning(self, pose: BodyPose, angle: float) -> Optional[MovementEvent]:
        """Straightening up. Reaching upright completes the movement."""
        if abs(angle) > self.config.neutral_angle:
            return None

        duration = pose.timestamp - (self._started_at or pose.timestamp)
        valid = duration >= self.config.min_duration

        if not valid:
            self._rejection = (
                f"too fast: {duration:.2f}s < {self.config.min_duration:.2f}s"
            )
        else:
            self._rejection = None

        event = None
        if valid:
            event = MovementEvent(
                type=(
                    MovementType.SIDE_RIGHT_COMPLETED
                    if self._direction > 0
                    else MovementType.SIDE_LEFT_COMPLETED
                ),
                timestamp=pose.timestamp,
                duration=duration,
                displacement=abs(self._peak_angle),
                quality=self._quality(duration),
            )

        self._state = LeanState.UPRIGHT
        self._clear_attempt()
        return event

    def _on_wait(self, angle: float) -> None:
        if abs(angle) <= self.config.neutral_angle:
            self._state = LeanState.UPRIGHT
            self._clear_attempt()

    # ---- helpers --------------------------------------------------------

    def _abandon(self, reason: str) -> None:
        if self._state is not LeanState.UPRIGHT:
            self._rejection = reason
            self._state = LeanState.WAIT_FOR_NEUTRAL
        self._clear_attempt()

    def _clear_attempt(self) -> None:
        self._started_at = None
        self._direction = 0
        self._peak_angle = 0.0

    def _quality(self, duration: float) -> float:
        angle_score = min(1.0, abs(self._peak_angle) / self.config.target_angle)
        pace_score = min(1.0, duration / (self.config.min_duration * 2))
        return round(0.75 * angle_score + 0.25 * pace_score, 3)

    # ---- inspection -----------------------------------------------------

    @property
    def state(self) -> LeanState:
        return self._state

    @property
    def last_rejection(self) -> Optional[str]:
        return self._rejection

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def progress(self) -> float:
        if self._state is LeanState.LEANING:
            return 0.5 * min(1.0, abs(self._peak_angle) / self.config.min_angle)
        if self._state is LeanState.AT_TARGET:
            return 0.5
        if self._state is LeanState.RETURNING:
            return 0.75
        return 0.0
