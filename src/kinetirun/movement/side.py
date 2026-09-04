"""Side-step detection.

    CENTER -> MOVING -> AT_TARGET -> RETURNING -> (event) -> CENTER

    Any state can fall out to WAIT_FOR_NEUTRAL, which returns to CENTER only
    once the user is standing in the middle again.

One detector handles both directions rather than two mirrored copies: the
logic is identical and only the sign of the offset differs, so duplicating it
would just mean fixing every bug twice.

The signal is hip displacement in image space. World space cannot see this at
all - it is anchored to the hips, so it moves with the user and reports nothing
when the whole body travels sideways.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from kinetirun.calibration import Baseline
from kinetirun.movement.config import SideConfig
from kinetirun.movement.events import MovementEvent, MovementType
from kinetirun.tracking import BodyPose, MotionHistory


class SideState(Enum):
    CENTER = "CENTER"
    MOVING = "MOVING"
    AT_TARGET = "AT_TARGET"
    RETURNING = "RETURNING"
    WAIT_FOR_NEUTRAL = "WAIT_FOR_NEUTRAL"


class SideDetector:
    """Detects complete side steps, left and right."""

    def __init__(self, config: Optional[SideConfig] = None) -> None:
        self.config = config or SideConfig()
        self._state = SideState.CENTER
        self._started_at: Optional[float] = None
        self._direction = 0          # -1 left, +1 right, 0 none in progress
        self._peak_offset = 0.0      # signed
        self._peak_foot_travel = 0.0
        self._rejection: Optional[str] = None

    # ---- driving --------------------------------------------------------

    def update(
        self,
        pose: Optional[BodyPose],
        baseline: Baseline,
        history: MotionHistory,
    ) -> Optional[MovementEvent]:
        """Feed one frame. Returns an event only on the frame that completes a step."""
        config = self.config

        if pose is None or not pose.is_reliable(config.min_visibility):
            self._abandon("lower body not tracked")
            return None

        offset = baseline.lateral_offset(pose)
        rate = history.velocity(baseline.lateral_offset, config.velocity_window)
        if rate is None:
            return None  # not enough history yet

        if self._started_at is not None:
            # Track the peak in the direction of travel, so a wobble back
            # toward centre cannot inflate the recorded distance.
            if abs(offset) > abs(self._peak_offset):
                self._peak_offset = offset
            self._peak_foot_travel = max(
                self._peak_foot_travel, abs(baseline.foot_offset(pose))
            )

            if pose.timestamp - self._started_at > config.max_duration:
                self._abandon("took too long")
                return None

        if self._state is SideState.CENTER:
            self._on_center(pose, offset, rate)
        elif self._state is SideState.MOVING:
            self._on_moving(offset, rate)
        elif self._state is SideState.AT_TARGET:
            self._on_at_target(rate)
        elif self._state is SideState.RETURNING:
            return self._on_returning(pose, offset)
        elif self._state is SideState.WAIT_FOR_NEUTRAL:
            self._on_wait(offset)

        return None

    def reset(self) -> None:
        self._state = SideState.CENTER
        self._rejection = None
        self._clear_attempt()

    # ---- states ---------------------------------------------------------

    def _on_center(self, pose: BodyPose, offset: float, rate: float) -> None:
        """Standing in the middle. A step starts on deliberate sideways speed."""
        if abs(offset) <= self.config.neutral_offset:
            return
        if abs(rate) < self.config.start_rate:
            return
        # Only count movement heading away from centre, not a drift back.
        if (offset > 0) != (rate > 0):
            return

        self._state = SideState.MOVING
        self._started_at = pose.timestamp
        self._direction = 1 if offset > 0 else -1
        self._peak_offset = offset
        self._peak_foot_travel = 0.0

    def _on_moving(self, offset: float, rate: float) -> None:
        """Travelling outward. Ends when the outward speed dies away."""
        if abs(rate) > self.config.settle_rate and (rate > 0) == (self._direction > 0):
            return  # still moving out

        if abs(self._peak_offset) >= self.config.min_offset:
            self._state = SideState.AT_TARGET
        else:
            # Stopped short. Requiring a return to centre stops a series of
            # small sways from eventually being accepted.
            self._rejection = (
                f"too small: {abs(self._peak_offset):.2f} < "
                f"{self.config.min_offset:.2f} shoulder widths"
            )
            self._state = SideState.WAIT_FOR_NEUTRAL

    def _on_at_target(self, rate: float) -> None:
        """Out at the target. Waiting for a move back toward centre."""
        heading_back = -self._direction
        if abs(rate) > self.config.return_rate and (rate > 0) == (heading_back > 0):
            self._state = SideState.RETURNING

    def _on_returning(self, pose: BodyPose, offset: float) -> Optional[MovementEvent]:
        """Coming back. Reaching centre completes the movement."""
        if abs(offset) > self.config.neutral_offset:
            return None

        duration = pose.timestamp - (self._started_at or pose.timestamp)

        # The two gates that make this a real step rather than a lean or a
        # twitch: it took a plausible amount of time, and the feet actually
        # moved. Hip displacement alone is fooled by tilting the upper body.
        valid = (
            duration >= self.config.min_duration
            and self._peak_foot_travel >= self.config.min_foot_travel
        )

        if not valid:
            self._rejection = (
                f"too fast: {duration:.2f}s < {self.config.min_duration:.2f}s"
                if duration < self.config.min_duration
                else f"feet did not move: {self._peak_foot_travel:.2f} < "
                     f"{self.config.min_foot_travel:.2f}"
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
                displacement=abs(self._peak_offset),
                quality=self._quality(duration),
            )

        self._state = SideState.CENTER
        self._clear_attempt()
        return event

    def _on_wait(self, offset: float) -> None:
        if abs(offset) <= self.config.neutral_offset:
            self._state = SideState.CENTER
            self._clear_attempt()

    # ---- helpers --------------------------------------------------------

    def _abandon(self, reason: str) -> None:
        if self._state is not SideState.CENTER:
            self._rejection = reason
            self._state = SideState.WAIT_FOR_NEUTRAL
        self._clear_attempt()

    def _clear_attempt(self) -> None:
        self._started_at = None
        self._direction = 0
        self._peak_offset = 0.0
        self._peak_foot_travel = 0.0

    def _quality(self, duration: float) -> float:
        distance_score = min(1.0, abs(self._peak_offset) / self.config.target_offset)
        pace_score = min(1.0, duration / (self.config.min_duration * 2))
        return round(0.75 * distance_score + 0.25 * pace_score, 3)

    # ---- inspection -----------------------------------------------------

    @property
    def state(self) -> SideState:
        return self._state

    @property
    def last_rejection(self) -> Optional[str]:
        """Why the most recent attempt was not accepted, if any."""
        return self._rejection

    @property
    def direction(self) -> int:
        """-1 left, +1 right, 0 when nothing is in progress."""
        return self._direction

    @property
    def progress(self) -> float:
        if self._state is SideState.MOVING:
            return 0.5 * min(1.0, abs(self._peak_offset) / self.config.min_offset)
        if self._state is SideState.AT_TARGET:
            return 0.5
        if self._state is SideState.RETURNING:
            return 0.75
        return 0.0
