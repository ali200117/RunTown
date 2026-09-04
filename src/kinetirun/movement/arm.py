"""Arm swipe detection.

    REST -> SWIPING -> AT_TARGET -> RETURNING -> (event) -> REST

    Any state can fall out to WAIT_FOR_NEUTRAL, which returns to REST only once
    the arms are back down.

The third option for sideways input, alongside stepping and leaning:

    RAISE YOUR RIGHT ARM out to the side, then lower it   -> RIGHT
    RAISE YOUR LEFT ARM out to the side, then lower it    -> LEFT

The signal is the arm's ELEVATION ANGLE, not how far the hand travelled
sideways. That distinction matters: swinging a hanging arm out to horizontal
moves the wrist only about half a shoulder width across the frame - small
enough to be confused with fidgeting - but a full 90 degrees of elevation,
which nothing else does by accident.

Two practical advantages over the other two, which is why it exists: it is
quick enough to react with in a fast game, and it only needs the SHOULDERS and
WRISTS to be visible. Step and lean both depend on the lower body, which is
often out of frame in a small room.

Like the others it emits SIDE_LEFT_COMPLETED / SIDE_RIGHT_COMPLETED, so the
game profile and input adapter are untouched by the choice.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from kinetirun.calibration import Baseline
from kinetirun.movement.config import ArmConfig
from kinetirun.movement.events import MovementEvent, MovementType
from kinetirun.tracking import BodyPose, MotionHistory
from kinetirun.vision.landmarks import Landmark


class ArmState(Enum):
    REST = "REST"
    SWIPING = "SWIPING"
    AT_TARGET = "AT_TARGET"
    RETURNING = "RETURNING"
    WAIT_FOR_NEUTRAL = "WAIT_FOR_NEUTRAL"


class ArmDetector:
    """Detects complete arm swipes, left and right."""

    def __init__(self, config: Optional[ArmConfig] = None) -> None:
        self.config = config or ArmConfig()
        self._state = ArmState.REST
        self._started_at: Optional[float] = None
        self._direction = 0        # -1 left, +1 right, 0 none in progress
        self._peak_angle = 0.0     # signed degrees; + is the right arm
        self._rejection: Optional[str] = None

    # ---- driving --------------------------------------------------------

    def update(
        self,
        pose: Optional[BodyPose],
        baseline: Baseline,
        history: MotionHistory,
    ) -> Optional[MovementEvent]:
        """Feed one frame. Returns an event only on the frame that completes a swipe."""
        config = self.config

        # Only the upper body is checked. Requiring reliable ankles here would
        # throw away the main advantage of this mode.
        if pose is None or not self._upper_body_tracked(pose):
            self._abandon("arms not tracked")
            return None

        angle = baseline.arm_reach(pose)
        rate = history.velocity(baseline.arm_reach, config.velocity_window)
        if rate is None:
            return None  # not enough history yet

        if self._started_at is not None:
            if abs(angle) > abs(self._peak_angle):
                self._peak_angle = angle

            if pose.timestamp - self._started_at > config.max_duration:
                self._abandon("took too long")
                return None

        if self._state is ArmState.REST:
            self._on_rest(pose, angle, rate)
        elif self._state is ArmState.SWIPING:
            self._on_swiping(rate)
        elif self._state is ArmState.AT_TARGET:
            self._on_at_target(rate)
        elif self._state is ArmState.RETURNING:
            return self._on_returning(pose, angle)
        elif self._state is ArmState.WAIT_FOR_NEUTRAL:
            self._on_wait(angle)

        return None

    def reset(self) -> None:
        self._state = ArmState.REST
        self._rejection = None
        self._clear_attempt()

    # ---- states ---------------------------------------------------------

    def _upper_body_tracked(self, pose: BodyPose) -> bool:
        return (
            pose.visibility_of(
                Landmark.LEFT_WRIST,
                Landmark.RIGHT_WRIST,
                Landmark.LEFT_SHOULDER,
                Landmark.RIGHT_SHOULDER,
            )
            >= self.config.min_visibility
        )

    def _on_rest(self, pose: BodyPose, angle: float, rate: float) -> None:
        """Arms down. A swipe starts when an arm is raised deliberately."""
        if abs(angle) <= self.config.neutral_angle:
            return
        if abs(rate) < self.config.start_rate:
            return
        # Only count an arm going UP, not one being lowered.
        if (angle > 0) != (rate > 0):
            return

        self._state = ArmState.SWIPING
        self._started_at = pose.timestamp
        self._direction = 1 if angle > 0 else -1
        self._peak_angle = angle

    def _on_swiping(self, rate: float) -> None:
        """Arm going up. Ends when the raising speed dies away."""
        if abs(rate) > self.config.settle_rate and (rate > 0) == (self._direction > 0):
            return  # still sweeping

        if abs(self._peak_angle) >= self.config.min_angle:
            self._state = ArmState.AT_TARGET
        else:
            self._rejection = (
                f"not raised enough: {abs(self._peak_angle):.0f} < "
                f"{self.config.min_angle:.0f} deg"
            )
            self._state = ArmState.WAIT_FOR_NEUTRAL

    def _on_at_target(self, rate: float) -> None:
        """Arm out. Waiting for it to come back."""
        heading_back = -self._direction
        if abs(rate) > self.config.return_rate and (rate > 0) == (heading_back > 0):
            self._state = ArmState.RETURNING

    def _on_returning(self, pose: BodyPose, angle: float) -> Optional[MovementEvent]:
        """Arm coming down. Reaching rest completes the movement."""
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

        self._state = ArmState.REST
        self._clear_attempt()
        return event

    def _on_wait(self, angle: float) -> None:
        if abs(angle) <= self.config.neutral_angle:
            self._state = ArmState.REST
            self._clear_attempt()

    # ---- helpers --------------------------------------------------------

    def _abandon(self, reason: str) -> None:
        if self._state is not ArmState.REST:
            self._rejection = reason
            self._state = ArmState.WAIT_FOR_NEUTRAL
        self._clear_attempt()

    def _clear_attempt(self) -> None:
        self._started_at = None
        self._direction = 0
        self._peak_angle = 0.0

    def _quality(self, duration: float) -> float:
        return round(min(1.0, abs(self._peak_angle) / self.config.target_angle), 3)

    # ---- inspection -----------------------------------------------------

    @property
    def state(self) -> ArmState:
        return self._state

    @property
    def last_rejection(self) -> Optional[str]:
        return self._rejection

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def progress(self) -> float:
        if self._state is ArmState.SWIPING:
            return 0.5 * min(1.0, abs(self._peak_angle) / self.config.min_angle)
        if self._state is ArmState.AT_TARGET:
            return 0.5
        if self._state is ArmState.RETURNING:
            return 0.75
        return 0.0
