"""Jump detection.

    GROUNDED -> TAKEOFF -> AIRBORNE -> LANDING -> (event) -> GROUNDED

    Any state can fall out to WAIT_FOR_NEUTRAL, which returns to GROUNDED only
    once the user is standing at their normal height again.

A jump is the hardest of the three to detect honestly, because the
displacement is small - a decent jump moves the hips less far than a squat
does. What makes it unmistakable is SPEED: nothing else in normal standing
produces that much upward velocity. So the takeoff threshold does most of the
work here, where depth did it for the squat.

Everything is expressed as `lift`: upward displacement in shoulder widths,
positive up. That is the opposite sign from raw image coordinates, where y
grows downward, and the flip happens once here rather than in every comparison.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from kinetirun.calibration import Baseline
from kinetirun.movement.config import JumpConfig
from kinetirun.movement.events import MovementEvent, MovementType
from kinetirun.tracking import BodyPose, MotionHistory


class JumpState(Enum):
    GROUNDED = "GROUNDED"
    TAKEOFF = "TAKEOFF"
    AIRBORNE = "AIRBORNE"
    LANDING = "LANDING"
    WAIT_FOR_NEUTRAL = "WAIT_FOR_NEUTRAL"


class JumpDetector:
    """Detects complete jumps from a stream of poses."""

    def __init__(self, config: Optional[JumpConfig] = None) -> None:
        self.config = config or JumpConfig()
        self._state = JumpState.GROUNDED
        self._started_at: Optional[float] = None
        self._peak_lift = 0.0
        self._peak_foot_lift = 0.0
        self._rejection: Optional[str] = None

    # ---- driving --------------------------------------------------------

    def update(
        self,
        pose: Optional[BodyPose],
        baseline: Baseline,
        history: MotionHistory,
    ) -> Optional[MovementEvent]:
        """Feed one frame. Returns an event only on the frame that completes a jump."""
        config = self.config

        if pose is None or not pose.is_reliable(config.min_visibility):
            self._abandon("lower body not tracked")
            return None

        # One sign flip, applied once: upward is positive from here on.
        def lift_of(sample: BodyPose) -> float:
            return -baseline.vertical_offset(sample)

        lift = lift_of(pose)
        rate = history.velocity(lift_of, config.velocity_window)
        if rate is None:
            return None  # not enough history yet

        if self._started_at is not None:
            self._peak_lift = max(self._peak_lift, lift)
            self._peak_foot_lift = max(self._peak_foot_lift, baseline.foot_lift(pose))

            if pose.timestamp - self._started_at > config.max_duration:
                self._abandon("took too long")
                return None

        if self._state is JumpState.GROUNDED:
            self._on_grounded(pose, rate)
        elif self._state is JumpState.TAKEOFF:
            self._on_takeoff(rate)
        elif self._state is JumpState.AIRBORNE:
            self._on_airborne(rate)
        elif self._state is JumpState.LANDING:
            return self._on_landing(pose, lift)
        elif self._state is JumpState.WAIT_FOR_NEUTRAL:
            self._on_wait(lift)

        return None

    def reset(self) -> None:
        self._state = JumpState.GROUNDED
        self._rejection = None
        self._clear_attempt()

    # ---- states ---------------------------------------------------------

    def _on_grounded(self, pose: BodyPose, rate: float) -> None:
        """On the ground. Only a genuinely fast upward move starts a jump.

        Deliberately keyed on speed alone rather than height: at takeoff the
        body has barely moved yet, so waiting for displacement would mean
        starting the clock late and measuring a shorter jump than happened.
        """
        if rate < self.config.takeoff_rate:
            return

        self._state = JumpState.TAKEOFF
        self._started_at = pose.timestamp
        self._peak_lift = 0.0
        self._peak_foot_lift = 0.0

    def _on_takeoff(self, rate: float) -> None:
        """Rising. The rise ends at the apex, where upward speed dies away."""
        if rate > self.config.settle_rate:
            return  # still going up

        if self._peak_lift >= self.config.min_height:
            self._state = JumpState.AIRBORNE
        else:
            # A fast twitch that never got anywhere. Requiring a return to
            # neutral stops repeated bobbing from eventually being accepted.
            self._rejection = (
                f"too low: {self._peak_lift:.2f} < {self.config.min_height:.2f}"
            )
            self._state = JumpState.WAIT_FOR_NEUTRAL

    def _on_airborne(self, rate: float) -> None:
        """At the top. Waiting for the fall back down."""
        if rate < -self.config.landing_rate:
            self._state = JumpState.LANDING

    def _on_landing(self, pose: BodyPose, lift: float) -> Optional[MovementEvent]:
        """Coming down. Reaching normal standing height completes the jump."""
        if lift > self.config.neutral_band:
            return None

        duration = pose.timestamp - (self._started_at or pose.timestamp)

        # The final gates: a plausible airtime, and feet that actually left the
        # ground. Hip height alone is fooled by rising onto the toes.
        valid = (
            duration >= self.config.min_duration
            and self._peak_foot_lift >= self.config.min_foot_lift
        )

        if not valid:
            self._rejection = (
                f"too brief: {duration:.2f}s < {self.config.min_duration:.2f}s"
                if duration < self.config.min_duration
                else f"feet stayed down: {self._peak_foot_lift:.2f} < "
                     f"{self.config.min_foot_lift:.2f}"
            )
        else:
            self._rejection = None

        event = None
        if valid:
            event = MovementEvent(
                type=MovementType.JUMP_COMPLETED,
                timestamp=pose.timestamp,
                duration=duration,
                displacement=self._peak_lift,
                quality=self._quality(),
            )

        self._state = JumpState.GROUNDED
        self._clear_attempt()
        return event

    def _on_wait(self, lift: float) -> None:
        if abs(lift) <= self.config.neutral_band:
            self._state = JumpState.GROUNDED
            self._clear_attempt()

    # ---- helpers --------------------------------------------------------

    def _abandon(self, reason: str) -> None:
        if self._state is not JumpState.GROUNDED:
            self._rejection = reason
            self._state = JumpState.WAIT_FOR_NEUTRAL
        self._clear_attempt()

    def _clear_attempt(self) -> None:
        self._started_at = None
        self._peak_lift = 0.0
        self._peak_foot_lift = 0.0

    def _quality(self) -> float:
        """Height is the whole score here.

        Unlike the squat, there is no pace component: a jump cannot be
        performed too quickly, and airtime is already a consequence of height.
        """
        return round(min(1.0, self._peak_lift / self.config.target_height), 3)

    # ---- inspection -----------------------------------------------------

    @property
    def state(self) -> JumpState:
        return self._state

    @property
    def last_rejection(self) -> Optional[str]:
        """Why the most recent attempt was not accepted, if any."""
        return self._rejection

    @property
    def progress(self) -> float:
        if self._state is JumpState.TAKEOFF:
            return 0.5 * min(1.0, self._peak_lift / self.config.min_height)
        if self._state is JumpState.AIRBORNE:
            return 0.5
        if self._state is JumpState.LANDING:
            return 0.75
        return 0.0
