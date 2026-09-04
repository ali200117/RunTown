"""Squat detection.

    READY -> DESCENDING -> BOTTOM -> ASCENDING -> (event) -> READY

    Any state can fall out to WAIT_FOR_NEUTRAL, which returns to READY only
    once the user is standing again.

The point of the state machine is that no single frame can produce an event.
A squat only counts once the whole cycle has happened: a descent that was fast
enough to be intentional, deep enough to be real, bent the knees rather than
just the waist, took a plausible amount of time, and finished with a return to
standing.

WAIT_FOR_NEUTRAL is what removes the need for an arbitrary cooldown. After a
failed or completed attempt the user must be standing again before a new squat
can start, so bouncing in a half-squat produces nothing.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from kinetirun.calibration import Baseline
from kinetirun.movement.config import SquatConfig
from kinetirun.movement.events import MovementEvent, MovementType
from kinetirun.tracking import BodyPose, MotionHistory


class SquatState(Enum):
    READY = "READY"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"
    WAIT_FOR_NEUTRAL = "WAIT_FOR_NEUTRAL"


class SquatDetector:
    """Detects complete squats from a stream of poses."""

    def __init__(self, config: Optional[SquatConfig] = None) -> None:
        self.config = config or SquatConfig()
        self._state = SquatState.READY
        self._started_at: Optional[float] = None
        self._max_depth = 0.0
        self._min_knee_angle = 180.0
        # Why the last attempt failed. Tuning without this is guesswork: a
        # movement that "just does not register" could be too shallow, too
        # fast, or not tracked at all, and those need opposite fixes.
        self._rejection: Optional[str] = None

    # ---- driving --------------------------------------------------------

    def update(
        self,
        pose: Optional[BodyPose],
        baseline: Baseline,
        history: MotionHistory,
    ) -> Optional[MovementEvent]:
        """Feed one frame. Returns an event only on the frame that completes a squat."""
        config = self.config

        # No decision is made on data we do not trust. MediaPipe returns
        # coordinates for joints it cannot see, so without this the detector
        # would happily measure a fabricated leg.
        if pose is None or not pose.is_reliable(config.min_visibility):
            self._abandon("lower body not tracked")
            return None

        depth = baseline.hip_drop(pose)
        rate = history.velocity(baseline.hip_drop, config.velocity_window)
        if rate is None:
            return None  # not enough history yet

        if self._started_at is not None:
            self._max_depth = max(self._max_depth, depth)
            self._min_knee_angle = min(self._min_knee_angle, pose.mean_knee_angle)

            if pose.timestamp - self._started_at > config.max_duration:
                # Not a movement any more, just standing around bent over.
                self._abandon("took too long")
                return None

        if self._state is SquatState.READY:
            self._on_ready(pose, depth, rate)
        elif self._state is SquatState.DESCENDING:
            self._on_descending(rate)
        elif self._state is SquatState.BOTTOM:
            self._on_bottom(rate)
        elif self._state is SquatState.ASCENDING:
            return self._on_ascending(pose, depth, baseline)
        elif self._state is SquatState.WAIT_FOR_NEUTRAL:
            self._on_wait(depth)

        return None

    def reset(self) -> None:
        self._state = SquatState.READY
        self._rejection = None
        self._clear_attempt()

    # ---- states ---------------------------------------------------------

    def _on_ready(self, pose: BodyPose, depth: float, rate: float) -> None:
        """Standing. A descent starts when the hips move down deliberately."""
        if depth > self.config.neutral_depth and rate > self.config.start_rate:
            self._state = SquatState.DESCENDING
            self._started_at = pose.timestamp
            self._max_depth = depth
            self._min_knee_angle = pose.mean_knee_angle

    def _on_descending(self, rate: float) -> None:
        """Going down. The descent ends when downward speed dies away."""
        if rate > self.config.settle_rate:
            return  # still descending

        if self._max_depth >= self.config.min_depth:
            self._state = SquatState.BOTTOM
        else:
            # Stopped short. Requiring a return to neutral is what stops a
            # series of shallow bobs from eventually being accepted.
            self._rejection = (
                f"too shallow: {self._max_depth:.2f} < {self.config.min_depth:.2f}"
            )
            self._state = SquatState.WAIT_FOR_NEUTRAL

    def _on_bottom(self, rate: float) -> None:
        """At the bottom. Waiting for a real upward move."""
        if rate < -self.config.rise_rate:
            self._state = SquatState.ASCENDING

    def _on_ascending(
        self, pose: BodyPose, depth: float, baseline: Baseline
    ) -> Optional[MovementEvent]:
        """Coming back up. Reaching neutral completes the movement."""
        if depth > self.config.neutral_depth:
            return None

        duration = pose.timestamp - (self._started_at or pose.timestamp)
        knee_bend = baseline.knee_angle - self._min_knee_angle

        # Final gate. Both checks catch a movement that looked right on the way
        # down: too quick to be anything but a bounce, or hips lowered by
        # bending at the waist while the legs stayed straight.
        valid = (
            duration >= self.config.min_duration
            and knee_bend >= self.config.min_knee_bend
        )

        if not valid:
            self._rejection = (
                f"too fast: {duration:.2f}s < {self.config.min_duration:.2f}s"
                if duration < self.config.min_duration
                else f"knees barely bent: {knee_bend:.0f} < {self.config.min_knee_bend:.0f} deg"
            )

        event = None
        if valid:
            event = MovementEvent(
                type=MovementType.SQUAT_COMPLETED,
                timestamp=pose.timestamp,
                duration=duration,
                displacement=self._max_depth,
                quality=self._quality(duration),
            )

        if valid:
            self._rejection = None
        self._state = SquatState.READY
        self._clear_attempt()
        return event

    def _on_wait(self, depth: float) -> None:
        """Re-arm only once the user is properly standing again."""
        if depth <= self.config.neutral_depth:
            self._state = SquatState.READY
            self._clear_attempt()

    # ---- helpers --------------------------------------------------------

    def _abandon(self, reason: str) -> None:
        """Give up on the current attempt without emitting anything."""
        if self._state is not SquatState.READY:
            self._rejection = reason
            self._state = SquatState.WAIT_FOR_NEUTRAL
        self._clear_attempt()

    def _clear_attempt(self) -> None:
        self._started_at = None
        self._max_depth = 0.0
        self._min_knee_angle = 180.0

    def _quality(self, duration: float) -> float:
        """0..1 score for how well the squat was performed.

        Depth is most of it. Duration contributes a smaller penalty for rushing
        - a squat right at the minimum duration is legal but sloppy.
        """
        depth_score = min(1.0, self._max_depth / self.config.target_depth)
        pace_score = min(1.0, duration / (self.config.min_duration * 2))
        return round(0.75 * depth_score + 0.25 * pace_score, 3)

    # ---- inspection -----------------------------------------------------

    @property
    def state(self) -> SquatState:
        return self._state

    @property
    def last_rejection(self) -> Optional[str]:
        """Why the most recent attempt was not accepted, if any."""
        return self._rejection

    @property
    def depth(self) -> float:
        """Deepest point reached in the current attempt."""
        return self._max_depth

    @property
    def progress(self) -> float:
        """0..1 through the current movement, for the debug overlay."""
        if self._state is SquatState.DESCENDING:
            return 0.5 * min(1.0, self._max_depth / self.config.min_depth)
        if self._state is SquatState.BOTTOM:
            return 0.5
        if self._state is SquatState.ASCENDING:
            return 0.75
        return 0.0
