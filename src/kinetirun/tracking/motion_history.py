"""A short, time-indexed window of recent poses.

A movement is a physical action over time, so a detector needs to ask
questions the current frame cannot answer: where were the hips 300 ms ago, how
fast are they moving, how deep did this squat get, when did the descent start.

Everything here is indexed by TIME, never by frame count. Frame rate on this
setup varies with the lighting - "15 frames ago" is 0.5 s at 30 FPS and 1.5 s
at 10 FPS. Thresholds tuned against frame counts silently break when the room
gets darker; thresholds in seconds do not.

Only measurements are kept, never images.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

from kinetirun.tracking.body_pose import BodyPose

# A scalar read off a pose: hip drop, lateral offset, knee angle, and so on.
Extract = Callable[[BodyPose], float]


class MotionHistory:
    """The last `duration` seconds of poses."""

    def __init__(self, duration: float = 2.0) -> None:
        """
        Args:
            duration: seconds of history to keep. 2 s comfortably covers a
                squat or a jump; keeping more just costs memory and lets stale
                data influence decisions.
        """
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        self.duration = duration
        self._poses: Deque[BodyPose] = deque()

    # ---- filling --------------------------------------------------------

    def append(self, pose: Optional[BodyPose]) -> None:
        """Add a pose and drop anything older than the window.

        None means tracking was lost. The history is cleared rather than left
        to age out: a detector must not compute velocity across a gap where the
        user was absent, because the two endpoints are unrelated.
        """
        if pose is None:
            self.clear()
            return

        self._poses.append(pose)
        cutoff = pose.timestamp - self.duration
        while self._poses and self._poses[0].timestamp < cutoff:
            self._poses.popleft()

    def clear(self) -> None:
        self._poses.clear()

    # ---- inspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._poses)

    @property
    def latest(self) -> Optional[BodyPose]:
        return self._poses[-1] if self._poses else None

    @property
    def span(self) -> float:
        """Seconds actually covered - less than `duration` while filling up."""
        if len(self._poses) < 2:
            return 0.0
        return self._poses[-1].timestamp - self._poses[0].timestamp

    def since(self, seconds: float) -> List[BodyPose]:
        """Poses from the last `seconds`."""
        if not self._poses:
            return []
        cutoff = self._poses[-1].timestamp - seconds
        return [pose for pose in self._poses if pose.timestamp >= cutoff]

    # ---- questions detectors ask ----------------------------------------

    def value_at(self, seconds_ago: float, extract: Extract) -> Optional[float]:
        """A measurement as it was `seconds_ago`, linearly interpolated.

        Interpolation is what makes the result frame-rate independent: the
        answer to "where were the hips 300 ms ago" should not depend on whether
        a frame happened to land exactly then.

        Returns None if the history does not reach back that far.
        """
        if len(self._poses) < 2:
            return None

        target = self._poses[-1].timestamp - seconds_ago
        if target < self._poses[0].timestamp:
            return None

        for older, newer in zip(self._poses, list(self._poses)[1:]):
            if older.timestamp <= target <= newer.timestamp:
                gap = newer.timestamp - older.timestamp
                if gap <= 0:
                    return extract(newer)
                weight = (target - older.timestamp) / gap
                return extract(older) + weight * (extract(newer) - extract(older))

        return extract(self._poses[-1])

    def displacement(self, seconds: float, extract: Extract) -> Optional[float]:
        """How much a measurement has changed over the last `seconds`.

        Positive means the value grew. Remember that image y grows downward, so
        a rising body produces a negative vertical displacement.
        """
        past = self.value_at(seconds, extract)
        if past is None or not self._poses:
            return None
        return extract(self._poses[-1]) - past

    def velocity(self, extract: Extract, window: float = 0.15) -> Optional[float]:
        """Rate of change per second, averaged over a short window.

        The window is a deliberate compromise: too short and this is dominated
        by jitter even after smoothing, too long and a direction change is
        reported late. 150 ms is roughly 4-5 frames at 30 FPS.
        """
        change = self.displacement(window, extract)
        if change is None:
            return None
        return change / window

    def extreme(
        self, extract: Extract, seconds: Optional[float] = None, largest: bool = True
    ) -> Optional[Tuple[float, float]]:
        """The largest (or smallest) value in the window, as (value, timestamp).

        The timestamp is the useful half: it answers "when did this squat reach
        the bottom", which is what separates a completed movement from one
        still in progress.
        """
        poses = self._poses if seconds is None else self.since(seconds)
        if not poses:
            return None

        chooser = max if largest else min
        best = chooser(poses, key=extract)
        return extract(best), best.timestamp

    def time_since(self, timestamp: float) -> Optional[float]:
        """Seconds elapsed between `timestamp` and the newest pose."""
        if not self._poses:
            return None
        return self._poses[-1].timestamp - timestamp
