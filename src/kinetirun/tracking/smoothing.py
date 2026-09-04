"""Landmark smoothing.

Pose estimation jitters: a joint that is physically still moves a few
millimetres frame to frame. Velocity is a derivative, so that jitter gets
amplified - a still hip can appear to be moving at a real speed.

Kept as its own stage rather than inside each detector: raw pose -> smoother ->
stable pose -> movement engine. Otherwise every detector reimplements it
slightly differently and the whole system becomes impossible to tune.
"""

from __future__ import annotations

from typing import Optional

from kinetirun.tracking.body_pose import BodyPose, Point3D


class PoseSmoother:
    """Exponential moving average over landmark positions.

        smoothed = alpha * new + (1 - alpha) * previous

    Every smoother trades jitter against lag, and there is no setting that
    avoids the trade. Low alpha gives a rock-steady skeleton that reports a
    jump a beat after it happened; high alpha reacts instantly and shivers.

    0.5 is a deliberate compromise: at 30 FPS a step change is ~97% applied
    after 5 frames, about 170 ms. If jump detection later feels late, raise it
    rather than compensating inside the detector.

    Visibility is passed through unsmoothed. It is a confidence value, not a
    position, and averaging it would let a few confident frames prop up a joint
    the model has actually lost.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._previous: Optional[BodyPose] = None

    def smooth(self, pose: Optional[BodyPose]) -> Optional[BodyPose]:
        """Return a smoothed pose, or None when tracking is lost.

        Losing the pose clears the history. Blending a new pose against one
        from before the user walked out of frame would drag the skeleton across
        the screen and produce a large fake velocity.
        """
        if pose is None:
            self._previous = None
            return None

        if self._previous is None:
            self._previous = pose
            return pose

        smoothed = BodyPose(
            timestamp=pose.timestamp,
            world=self._blend(self._previous.world, pose.world),
            image=self._blend(self._previous.image, pose.image),
        )
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None

    def _blend(
        self, previous: tuple[Point3D, ...], current: tuple[Point3D, ...]
    ) -> tuple[Point3D, ...]:
        a = self.alpha
        b = 1.0 - a
        return tuple(
            Point3D(
                x=a * new.x + b * old.x,
                y=a * new.y + b * old.y,
                z=a * new.z + b * old.z,
                visibility=new.visibility,
            )
            for old, new in zip(previous, current)
        )
