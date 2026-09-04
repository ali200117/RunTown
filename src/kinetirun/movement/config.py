"""Movement thresholds.

Kept in one place, as data, so tuning does not mean hunting for magic numbers
scattered through detector logic. Every distance here is body-relative - a
fraction of standing hip height or of shoulder width - so the same values work
for different bodies at different camera distances.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SquatConfig:
    """Tuning for SquatDetector.

    Depth is `hip_drop`: 0.0 standing, 1.0 hips at ankle height. A comfortable
    squat lands somewhere around 0.30-0.40.
    """

    min_depth: float = 0.28
    """How deep the hips must go for the squat to count. The single most
    important number here: raise it to force harder squats."""

    target_depth: float = 0.38
    """Depth that scores full quality. Above min_depth, so a squat that merely
    qualifies is not rewarded the same as a good one."""

    neutral_depth: float = 0.10
    """At or below this the user counts as standing. Returning here is what
    completes a squat, and what re-arms the detector afterwards."""

    start_rate: float = 0.25
    """Downward speed, in depth per second, that starts a descent. Filters out
    slow drift and residual jitter."""

    settle_rate: float = 0.12
    """Below this speed the descent is considered finished - the bottom."""

    rise_rate: float = 0.15
    """Upward speed that marks the start of the ascent."""

    min_duration: float = 0.40
    """Faster than this is a bounce, not a squat. Guards against the cheating
    the whole project is meant to prevent."""

    max_duration: float = 4.0
    """Slower than this is not a movement, it is standing around bent over."""

    min_knee_bend: float = 15.0
    """Degrees the knees must bend below the calibrated standing angle. A
    second, independent signal: bending at the waist lowers the hips without
    bending the knees, and should not count as a squat."""

    min_visibility: float = 0.5
    """Below this the lower body is not reliably tracked and no decision is
    made at all."""

    velocity_window: float = 0.15
    """Seconds over which speed is averaged."""
