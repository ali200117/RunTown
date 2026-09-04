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


@dataclass(frozen=True)
class SideConfig:
    """Tuning for SideDetector.

    Offset is `lateral_offset`: sideways hip displacement from neutral measured
    in SHOULDER WIDTHS. Negative is toward the user's own left, because frames
    are mirrored. One shoulder width is roughly a comfortable side step.
    """

    min_offset: float = 0.55
    """How far the hips must travel sideways for the step to count."""

    target_offset: float = 0.85
    """Offset that scores full quality."""

    neutral_offset: float = 0.20
    """Inside this band the user counts as centred. Returning here completes
    the movement and re-arms the detector."""

    min_foot_travel: float = 0.20
    """How far the FEET must move, in shoulder widths. This is the anti-lean
    rule: tilting the upper body shifts the hips without moving the feet, and
    must not register as a side step. Set to 0.0 to disable - useful if the
    feet fall outside the camera frame."""

    start_rate: float = 0.6
    """Sideways speed, in shoulder widths per second, that starts a step."""

    settle_rate: float = 0.35
    """Below this speed the outward movement is considered finished."""

    return_rate: float = 0.35
    """Speed back toward centre that marks the return."""

    min_duration: float = 0.25
    """Faster than this is a twitch, not a step."""

    max_duration: float = 3.0

    min_visibility: float = 0.5

    velocity_window: float = 0.15
