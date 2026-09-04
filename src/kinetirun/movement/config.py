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


@dataclass(frozen=True)
class JumpConfig:
    """Tuning for JumpDetector.

    Height is upward hip displacement from neutral, in SHOULDER WIDTHS.
    Positive is up, which is the opposite sign from raw image coordinates.
    """

    min_height: float = 0.18
    """How high the hips must rise for the jump to count. Lower than the squat
    and side thresholds on purpose: a jump is a small displacement performed
    fast, and speed does most of the work of proving intent."""

    target_height: float = 0.32
    """Height that scores full quality."""

    min_foot_lift: float = 0.08
    """How far the FEET must leave the ground, in shoulder widths. The
    anti-cheat rule: rising onto the toes lifts the hips without lifting the
    ankles. Set to 0.0 to disable when the feet are outside the frame."""

    neutral_band: float = 0.07
    """Inside this band of the baseline the user counts as grounded."""

    takeoff_rate: float = 1.0
    """Upward speed, in shoulder widths per second, that starts a jump. This is
    the main thing separating a jump from slowly standing up - it is far higher
    than the squat and side thresholds."""

    settle_rate: float = 0.35
    """Below this upward speed the rise is over: the apex."""

    landing_rate: float = 0.5
    """Downward speed that marks the descent back to the ground."""

    min_duration: float = 0.15
    max_duration: float = 1.5
    """A jump is over quickly. Anything longer is not a jump."""

    min_visibility: float = 0.5

    velocity_window: float = 0.12
    """Shorter than the other detectors: a jump is brief, and a long averaging
    window would smear takeoff and apex together."""


@dataclass(frozen=True)
class LeanConfig:
    """Tuning for LeanDetector.

    Everything is an angle in DEGREES, relative to the calibrated standing
    posture. Positive is a lean to the user's own right.

    Angles rather than distances on purpose: an angle is already independent
    of body size and camera distance, so unlike the step thresholds these
    numbers need no shoulder-width normalization.
    """

    min_angle: float = 14.0
    """How far the torso must tilt for the lean to count. Roughly the point
    where it stops looking like standing and starts looking deliberate."""

    target_angle: float = 25.0
    """Tilt that scores full quality."""

    neutral_angle: float = 5.0
    """Inside this band the user counts as upright. Returning here completes
    the movement and re-arms the detector."""

    start_rate: float = 22.0
    """Tilt speed, in degrees per second, that starts a lean. This is what
    separates a deliberate bow from slowly drifting off balance."""

    settle_rate: float = 10.0
    """Below this speed the lean has reached its furthest point."""

    return_rate: float = 10.0
    """Speed back toward upright that marks the return."""

    min_duration: float = 0.20
    """Faster than this is a twitch, not a lean."""

    max_duration: float = 3.0

    min_visibility: float = 0.5

    velocity_window: float = 0.15


@dataclass(frozen=True)
class ArmConfig:
    """Tuning for ArmDetector.

    Everything is an angle in DEGREES: how far one arm is raised, relative to
    the calibrated resting pose. Positive means the RIGHT arm, negative the
    LEFT.

    Reference points: 0 is arms hanging, 90 is one arm straight out
    horizontally, 180 is straight up.

    The most reliable of the three sideways options. Hands are large and rarely
    occluded, the movement is unambiguous, and only the shoulders and wrists
    need to be in frame - where step and lean both depend on the lower body.
    """

    min_angle: float = 55.0
    """How far the arm must be raised for the swipe to count. Comfortably
    below horizontal, so the gesture does not demand precision, but far above
    anything that happens by accident."""

    target_angle: float = 90.0
    """Arm straight out horizontally: full quality."""

    neutral_angle: float = 20.0
    """Below this the arms count as down. Returning here completes the movement
    and re-arms the detector."""

    start_rate: float = 90.0
    """Raising speed, in degrees per second, that starts a swipe. Roughly one
    second from hanging to horizontal - slower than that is not a gesture."""

    settle_rate: float = 40.0
    """Below this speed the arm has reached its highest point."""

    return_rate: float = 40.0
    """Lowering speed that marks the return."""

    min_duration: float = 0.12
    """Deliberately short. This is the movement for reacting in a game."""

    max_duration: float = 2.5

    min_visibility: float = 0.5
    """Applied to WRISTS and SHOULDERS only, not the lower body. Arm swipes
    work with the legs out of frame, which is the point of having this mode."""

    velocity_window: float = 0.12
