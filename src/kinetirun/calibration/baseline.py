"""The calibrated neutral pose.

Everything a movement detector needs in order to express a threshold relative
to THIS body, standing at THIS distance from the camera - rather than in pixels
or raw metres that only work for one person in one room.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from kinetirun.tracking import BodyPose


@dataclass(frozen=True)
class Baseline:
    """Measurements of the user standing still in a neutral position.

    Metres for body shape (distance-independent), normalized image units for
    position in the room (which world space cannot see, since world space is
    anchored to the hips and moves with you).
    """

    shoulder_width: float          # metres - the primary ruler
    torso_height: float            # metres
    hip_height: float              # metres, hips above ankles, standing
    knee_angle: float              # degrees, standing
    torso_tilt: float              # degrees, sideways lean when standing
    hip_x_image: float             # 0..1 across the frame
    hip_y_image: float             # 0..1 down the frame
    ankle_x_image: float           # 0..1 across the frame
    ankle_y_image: float           # 0..1 down the frame
    arm_raise: float               # degrees, arms at rest
    shoulder_width_image: float    # fraction of frame width - the image ruler
    sample_count: int

    # ---- turning raw measurements into body-relative ratios -------------

    def hip_drop(self, pose: BodyPose) -> float:
        """How far the hips have dropped, as a fraction of standing hip height.

        0.0 standing, 0.25 a shallow bend, 0.35+ a real squat. This is the
        squat signal, and it is a ratio so it means the same thing for a tall
        and a short person.
        """
        if self.hip_height <= 0:
            return 0.0
        return (self.hip_height - pose.hip_height_above_ankles) / self.hip_height

    def lateral_offset(self, pose: BodyPose) -> float:
        """Sideways displacement from neutral, in shoulder widths.

        Negative is toward the left of the screen. Frames are mirrored, so the
        left of the screen is the user's own left.

        Dividing by the image-space shoulder width cancels out camera distance:
        step back and both the movement and the ruler shrink together.
        """
        if self.shoulder_width_image <= 0:
            return 0.0
        return (pose.hip_center_image.x - self.hip_x_image) / self.shoulder_width_image

    def arm_reach(self, pose: BodyPose) -> float:
        """How far an arm is raised relative to the calibrated resting pose.

        Degrees. Positive means the RIGHT arm is up, negative the LEFT.

        Measured against the baseline rather than against absolute vertical so
        that resting with the arms slightly out, or with one hand in a pocket,
        does not bias the signal.
        """
        return pose.arm_raise - self.arm_raise

    def lean_angle(self, pose: BodyPose) -> float:
        """Sideways lean relative to the calibrated neutral posture, in degrees.

        Measured against the baseline rather than against true vertical:
        nobody stands perfectly straight, and a camera is rarely perfectly
        level. Both errors cancel out here.
        """
        return pose.torso_tilt - self.torso_tilt

    def foot_offset(self, pose: BodyPose) -> float:
        """Sideways foot displacement from neutral, in shoulder widths.

        Same units and sign convention as lateral_offset, so the two are
        directly comparable: a real step moves both, a lean moves only the hips.
        """
        if self.shoulder_width_image <= 0:
            return 0.0
        return (pose.ankle_center_image.x - self.ankle_x_image) / self.shoulder_width_image

    def foot_lift(self, pose: BodyPose) -> float:
        """How far the feet have risen from neutral, in shoulder widths.

        Positive is upward, unlike vertical_offset - a lift reads more
        naturally as a positive number. This is the anti-cheat signal for
        jumps: rising onto the toes lifts the hips a little while the ankles
        barely move, whereas an actual jump takes the whole body up.
        """
        if self.shoulder_width_image <= 0:
            return 0.0
        return (self.ankle_y_image - pose.ankle_center_image.y) / self.shoulder_width_image

    def vertical_offset(self, pose: BodyPose) -> float:
        """Vertical displacement from neutral, in shoulder widths.

        Negative is upward, because image y grows downward. This is the jump
        signal in Phase 9: a jump moves the whole body up the frame, which
        world space cannot detect at all.
        """
        if self.shoulder_width_image <= 0:
            return 0.0
        return (pose.hip_center_image.y - self.hip_y_image) / self.shoulder_width_image

    # ---- persistence ----------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
