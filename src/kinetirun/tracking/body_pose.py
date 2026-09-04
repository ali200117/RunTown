"""Our own body representation.

This is the boundary. Above this module nothing knows that MediaPipe exists:
movement detectors, calibration and motion history all speak BodyPose. Swapping
the pose backend later means rewriting `from_landmarker_result` and nothing
else.

It also makes the rest of the project testable. A BodyPose can be constructed
by hand, so a squat detector can be fed a scripted sequence of poses with no
camera, no model and no lighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from kinetirun.vision.landmarks import Landmark


@dataclass(frozen=True)
class Point3D:
    """A single joint.

    Frozen because a pose is a measurement of a moment: once recorded it must
    not change. Motion history in Phase 6 keeps references to old poses, and
    silent mutation there would be very hard to debug.

    Axis convention (inherited from MediaPipe, image-style):
        x - increases to the right of the image
        y - increases DOWNWARD, so a lower body part has a LARGER y
        z - depth; negative is toward the camera

    The y direction is worth burning into memory: "the hip went down" means
    "hip.y increased". Getting this backwards inverts squat detection.
    """

    x: float
    y: float
    z: float
    visibility: float

    def distance_to(self, other: "Point3D") -> float:
        """Straight-line 3D distance to another point, in the same units."""
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def midpoint(self, other: "Point3D") -> "Point3D":
        """The point halfway between two joints.

        Visibility of the midpoint is the *minimum* of the two, not the mean:
        a midpoint derived from one confident and one guessed joint is itself
        a guess, and averaging would flatter it.
        """
        return Point3D(
            x=(self.x + other.x) / 2,
            y=(self.y + other.y) / 2,
            z=(self.z + other.z) / 2,
            visibility=min(self.visibility, other.visibility),
        )


def angle_between(a: Point3D, vertex: Point3D, b: Point3D) -> float:
    """Angle in degrees at `vertex`, between the segments to `a` and `b`.

    Used for joint angles: angle_between(hip, knee, ankle) is the knee angle,
    where roughly 180 degrees is a straight leg and 90 is a deep squat.

    Computed from the dot product:
        cos(theta) = (u . v) / (|u| |v|)
    """
    ux, uy, uz = a.x - vertex.x, a.y - vertex.y, a.z - vertex.z
    vx, vy, vz = b.x - vertex.x, b.y - vertex.y, b.z - vertex.z

    dot = ux * vx + uy * vy + uz * vz
    magnitude = math.sqrt(ux * ux + uy * uy + uz * uz) * math.sqrt(
        vx * vx + vy * vy + vz * vz
    )

    if magnitude == 0:
        return 0.0

    # Floating point can push the ratio a hair outside [-1, 1], and acos would
    # then raise. Clamping is cheaper than trusting the arithmetic.
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / magnitude))))


@dataclass(frozen=True)
class BodyPose:
    """One person, at one instant, in two coordinate spaces.

    Attributes:
        timestamp: seconds from time.perf_counter(). Every temporal decision in
            this project is made in seconds, never in frame counts - frame rate
            varies with the lighting, so "15 frames ago" is not a fixed
            duration but "300 ms ago" is.
        world: 33 points in METRES, origin at the centre between the hips.
            Independent of how far you stand from the camera. Everything that
            is MEASURED uses these.
        image: 33 points in NORMALIZED image coordinates (0..1 across the
            frame). Everything that is DRAWN, plus horizontal position in the
            room for Phase 8, uses these.
    """

    timestamp: float
    world: tuple[Point3D, ...]
    image: tuple[Point3D, ...]

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_landmarker_result(cls, result, timestamp: float) -> Optional["BodyPose"]:
        """Convert a MediaPipe result, or return None if nobody was detected.

        Deliberately duck-typed: it reads .x/.y/.z/.visibility off whatever it
        is given and never imports mediapipe. That keeps the dependency arrow
        pointing one way and lets tests build fake results out of plain
        objects.
        """
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None

        return cls(
            timestamp=timestamp,
            world=cls._convert(result.pose_world_landmarks[0]),
            image=cls._convert(result.pose_landmarks[0]),
        )

    @staticmethod
    def _convert(landmarks: Sequence) -> tuple[Point3D, ...]:
        return tuple(
            Point3D(
                x=point.x,
                y=point.y,
                z=point.z,
                # World landmarks sometimes omit visibility; fall back to the
                # optimistic value rather than crashing, since the image
                # landmarks carry the trustworthy figure anyway.
                visibility=getattr(point, "visibility", 1.0) or 0.0,
            )
            for point in landmarks
        )

    # ---- measured in metres (world space) -------------------------------

    @property
    def hip_center(self) -> Point3D:
        """Centre between the hips. In world space this is the origin, so it is
        near (0, 0, 0) by definition - useful mainly for its visibility."""
        return self.world[Landmark.LEFT_HIP].midpoint(self.world[Landmark.RIGHT_HIP])

    @property
    def shoulder_center(self) -> Point3D:
        return self.world[Landmark.LEFT_SHOULDER].midpoint(
            self.world[Landmark.RIGHT_SHOULDER]
        )

    @property
    def shoulder_width(self) -> float:
        """Shoulder-to-shoulder distance in metres.

        This is the project's ruler. Phase 5 expresses movement thresholds as
        fractions of it ("0.7 x shoulder width") so that the same numbers work
        for different body sizes and camera distances.
        """
        return self.world[Landmark.LEFT_SHOULDER].distance_to(
            self.world[Landmark.RIGHT_SHOULDER]
        )

    @property
    def torso_height(self) -> float:
        """Shoulder centre to hip centre, in metres. The second body-size ruler."""
        return self.shoulder_center.distance_to(self.hip_center)

    @property
    def left_knee_angle(self) -> float:
        """Degrees. ~180 is a straight leg, ~90 a deep squat."""
        return angle_between(
            self.world[Landmark.LEFT_HIP],
            self.world[Landmark.LEFT_KNEE],
            self.world[Landmark.LEFT_ANKLE],
        )

    @property
    def right_knee_angle(self) -> float:
        return angle_between(
            self.world[Landmark.RIGHT_HIP],
            self.world[Landmark.RIGHT_KNEE],
            self.world[Landmark.RIGHT_ANKLE],
        )

    @property
    def mean_knee_angle(self) -> float:
        """Both knees averaged.

        Squat detection uses this rather than a single leg: one knee can be
        badly estimated when the far leg is partly occluded, and averaging
        keeps a single bad estimate from dominating.
        """
        return (self.left_knee_angle + self.right_knee_angle) / 2

    @property
    def hip_height_above_ankles(self) -> float:
        """How high the hips sit above the ankles, in metres.

        This is the squat signal. Note the sign flip: y grows downward, so
        "higher" means a more negative y, and we subtract in this order to get
        a positive number that shrinks as you descend.
        """
        ankle_center = self.world[Landmark.LEFT_ANKLE].midpoint(
            self.world[Landmark.RIGHT_ANKLE]
        )
        return ankle_center.y - self.hip_center.y

    # ---- position in the room (image space) -----------------------------

    @property
    def hip_center_image(self) -> Point3D:
        """Hip centre in normalized image coordinates.

        Where the body IS, rather than what shape it is in. Side-step detection
        in Phase 8 tracks this: world space cannot see it, because world space
        is anchored to the hips and therefore moves with you.
        """
        return self.image[Landmark.LEFT_HIP].midpoint(self.image[Landmark.RIGHT_HIP])

    @property
    def shoulder_center_image(self) -> Point3D:
        return self.image[Landmark.LEFT_SHOULDER].midpoint(
            self.image[Landmark.RIGHT_SHOULDER]
        )

    @property
    def ankle_center_image(self) -> Point3D:
        """Centre between the ankles, in normalized image coordinates.

        The anti-lean signal. Leaning sideways swings the shoulders and drags
        the hips a little, but the feet stay put. An actual side step moves
        them. Comparing the two is what separates a step from a tilt.
        """
        return self.image[Landmark.LEFT_ANKLE].midpoint(
            self.image[Landmark.RIGHT_ANKLE]
        )

    def arm_elevation(self, side: str) -> float:
        """How far one arm is raised, in degrees. "left" or "right".

        Measured as the angle of the shoulder-to-wrist line away from hanging
        straight down:

            0   - arm hanging at your side
            90  - arm straight out horizontally
            180 - arm straight up

        An angle, so it is automatically independent of body size and camera
        distance, and it uses the arm's WHOLE travel rather than only its
        sideways component. Raising a hanging arm to horizontal moves the wrist
        barely half a shoulder width sideways but a full 90 degrees here, which
        is why this signal is far harder to confuse with ordinary fidgeting.
        """
        shoulder_index = (
            Landmark.LEFT_SHOULDER if side == "left" else Landmark.RIGHT_SHOULDER
        )
        wrist_index = Landmark.LEFT_WRIST if side == "left" else Landmark.RIGHT_WRIST

        shoulder = self.image[shoulder_index]
        wrist = self.image[wrist_index]

        across = abs(wrist.x - shoulder.x)
        # Image y grows downward, so a hanging wrist sits BELOW the shoulder
        # and this is positive.
        below = wrist.y - shoulder.y

        return math.degrees(math.atan2(across, below))

    @property
    def arm_raise(self) -> float:
        """Which arm the USER raised, and how far, in degrees.

            +90 - the user's own right arm out horizontally
            -90 - the user's own left arm out horizontally
              0 - both arms hanging, OR both raised equally

        Note the crossed mapping: MediaPipe's LEFT_* landmarks are the user's
        RIGHT side, because Camera mirrors the frame before the model sees it
        and the model reads anatomy from appearance. See the Landmark docstring
        - this is the one place in the project where that distinction bites.

        Collapsing two arms into one signed number keeps the detector simple
        and makes the ambiguous case safe: raising both arms cancels to zero
        and triggers nothing, which is the right answer when the gesture does
        not name a direction.
        """
        return self.arm_elevation("left") - self.arm_elevation("right")

    @property
    def torso_tilt(self) -> float:
        """Sideways lean of the torso, in degrees. Positive is to the right.

        The angle of the hip-centre-to-shoulder-centre line away from vertical,
        measured in image space. 0 is upright, +25 is a clear lean to the
        screen's right - which is the user's own right, since frames are
        mirrored.

        An angle rather than a distance, because an angle is already
        independent of body size and camera distance: leaning 20 degrees is 20
        degrees whether you are tall or standing far back. No shoulder-width
        division needed.
        """
        hips = self.hip_center_image
        shoulders = self.shoulder_center_image

        across = shoulders.x - hips.x
        # Image y grows downward, so shoulders sit at a SMALLER y than hips.
        # Subtracting in this order makes `up` positive.
        up = hips.y - shoulders.y

        if up <= 0:
            # Shoulders at or below the hips: the person is bent double or
            # badly tracked, and an angle would be meaningless.
            return 0.0

        return math.degrees(math.atan2(across, up))

    @property
    def shoulder_width_image(self) -> float:
        """Shoulder width as a fraction of frame width.

        Shrinks as you step back from the camera, which is exactly why it is
        useful: dividing an image-space displacement by it cancels the distance
        out, turning pixels into body-relative units.
        """
        left = self.image[Landmark.LEFT_SHOULDER]
        right = self.image[Landmark.RIGHT_SHOULDER]
        return math.hypot(left.x - right.x, left.y - right.y)

    # ---- confidence -----------------------------------------------------

    def visibility_of(self, *landmarks: Landmark) -> float:
        """The weakest visibility among the given landmarks.

        A measurement is only as trustworthy as its worst input, so this
        deliberately reports the minimum rather than an average.
        """
        return min(self.image[landmark].visibility for landmark in landmarks)

    @property
    def lower_body_visibility(self) -> float:
        """Confidence in the joints squats and jumps depend on."""
        return self.visibility_of(
            Landmark.LEFT_HIP,
            Landmark.RIGHT_HIP,
            Landmark.LEFT_KNEE,
            Landmark.RIGHT_KNEE,
            Landmark.LEFT_ANKLE,
            Landmark.RIGHT_ANKLE,
        )

    def is_reliable(self, threshold: float = 0.5) -> bool:
        """Whether the lower body is confidently tracked.

        Movement detectors should refuse to make decisions when this is False -
        MediaPipe always returns coordinates, including for joints it cannot
        see, so without this check a detector happily measures fiction.
        """
        return self.lower_body_visibility >= threshold
