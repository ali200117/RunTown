"""BlazePose landmark indices and the skeleton we draw.

MediaPipe returns 33 landmarks as a flat list. Indexing that list with bare
numbers is unreadable and easy to get wrong, so the ones we care about are
named here. Phase 4 builds BodyPose on top of these names.
"""

from enum import IntEnum


class Landmark(IntEnum):
    """Indices into MediaPipe's 33-point pose landmark list.

    IMPORTANT - these labels are NOT the user's own left and right.

    MediaPipe infers anatomy from what the image looks like, and Camera mirrors
    every frame before the model sees it. In a mirrored frame the user's left
    arm is the one that LOOKS like a right arm, so MediaPipe labels it
    RIGHT_WRIST.

    The reliable way to read these names, for a person facing the camera:

        RIGHT_* is always the side that appears on the LEFT of the screen
        LEFT_*  is always the side that appears on the RIGHT of the screen

    and because frames are mirrored, screen-right is the user's own right. So:

        LEFT_*  = the user's RIGHT side
        RIGHT_* = the user's LEFT side

    Anything symmetric - hip centre, shoulder width, mean knee angle - is
    unaffected. Only code that treats one side differently from the other has
    to care, which in practice means arm_raise.
    """

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# The connections we actually draw. Face detail is deliberately left out: it
# adds clutter and tells us nothing about squats, side steps or jumps.
POSE_CONNECTIONS: tuple[tuple[Landmark, Landmark], ...] = (
    # Shoulders and arms
    (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    # Torso
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    # Legs - the ones that matter most for this project
    (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    # Feet
    (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
    (Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_FOOT_INDEX),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
    (Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_FOOT_INDEX),
)
