"""BlazePose landmark indices and the skeleton we draw.

MediaPipe returns 33 landmarks as a flat list. Indexing that list with bare
numbers is unreadable and easy to get wrong, so the ones we care about are
named here. Phase 4 builds BodyPose on top of these names.
"""

from enum import IntEnum


class Landmark(IntEnum):
    """Indices into MediaPipe's 33-point pose landmark list.

    Left and right are from the SUBJECT's point of view, not the camera's.
    Because Camera mirrors every frame, the subject's left hip also appears on
    the left of the screen - which is what makes the mirror feel natural.
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
