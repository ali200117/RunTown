"""Vision layer.

Turns camera frames into pose landmarks. The only place in KinetiRun that
imports mediapipe.
"""

from kinetirun.vision.landmarks import POSE_CONNECTIONS, Landmark
from kinetirun.vision.pose_estimator import PoseEstimationError, PoseEstimator

__all__ = [
    "Landmark",
    "POSE_CONNECTIONS",
    "PoseEstimator",
    "PoseEstimationError",
]
