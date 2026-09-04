"""Tracking layer.

Our own body representation, its smoothed form, and its recent history. Knows
nothing about MediaPipe, cameras, games or input.
"""

from kinetirun.tracking.body_pose import BodyPose, Point3D, angle_between
from kinetirun.tracking.motion_history import MotionHistory
from kinetirun.tracking.smoothing import PoseSmoother

__all__ = [
    "BodyPose",
    "Point3D",
    "angle_between",
    "MotionHistory",
    "PoseSmoother",
]
