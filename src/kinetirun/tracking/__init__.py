"""Tracking layer.

Our own body representation, and (from Phase 6) its history over time. Knows
nothing about MediaPipe, cameras, games or input.
"""

from kinetirun.tracking.body_pose import BodyPose, Point3D, angle_between

__all__ = ["BodyPose", "Point3D", "angle_between"]
