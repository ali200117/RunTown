"""Calibration layer.

Turns "stand still for a few seconds" into a Baseline that lets every movement
threshold be expressed relative to this body, at this distance.
"""

from kinetirun.calibration.baseline import Baseline
from kinetirun.calibration.calibrator import CalibrationState, Calibrator

__all__ = ["Baseline", "Calibrator", "CalibrationState"]
