"""Collecting a Baseline from a user standing still.

State machine:
    WAITING    - no usable pose, or the user is moving
    COLLECTING - a still, reliable pose is being sampled
    COMPLETE   - enough samples gathered, baseline available

Movement during collection resets progress rather than being averaged in. A
baseline quietly polluted by a half-step would shift every threshold in the
system, and nothing downstream could detect that it had happened.
"""

from __future__ import annotations

import statistics
from enum import Enum
from typing import List, Optional

from kinetirun.calibration.baseline import Baseline
from kinetirun.tracking import BodyPose


class CalibrationState(Enum):
    WAITING = "WAITING"
    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"


class Calibrator:
    """Builds a Baseline from a sequence of poses."""

    def __init__(
        self,
        duration: float = 3.0,
        min_samples: int = 20,
        stillness_tolerance: float = 0.25,
        visibility_threshold: float = 0.5,
    ) -> None:
        """
        Args:
            duration: seconds the user must hold the neutral pose.
            min_samples: floor on sample count, so a bad frame rate cannot
                produce a baseline built from four measurements.
            stillness_tolerance: how far the hips may drift during collection,
                in shoulder widths, before progress resets.
            visibility_threshold: poses below this are ignored entirely.
        """
        self.duration = duration
        self.min_samples = min_samples
        self.stillness_tolerance = stillness_tolerance
        self.visibility_threshold = visibility_threshold

        self._samples: List[BodyPose] = []
        self._state = CalibrationState.WAITING
        self._baseline: Optional[Baseline] = None

    # ---- driving --------------------------------------------------------

    def update(self, pose: Optional[BodyPose]) -> CalibrationState:
        """Feed one pose. Returns the resulting state."""
        if self._state is CalibrationState.COMPLETE:
            return self._state

        if pose is None or not pose.is_reliable(self.visibility_threshold):
            self._restart()
            return self._state

        if self._samples and not self._is_still(pose):
            self._restart()
            # The current pose is itself fine - it is the drift from the older
            # samples that failed. Start the new window here rather than
            # throwing away a perfectly good frame.

        self._samples.append(pose)
        self._state = CalibrationState.COLLECTING

        if self._has_enough():
            self._baseline = self._build()
            self._state = CalibrationState.COMPLETE

        return self._state

    def reset(self) -> None:
        """Discard the baseline and start over."""
        self._samples.clear()
        self._baseline = None
        self._state = CalibrationState.WAITING

    # ---- inspection -----------------------------------------------------

    @property
    def state(self) -> CalibrationState:
        return self._state

    @property
    def baseline(self) -> Optional[Baseline]:
        return self._baseline

    @property
    def progress(self) -> float:
        """0.0 to 1.0. Reflects whichever requirement is further from being met."""
        if self._state is CalibrationState.COMPLETE:
            return 1.0
        if not self._samples:
            return 0.0

        elapsed = self._samples[-1].timestamp - self._samples[0].timestamp
        by_time = elapsed / self.duration if self.duration > 0 else 1.0
        by_count = len(self._samples) / self.min_samples if self.min_samples else 1.0
        return min(1.0, min(by_time, by_count))

    # ---- internals ------------------------------------------------------

    def _restart(self) -> None:
        self._samples.clear()
        self._state = CalibrationState.WAITING

    def _is_still(self, pose: BodyPose) -> bool:
        """Whether the hips have stayed put relative to the first sample."""
        first = self._samples[0]
        ruler = first.shoulder_width_image
        if ruler <= 0:
            return False

        drift_x = abs(pose.hip_center_image.x - first.hip_center_image.x) / ruler
        drift_y = abs(pose.hip_center_image.y - first.hip_center_image.y) / ruler
        return max(drift_x, drift_y) <= self.stillness_tolerance

    def _has_enough(self) -> bool:
        if len(self._samples) < self.min_samples:
            return False
        elapsed = self._samples[-1].timestamp - self._samples[0].timestamp
        return elapsed >= self.duration

    def _build(self) -> Baseline:
        """Reduce the samples to one baseline.

        Median rather than mean throughout: pose estimation produces occasional
        wild outliers, and a single frame where MediaPipe put an ankle on the
        wrong side of the room would drag a mean noticeably. The median simply
        ignores it.
        """
        def median_of(extract) -> float:
            return statistics.median(extract(sample) for sample in self._samples)

        return Baseline(
            shoulder_width=median_of(lambda p: p.shoulder_width),
            torso_height=median_of(lambda p: p.torso_height),
            hip_height=median_of(lambda p: p.hip_height_above_ankles),
            knee_angle=median_of(lambda p: p.mean_knee_angle),
            torso_tilt=median_of(lambda p: p.torso_tilt),
            hip_x_image=median_of(lambda p: p.hip_center_image.x),
            hip_y_image=median_of(lambda p: p.hip_center_image.y),
            ankle_x_image=median_of(lambda p: p.ankle_center_image.x),
            ankle_y_image=median_of(lambda p: p.ankle_center_image.y),
            arm_raise=median_of(lambda p: p.arm_raise),
            shoulder_width_image=median_of(lambda p: p.shoulder_width_image),
            sample_count=len(self._samples),
        )
