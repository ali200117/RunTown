"""MediaPipe pose estimation, wrapped.

This module is the ONLY place in KinetiRun allowed to import mediapipe. Phase 4
converts its output into our own BodyPose, and from there the rest of the
system never sees a MediaPipe type again. If we ever swap the pose backend,
this file and the Phase 4 converter are the only things that change.
"""

from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from kinetirun.metrics import RollingAverage

DEFAULT_MODEL = Path("models/pose_landmarker_lite.task")


class PoseEstimationError(RuntimeError):
    """Raised when the pose model cannot be loaded or run."""


class PoseEstimator:
    """Runs the MediaPipe Pose Landmarker over a stream of frames.

        with PoseEstimator() as estimator:
            result = estimator.estimate(frame, timestamp_ms)

    A context manager because the landmarker owns native resources that must
    be closed - the same reasoning as Camera.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Store configuration. The model is loaded in `open()`.

        The three confidence values do different jobs, and conflating them is a
        common source of confusion:
          - detection: how sure the model must be to decide "there is a person"
          - presence:  how sure it must be that a detected pose is still there
          - tracking:  how sure it must be to keep following the same person
                       between frames instead of re-detecting from scratch
        """
        self.model_path = Path(model_path)
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._landmarker: Optional[mp_vision.PoseLandmarker] = None
        self._latency = RollingAverage(window=30)
        self._last_timestamp_ms = -1

    def open(self) -> None:
        if self._landmarker is not None:
            return

        if not self.model_path.exists():
            raise PoseEstimationError(
                f"Pose model not found at {self.model_path}. "
                "Run: uv run python scripts/download_model.py"
            )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            # VIDEO mode lets the model track the person across frames instead
            # of detecting from scratch every time: faster, and far less
            # jittery. The price is that timestamps must strictly increase.
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )

        try:
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as error:  # noqa: BLE001 - surface any loader failure
            raise PoseEstimationError(
                f"Could not load the pose model from {self.model_path}: {error}"
            ) from error

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Run pose estimation on one frame.

        Args:
            frame_bgr: a frame straight from Camera. BGR, as OpenCV delivers.
            timestamp_ms: milliseconds, strictly increasing. VIDEO mode rejects
                a timestamp that does not advance.

        Returns:
            The raw MediaPipe PoseLandmarkerResult. Deliberately not wrapped
            yet - Phase 4 is where our own model appears. Returning the raw
            result here keeps this module a thin, honest adapter.
        """
        if self._landmarker is None:
            raise PoseEstimationError("PoseEstimator is not open.")

        # MediaPipe wants RGB; OpenCV gives BGR. This conversion is the reason
        # Camera captures at 640x480 rather than 720p - it costs real time per
        # frame and buys nothing, since the model rescales to 256x256 anyway.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Guard the monotonicity requirement here rather than trusting every
        # caller: a repeated timestamp raises deep inside native code with an
        # error message that explains nothing.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        start = cv2.getTickCount()
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        elapsed_ms = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000.0
        self._latency.add(elapsed_ms)

        return result

    @property
    def latency_ms(self) -> Optional[float]:
        """Rolling average inference time, in milliseconds.

        Worth watching closely: at 30 FPS the whole frame budget is 33 ms. If
        inference alone approaches that, the pipeline cannot keep up and we
        either accept a lower rate or move to a lighter configuration.
        """
        return self._latency.value

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "PoseEstimator":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()
