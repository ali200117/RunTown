"""Webcam capture.

Responsibilities of this module - and nothing beyond them:
  - open a capture device, failing loudly if it cannot
  - hand out frames one at a time
  - mirror each frame horizontally, in exactly one place
  - release the device reliably, including on crash
"""

from types import TracebackType
from typing import Optional, Type

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the capture device cannot be opened or read from.

    A dedicated exception type lets callers handle camera problems specifically
    instead of catching a bare RuntimeError and hoping for the best.
    """


class Camera:
    """A webcam, used as a context manager.

        with Camera() as cam:
            frame = cam.read()

    The device is released when the `with` block exits - normally, via break,
    or through an exception. That guarantee is the whole reason this is a
    context manager: a leaked device stays locked by the OS.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
    ) -> None:
        """Store configuration only. The device is opened in `open()`.

        Opening in __init__ means a failure leaves you holding a half-built
        object. Opening in __enter__ instead makes the device's lifetime match
        the `with` block exactly.

        Args:
            device_index: which camera. Built-in webcam is usually 0.
            width, height: requested resolution. The driver may ignore this -
                always trust `frame.shape`, never the value you asked for.
            mirror: flip horizontally so the user sees a mirror image. Keep
                this True. The rest of KinetiRun assumes frames are mirrored,
                and flipping in more than one place will invert left/right in
                Phase 8.
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the capture device and request the configured resolution."""
        if self._capture is not None:
            return

        # DirectShow rather than the Windows default (MSMF): MSMF takes several
        # seconds to open the device and frequently ignores resolution changes.
        capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera at index {self.device_index}. "
                "Check that a camera is connected, that no other application "
                "(Teams, Zoom, browser) is using it, and that camera access is "
                "allowed under Windows privacy settings."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._capture = capture

    def read(self) -> np.ndarray:
        """Return the next frame as a BGR uint8 array of shape (H, W, 3).

        Note for Phase 3: this frame is BGR. MediaPipe wants RGB. That
        conversion belongs in the vision layer - this module's job is to
        deliver raw camera frames, not to know who consumes them.
        """
        if self._capture is None:
            raise CameraError("Camera is not open. Use it as a context manager.")

        # `ok` can be False on a healthy, open device: a dropped frame, or the
        # camera being unplugged mid-session.
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError(
                f"Failed to read a frame from camera {self.device_index}. "
                "The device may have been disconnected."
            )

        if self.mirror:
            frame = cv2.flip(frame, 1)  # 1 = horizontal

        return frame

    def release(self) -> None:
        """Release the device. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_open(self) -> bool:
        """Whether the device is currently open."""
        return self._capture is not None

    @property
    def resolution(self) -> tuple[int, int]:
        """The resolution the driver actually gave us, as (width, height).

        Drivers routinely silently substitute a resolution they support, so
        this is the number worth trusting - and worth printing at startup.
        """
        if self._capture is None:
            raise CameraError("Camera is not open.")

        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        # Returning None means an in-flight exception keeps propagating: we are
        # cleaning up, not swallowing errors.
        self.release()
