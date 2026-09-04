"""Camera layer.

The only part of KinetiRun that talks to hardware. Everything above this layer
receives a stream of ready-to-use frames and never deals with device errors,
backends or mirroring.
"""

from kinetirun.camera.fps import FpsCounter
from kinetirun.camera.webcam import Camera, CameraError

__all__ = ["Camera", "CameraError", "FpsCounter"]
