"""Entry point: `uv run python -m kinetirun`.

Phase 2 milestone - webcam opens, live mirrored preview with an FPS readout,
Q quits cleanly. No pose estimation, no movement detection.
"""

import time
from typing import Optional

import cv2
import numpy as np

from kinetirun.camera import Camera, CameraError, FpsCounter

WINDOW_NAME = "KinetiRun - camera foundation"

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)  # BGR, not RGB
BLACK = (0, 0, 0)

KEY_ESCAPE = 27

# Below this the temporal signals movement detection depends on get too coarse:
# a one-second squat sampled 20 times is workable, sampled 10 times is not.
LOW_FPS_WARNING = 20.0
FPS_WARNING_AFTER_FRAMES = 60


def draw_fps(frame: np.ndarray, fps: Optional[float]) -> None:
    """Draw the FPS readout onto the frame, in place.

    `org` in putText is the BOTTOM-LEFT corner of the text, so a small y value
    puts the text off the top of the image. Hence y=30, not y=10.

    The text is drawn twice: a thick black pass underneath and a green pass on
    top. That outline keeps it readable against a bright background - webcam
    footage is not a controlled backdrop.
    """
    text = "FPS: --" if fps is None else f"FPS: {fps:.1f}"
    org = (10, 30)

    cv2.putText(frame, text, org, FONT, 0.8, BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, 0.8, GREEN, 2, cv2.LINE_AA)


def run() -> None:
    """Main capture loop."""
    counter = FpsCounter()
    frames = 0
    warned = False

    with Camera() as camera:
        width, height = camera.resolution
        print(f"Camera opened at {width}x{height}. Press Q or Escape to quit.")

        try:
            while True:
                frame = camera.read()
                counter.tick(time.perf_counter())
                frames += 1

                # Frame rate on this camera is lighting-dependent: a dim room
                # makes auto-exposure lengthen each frame, halving or thirding
                # the rate. Say so out loud, once - otherwise a dark evening
                # looks like a broken movement detector later on.
                if (
                    not warned
                    and frames == FPS_WARNING_AFTER_FRAMES
                    and counter.fps is not None
                    and counter.fps < LOW_FPS_WARNING
                ):
                    warned = True
                    print(
                        f"Warning: only {counter.fps:.1f} FPS. This is almost "
                        "always too little light - the camera lengthens its "
                        "exposure and drops frames. Try brighter lighting."
                    )

                draw_fps(frame, counter.fps)
                cv2.imshow(WINDOW_NAME, frame)

                # waitKey does double duty: it reads the keyboard AND gives the
                # OpenCV window the chance to actually paint itself. Without it
                # the window stays black or frozen.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), KEY_ESCAPE):
                    break

                # Closing the window with the X button should also stop us,
                # otherwise the loop runs on against a window that is gone.
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            # The camera is released by the `with` block, but the OpenCV window
            # is a separate resource with its own cleanup.
            cv2.destroyAllWindows()


def main() -> int:
    """Run the app, turning a camera failure into a readable message.

    Returns a process exit code: 0 on success, 1 on failure. An uncaught
    traceback also exits 1, but buries the actual problem in noise.
    """
    try:
        run()
    except CameraError as error:
        print(f"Camera error: {error}")
        return 1
    except KeyboardInterrupt:
        # Ctrl+C is a normal way to stop this program, not a crash.
        print("\nInterrupted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
