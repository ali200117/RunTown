"""Entry point: `uv run python -m kinetirun`.

The full pipeline:

    camera -> pose -> body model -> smoothing -> history -> calibration
           -> movement engine -> events -> game profile -> input adapter

Keys:
    Q / Escape  quit
    R           recalibrate
    G           toggle game input on and off
"""

import time

import cv2

from kinetirun.calibration import Calibrator
from kinetirun.camera import Camera, CameraError, FpsCounter
from kinetirun.game import SUBWAY_SURFERS
from kinetirun.input import InputDispatcher, PyAutoGuiAdapter
from kinetirun.movement import MovementEngine
from kinetirun.tracking import BodyPose, MotionHistory, PoseSmoother
from kinetirun.ui import draw_event_flash, draw_skeleton, draw_text, format_stats
from kinetirun.vision import PoseEstimationError, PoseEstimator

WINDOW_NAME = "KinetiRun"

KEY_ESCAPE = 27

# Below this the temporal signals movement detection depends on get too coarse:
# a one-second squat sampled 20 times is workable, sampled 10 times is not.
LOW_FPS_WARNING = 20.0
FPS_WARNING_AFTER_FRAMES = 60


def run() -> None:
    """Main loop."""
    counter = FpsCounter()
    smoother = PoseSmoother()
    history = MotionHistory()
    calibrator = Calibrator()
    # Sideways input is an ARM SWIPE: fast enough to react with in a game, and
    # it works with the legs out of frame. Use "lean" for a whole-body tilt or
    # "step" for a real side step.
    engine = MovementEngine(sideways="arm")

    # Input starts DISABLED. Keystrokes go to whichever window has focus, so
    # enabling it is always a deliberate act by the user.
    dispatcher = InputDispatcher(SUBWAY_SURFERS, PyAutoGuiAdapter())

    frames = 0
    warned = False

    with Camera() as camera, PoseEstimator() as estimator:
        width, height = camera.resolution
        print(f"Camera opened at {width}x{height}.")
        print(f"Profile: {SUBWAY_SURFERS.name}, sideways mode: {engine.sideways}.")
        print("Q quits, R recalibrates, G toggles input.")

        # MediaPipe's VIDEO mode wants a timestamp that starts near zero and
        # only ever increases.
        start_time = time.perf_counter()

        try:
            while True:
                frame = camera.read()
                now = time.perf_counter()
                counter.tick(now)
                frames += 1

                result = estimator.estimate(frame, int((now - start_time) * 1000))

                # The boundary crossing: from here on we work with our own type
                # and the MediaPipe result is discarded.
                pose = BodyPose.from_landmarker_result(result, timestamp=now)
                # Smooth before anything measures the pose, so every consumer
                # downstream sees the same stable numbers.
                pose = smoother.smooth(pose)
                history.append(pose)

                if pose is not None:
                    draw_skeleton(frame, pose.image)

                calibrator.update(pose)

                # Detection only runs once a baseline exists: every threshold
                # is expressed relative to it.
                if calibrator.baseline is not None:
                    events = engine.update(pose, calibrator.baseline, history)
                    keys = dispatcher.dispatch(events)

                    for event in events:
                        key = SUBWAY_SURFERS.key_for(event.type)
                        sent = "sent" if key in keys else "not sent"
                        print(
                            f"{event.type.value}  "
                            f"size={event.displacement:.2f}  "
                            f"{event.duration:.2f}s  q={event.quality:.2f}  "
                            f"-> {key.value if key else 'unmapped'} ({sent})"
                        )

                draw_text(
                    frame,
                    format_stats(
                        counter.fps,
                        estimator.latency_ms,
                        pose,
                        state=calibrator.state,
                        progress=calibrator.progress,
                        baseline=calibrator.baseline,
                        history=history,
                        engine=engine,
                        input_status=dispatcher.status,
                    ),
                )
                draw_event_flash(frame, engine.last_event, now)
                cv2.imshow(WINDOW_NAME, frame)

                if (
                    not warned
                    and frames == FPS_WARNING_AFTER_FRAMES
                    and counter.fps is not None
                    and counter.fps < LOW_FPS_WARNING
                ):
                    warned = True
                    print(
                        f"Warning: only {counter.fps:.1f} FPS. Compare it against "
                        "the inference time on screen: if inference is well under "
                        "the frame budget, the bottleneck is lighting, not the model."
                    )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), KEY_ESCAPE):
                    break
                if key == ord("r"):
                    calibrator.reset()
                    smoother.reset()
                    history.clear()
                    engine.reset()
                if key == ord("g"):
                    print(f"Game input {'ENABLED' if dispatcher.toggle() else 'disabled'}")

                # Closing the window with the X button should also stop us.
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            # The camera is released by the `with` block; the OpenCV window is
            # a separate resource with its own cleanup.
            cv2.destroyAllWindows()


def main() -> int:
    """Run the app, turning setup failures into readable messages.

    Returns a process exit code: 0 on success, 1 on failure.
    """
    try:
        run()
    except (CameraError, PoseEstimationError) as error:
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
