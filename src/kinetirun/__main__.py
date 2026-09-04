"""Entry point: `uv run python -m kinetirun`.

Phase 7 milestone - full squat detection with a live state machine readout.
Still no game input: completed squats are printed and shown, nothing else.
"""

import time

import cv2

from kinetirun.camera import Camera, CameraError, FpsCounter
from kinetirun.calibration import CalibrationState, Calibrator
from kinetirun.movement import SquatDetector
from kinetirun.tracking import BodyPose, MotionHistory, PoseSmoother
from kinetirun.ui import draw_skeleton, draw_text, format_stats
from kinetirun.vision import PoseEstimationError, PoseEstimator

WINDOW_NAME = "KinetiRun - squat"

KEY_ESCAPE = 27

# Below this the temporal signals movement detection depends on get too coarse:
# a one-second squat sampled 20 times is workable, sampled 10 times is not.
LOW_FPS_WARNING = 20.0
FPS_WARNING_AFTER_FRAMES = 60


def run() -> None:
    """Main capture and inference loop."""
    counter = FpsCounter()
    calibrator = Calibrator()
    smoother = PoseSmoother()
    history = MotionHistory()
    squats = SquatDetector()
    squat_count = 0
    last_event = None
    frames = 0
    warned = False

    with Camera() as camera, PoseEstimator() as estimator:
        width, height = camera.resolution
        print(f"Camera opened at {width}x{height}. Q quits, R recalibrates.")

        # MediaPipe's VIDEO mode wants a timestamp that starts near zero and
        # only ever increases. Anchoring to the moment we start, rather than to
        # the epoch, keeps the numbers small and readable while debugging.
        start_time = time.perf_counter()

        try:
            while True:
                frame = camera.read()
                now = time.perf_counter()
                counter.tick(now)
                frames += 1

                timestamp_ms = int((now - start_time) * 1000)
                result = estimator.estimate(frame, timestamp_ms)

                # This is the boundary crossing: from here on the loop works
                # with our own type, and the MediaPipe result is discarded.
                pose = BodyPose.from_landmarker_result(result, timestamp=now)
                # Order matters: smooth before anything measures the pose, so
                # every consumer sees the same stable numbers.
                pose = smoother.smooth(pose)
                history.append(pose)

                if pose is not None:
                    draw_skeleton(frame, pose.image)

                calibrator.update(pose)

                # Detection only runs once a baseline exists: every threshold
                # is expressed relative to it.
                if calibrator.baseline is not None:
                    event = squats.update(pose, calibrator.baseline, history)
                    if event is not None:
                        squat_count += 1
                        last_event = event
                        print(
                            f"{event.type.value}  #{squat_count}  "
                            f"depth={event.displacement:.2f}  "
                            f"{event.duration:.2f}s  quality={event.quality:.2f}"
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
                        squat_state=squats.state.value,
                        squat_progress=squats.progress,
                        squat_count=squat_count,
                        last_event=last_event,
                    ),
                )
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
                    squats.reset()

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
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
