"""Drawing pose and system state onto frames, for debugging.

Everything here is for the human watching the window. It never feeds back into
detection, so it is free to be approximate.
"""

from typing import Optional, Sequence

import cv2
import numpy as np

from kinetirun.calibration import Baseline, CalibrationState
from kinetirun.movement import MovementEngine
from kinetirun.tracking import BodyPose, MotionHistory
from kinetirun.vision.landmarks import POSE_CONNECTIONS, Landmark

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# Below this, MediaPipe is guessing. It ALWAYS returns a coordinate for every
# one of the 33 landmarks - including body parts outside the frame entirely -
# so visibility is the only thing separating a measurement from a fabrication.
VISIBILITY_THRESHOLD = 0.5


def to_pixels(landmark, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized landmark (0..1) to pixel coordinates."""
    return (int(landmark.x * width), int(landmark.y * height))


def draw_skeleton(frame: np.ndarray, landmarks: Sequence) -> None:
    """Draw the skeleton in place.

    Colour encodes confidence, so problems are visible at a glance:
        green  - both endpoints confidently visible
        yellow - one endpoint uncertain
        red    - the model is essentially guessing
    """
    height, width = frame.shape[:2]

    for start_index, end_index in POSE_CONNECTIONS:
        start = landmarks[start_index]
        end = landmarks[end_index]

        confident = sum(
            1 for point in (start, end) if point.visibility >= VISIBILITY_THRESHOLD
        )
        colour = (RED, YELLOW, GREEN)[confident]

        cv2.line(
            frame,
            to_pixels(start, width, height),
            to_pixels(end, width, height),
            colour,
            2,
            cv2.LINE_AA,
        )

    for index in Landmark:
        point = landmarks[index]
        if point.visibility < VISIBILITY_THRESHOLD:
            continue
        cv2.circle(frame, to_pixels(point, width, height), 3, GREEN, -1, cv2.LINE_AA)


def draw_text(
    frame: np.ndarray, lines: Sequence[str], origin: tuple[int, int] = (10, 26)
) -> None:
    """Draw stacked lines of debug text with a black outline.

    The outline matters: plain green text vanishes against a bright wall, and
    webcam footage is not a controlled backdrop.
    """
    x, y = origin
    for offset, line in enumerate(lines):
        position = (x, y + offset * 22)
        cv2.putText(frame, line, position, FONT, 0.55, BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, line, position, FONT, 0.55, GREEN, 1, cv2.LINE_AA)


# How long a completed movement stays on screen, in seconds. Long enough to
# notice in peripheral vision while looking at the game, short enough that two
# quick movements do not blur together.
FLASH_DURATION = 0.6


def draw_event_flash(frame: np.ndarray, event, now: float) -> None:
    """Announce a just-completed movement in large text across the frame.

    The small corner readout is unreadable while actually playing - your eyes
    are on the game. This is big enough to register without looking directly
    at it, which is what makes practising the movements possible at all.
    """
    if event is None:
        return

    age = now - event.timestamp
    if age > FLASH_DURATION:
        return

    label = event.type.value.replace("_COMPLETED", "").replace("_", " ")
    height, width = frame.shape[:2]

    (text_width, text_height), _ = cv2.getTextSize(label, FONT, 1.6, 4)
    origin = ((width - text_width) // 2, height - 40)

    cv2.putText(frame, label, origin, FONT, 1.6, BLACK, 9, cv2.LINE_AA)
    cv2.putText(frame, label, origin, FONT, 1.6, GREEN, 4, cv2.LINE_AA)


def format_stats(
    fps: Optional[float],
    latency_ms: Optional[float],
    pose: Optional[BodyPose],
    state: Optional[CalibrationState] = None,
    progress: float = 0.0,
    baseline: Optional[Baseline] = None,
    history: Optional[MotionHistory] = None,
    engine: Optional[MovementEngine] = None,
    input_status: Optional[str] = None,
) -> list[str]:
    """Build the debug readout shown in the corner of the window."""
    lines = [
        f"FPS: {fps:.1f}" if fps is not None else "FPS: --",
        f"Inference: {latency_ms:.1f} ms" if latency_ms is not None else "Inference: --",
    ]

    if input_status is not None:
        lines.append(f"INPUT: {input_status}   (G to toggle)")

    if state is not None and state is not CalibrationState.COMPLETE:
        lines.append(f"CALIBRATING {state.value}  {progress * 100:.0f}%")
        lines.append("Stand still, facing the camera")

    if pose is None:
        lines.append("Pose: NOT DETECTED")
        return lines

    if baseline is None:
        # Before calibration the useful question is not "how big is the body"
        # but "can the camera actually see the legs" - that is the single
        # reason calibration gets stuck in WAITING.
        lines.append(
            f"Lower-body vis: {pose.lower_body_visibility:.2f}"
            f"{'' if pose.is_reliable() else '  TOO LOW'}"
        )
        if not pose.is_reliable():
            lines.append("Legs/feet not visible - step back, tilt camera down")
        lines.append(f"Shoulder width: {pose.shoulder_width:.3f} m")
        lines.append(
            f"Hip above ankles: {pose.hip_height_above_ankles:.3f} m"
            f"{'  (suspiciously small)' if pose.hip_height_above_ankles < 0.5 else ''}"
        )
        return lines

    # The state machines are the thing worth watching: they show WHY a movement
    # was or was not accepted, which a bare counter cannot.
    if engine is not None:
        rejections = engine.rejections
        for name, (detector_state, detector_progress) in engine.states.items():
            line = f"{name:<6}{detector_state}  {detector_progress * 100:.0f}%"
            # The reason the last attempt failed, right next to the state. This
            # is what turns "it just does not work" into a specific number to
            # change.
            reason = rejections.get(name)
            if reason:
                line += f"  [{reason}]"
            lines.append(line)

        if engine.counts:
            lines.append(
                "  ".join(
                    f"{name.replace('_COMPLETED', '')}={count}"
                    for name, count in sorted(engine.counts.items())
                )
            )
        if engine.last_event is not None:
            last = engine.last_event
            lines.append(
                f"Last: {last.type.value.replace('_COMPLETED', '')}  "
                f"{last.displacement:.2f}  {last.duration:.2f}s  q={last.quality:.2f}"
            )

    # The body-relative signals the detectors actually consume. Which sideways
    # one is shown follows the mode the engine is running in, so the number on
    # screen is always the number being compared against a threshold.
    mode = "arm" if engine is None else engine.sideways
    if mode == "arm":
        sideways = f"arm {baseline.arm_reach(pose):+.0f}deg"
    elif mode == "lean":
        sideways = f"lean {baseline.lean_angle(pose):+.0f}deg"
    else:
        sideways = f"side {baseline.lateral_offset(pose):+.2f}"
    lines.append(
        f"drop {baseline.hip_drop(pose):+.2f}   "
        f"{sideways}   "
        f"up {-baseline.vertical_offset(pose):+.2f}"
    )

    if history is not None:
        lines.append(f"History: {history.span:.1f}s ({len(history)} poses)")

    lines.append(
        f"Lower-body vis: {pose.lower_body_visibility:.2f}"
        f"{'' if pose.is_reliable() else '  UNRELIABLE'}"
    )
    return lines
