"""Drawing pose data onto frames, for debugging.

Everything here is for the human watching the window. It never feeds back into
detection, so it is free to be approximate.
"""

from typing import Optional, Sequence

import cv2
import numpy as np

from kinetirun.calibration import Baseline, CalibrationState
from kinetirun.movement import MovementEvent
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
    """Convert a normalized landmark (0..1) to pixel coordinates.

    Normalized means "fraction of the image", so x=0.5 is the horizontal
    centre whatever the resolution. Values can fall slightly outside 0..1 when
    the model extrapolates a joint just off-screen.
    """
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


def draw_text(frame: np.ndarray, lines: Sequence[str], origin: tuple[int, int] = (10, 30)) -> None:
    """Draw stacked lines of debug text with a black outline.

    The outline matters: plain green text vanishes against a bright wall, and
    webcam footage is not a controlled backdrop.
    """
    x, y = origin
    for offset, line in enumerate(lines):
        position = (x, y + offset * 26)
        cv2.putText(frame, line, position, FONT, 0.65, BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, line, position, FONT, 0.65, GREEN, 2, cv2.LINE_AA)


def format_stats(
    fps: Optional[float],
    latency_ms: Optional[float],
    pose: Optional[BodyPose],
    state: Optional[CalibrationState] = None,
    progress: float = 0.0,
    baseline: Optional[Baseline] = None,
    history: Optional[MotionHistory] = None,
    squat_state: Optional[str] = None,
    squat_progress: float = 0.0,
    side_state: Optional[str] = None,
    side_progress: float = 0.0,
    jump_state: Optional[str] = None,
    jump_progress: float = 0.0,
    counts: Optional[dict] = None,
    last_event: Optional[MovementEvent] = None,
) -> list[str]:
    """Build the debug readout shown in the corner of the window.

    Everything below the first two lines comes from BodyPose, not from
    MediaPipe. This readout is how we sanity-check the body model against
    reality before any detector depends on it.
    """
    lines = [
        f"FPS: {fps:.1f}" if fps is not None else "FPS: --",
        f"Inference: {latency_ms:.1f} ms" if latency_ms is not None else "Inference: --",
    ]

    if state is not None:
        if state is CalibrationState.COMPLETE:
            lines.append("Calibrated")
        else:
            lines.append(f"CALIBRATING {state.value}  {progress * 100:.0f}%")
            lines.append("Stand still, facing the camera")

    if pose is None:
        lines.append("Pose: NOT DETECTED")
        return lines

    # Once calibrated, the body-relative ratios matter far more than the raw
    # metres: these are the numbers the movement detectors actually consume.
    if baseline is not None:
        lines.append(f"Hip drop:  {baseline.hip_drop(pose):+.2f}")
        lines.append(f"Sideways:  {baseline.lateral_offset(pose):+.2f} sw")
        lines.append(f"Vertical:  {baseline.vertical_offset(pose):+.2f} sw")
        lines.append(f"Knee angle: {pose.mean_knee_angle:.0f} deg")
        if squat_state is not None:
            # The state machine is the thing worth watching: it shows WHY a
            # movement was or was not accepted, which a bare counter cannot.
            lines.append(f"SQUAT: {squat_state}  {squat_progress * 100:.0f}%")
        if side_state is not None:
            lines.append(f"SIDE:  {side_state}  {side_progress * 100:.0f}%")
        if jump_state is not None:
            lines.append(f"JUMP:  {jump_state}  {jump_progress * 100:.0f}%")
        if counts:
            lines.append("  ".join(
                f"{name.replace('_COMPLETED', '')}={count}"
                for name, count in sorted(counts.items())
            ))
            if last_event is not None:
                lines.append(
                    f"Last: depth {last_event.displacement:.2f}  "
                    f"{last_event.duration:.2f}s  q={last_event.quality:.2f}"
                )

        if history is not None:
            # Rates of change, in body units per second. These are what turn a
            # position into a movement: a detector reads direction and speed
            # from the sign and size of these numbers.
            drop_rate = history.velocity(baseline.hip_drop)
            side_rate = history.velocity(baseline.lateral_offset)
            lines.append(
                f"Drop rate: {drop_rate:+.2f}/s" if drop_rate is not None
                else "Drop rate: --"
            )
            lines.append(
                f"Side rate: {side_rate:+.2f}/s" if side_rate is not None
                else "Side rate: --"
            )
            lines.append(f"History: {history.span:.1f}s ({len(history)} poses)")

        lines.append(
            f"Lower-body vis: {pose.lower_body_visibility:.2f}"
            f"{'' if pose.is_reliable() else '  UNRELIABLE'}"
        )
        return lines

    lines.append(f"Lower-body vis: {pose.lower_body_visibility:.2f}"
                 f"{'' if pose.is_reliable() else '  UNRELIABLE'}")
    # The body-size rulers Phase 5 will calibrate against.
    lines.append(f"Shoulder width: {pose.shoulder_width:.3f} m")
    lines.append(f"Torso height:   {pose.torso_height:.3f} m")
    # The squat signals. Hip height should FALL as you descend; if it rises,
    # the y-axis convention is inverted and every detector would be backwards.
    lines.append(f"Hip above ankles: {pose.hip_height_above_ankles:.3f} m")
    lines.append(f"Knee angle: {pose.mean_knee_angle:.0f} deg")
    # The side-step signal, in image space: world space is anchored to the
    # hips and therefore cannot see the body moving across the room.
    lines.append(f"Hip x (image): {pose.hip_center_image.x:.3f}")

    return lines
