# KinetiRun

A full-body computer vision game controller. A regular webcam tracks your body
in real time, and **complete physical movements** — not single body positions —
are turned into game input.

The first demo target is Subway Surfers in the browser, but the movement engine
knows nothing about any specific game. Games plug in at the very end, through a
game profile and an input adapter.

## Core idea

Most webcam game-controller projects do this:

```
camera → a pose landmark crosses a threshold → keypress
```

KinetiRun does not. A movement is a physical action **over time**, validated as
a state machine, and only a fully completed cycle emits an event:

```
READY   → DESCENDING → BOTTOM    → ASCENDING  → COMPLETE   (squat)
GROUNDED→ TAKEOFF    → AIRBORNE  → LANDING    → COMPLETE   (jump)
REST    → SWIPING    → AT_TARGET → RETURNING  → COMPLETE   (arm swipe)
```

Every detector also has a `WAIT_FOR_NEUTRAL` state: after a completed or
rejected attempt you must return to a neutral posture before the same movement
can register again. That removes the need for arbitrary cooldowns — bouncing in
a half-squat produces nothing at all.

Thresholds are **body-relative**, derived from a calibration step, so they mean
the same thing for different body sizes and camera distances:

| Instead of | KinetiRun uses |
|---|---|
| "hips dropped 180 pixels" | "hips dropped 28% of standing hip height" |
| "hand moved 200 pixels" | "arm raised 55 degrees from rest" |

## Movements

| Movement | Gesture | Second signal that prevents cheating |
|---|---|---|
| Squat | Bend the knees and stand back up | Knees must actually bend — bending at the waist is rejected |
| Jump | Jump and land | Feet must leave the ground — rising onto the toes is rejected |
| Left / right | Raise one arm out to the side and lower it | Speed threshold — slow reaching is rejected |

Sideways input has three interchangeable implementations, selected with
`MovementEngine(sideways=...)`. All three emit the same events, so nothing
downstream changes:

- `"arm"` *(default)* — raise an arm. Fast, and the only mode that works with
  the legs out of frame.
- `"lean"` — tilt the whole torso sideways. No floor space needed.
- `"step"` — a real side step, verified by tracking the feet. The most
  physical, and needs the most room.

## Architecture

```
Webcam → Frame capture → Pose estimation → Body model → Smoothing
       → Calibration → Motion history → Movement engine → Movement event
       → Game profile → Input dispatcher → Game
```

Two rules hold the design together:

**The movement engine never knows about the game.** `SquatDetector` emits
`SQUAT_COMPLETED`. `SubwayProfile` decides that means a down arrow. The input
adapter has never heard of a squat.

**MediaPipe is confined to one module.** `PoseEstimator` is the only file that
imports it; everything above works with our own `BodyPose`. Swapping the pose
backend would touch two files.

A consequence worth stating: **the entire movement engine is testable without a
webcam.** A squat is scripted as a list of depths and pushed through the real
detector. 185 tests run in about a second, and not one of them opens a camera,
loads the model or sends a keystroke.

## Timing is measured in seconds, never in frames

Frame rate on a webcam depends on the lighting — this camera drops from 30 FPS
to 10 in a dim room, because auto-exposure lengthens each frame. Any threshold
expressed in frame counts silently changes meaning when the light does. Motion
history is therefore time-indexed and interpolated: "where were the hips 300 ms
ago" gives the same answer at 10 FPS and at 30.

## Requirements

- Python 3.12 — MediaPipe does not reliably ship wheels for newer versions
- [uv](https://docs.astral.sh/uv/)
- A webcam, and a room where your whole body fits in frame

## Setup

```bash
uv sync
uv run python scripts/download_model.py
uv run python -m kinetirun
```

The second command downloads the MediaPipe pose model (~5 MB) into `models/`.
It is a build input, not source, so it is gitignored rather than committed.

## Using it

1. Stand so your **whole body, including your feet**, is in frame
2. Stand still for three seconds until the overlay reads `Calibrated`
3. Practise the movements with input still off — a large green label flashes
   on every detection
4. Press **G** to enable game input, then click the game window so it has focus

| Key | |
|---|---|
| `Q` / `Esc` | quit |
| `R` | recalibrate |
| `G` | toggle game input |

Game input is **disabled at startup**, on purpose: keystrokes go to whichever
window has focus, so enabling it is always a deliberate act.

## Debug overlay

The overlay is the main development tool. It shows FPS, inference latency, each
detector's state and progress, the body-relative signals being compared against
thresholds, and — most usefully — **why the last attempt was rejected**:

```
SQUAT  WAIT_FOR_NEUTRAL 0%  [too shallow: 0.19 < 0.28]
SIDE   REST 0%              [not raised enough: 41 < 55 deg]
```

That line turns "it doesn't work" into a specific number to change. Every
threshold lives in `src/kinetirun/movement/config.py`.

## Current state

All four movements work and drive the game. The honest limitation: **detection
latency is too high for Subway Surfers at speed.** A movement has to complete
before it counts, which is the entire point of the design, but it means roughly
300–600 ms between starting a gesture and the keypress arriving — fine for a
slower game, marginal for this one.

Reducing it is the next problem, and the options are known: raise the smoothing
alpha, emit on reaching the target rather than on the return, or predict intent
part-way through a movement.

## Configuration

Every threshold is a frozen dataclass in
[`config.py`](src/kinetirun/movement/config.py). Requiring harder squats is a
config change, not a code change:

```python
MovementEngine(squat=SquatConfig(min_depth=0.40))
```

## Project layout

```
src/kinetirun/
    camera/        webcam capture, FPS measurement
    vision/        MediaPipe pose estimation (the only place it is imported)
    tracking/      BodyPose, smoothing, time-indexed motion history
    calibration/   neutral-pose baseline, body-relative ratios
    movement/      state machines, thresholds, movement events
    game/          movement events → game keys
    input/         game keys → keystrokes
    ui/            debug overlay
tests/             185 tests, no webcam required
scripts/           model download, camera diagnostics
```

## Tests

```bash
uv run pytest
```
