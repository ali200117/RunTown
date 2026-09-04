# KinetiRun

A full-body computer vision game controller. A regular webcam tracks your body in
real time, and **complete physical movements** - not single body positions - are
turned into game input.

First demo target is Subway Surfers in the browser, but the movement engine knows
nothing about any specific game. Games are plugged in at the very end via a
game profile and an input adapter.

## Core idea

Most webcam game-controller projects do this:

```
camera -> pose landmark crosses a threshold -> keypress
```

KinetiRun does not. A movement is a physical action **over time**, validated as a
state machine:

```
READY -> DESCENDING -> BOTTOM -> ASCENDING -> COMPLETE   (squat)
CENTER -> MOVING_LEFT -> TARGET -> RETURNING -> COMPLETE (side step)
GROUNDED -> TAKEOFF -> AIRBORNE -> LANDING -> COMPLETE   (jump)
```

Only a fully completed movement emits an event. Thresholds are relative to your
calibrated body size (e.g. `0.7 x shoulder width`), not to raw pixels.

## Architecture (target)

```
Webcam -> Frame Capture -> Pose Estimation -> Body Model -> Smoothing
       -> Calibration -> Motion History -> Movement Engine -> Movement Event
       -> Game Profile -> Input Dispatcher -> Game
```

Hard rule: the movement engine never knows about the game, and the input adapter
never knows what a squat is.

## Status

Phase 1 of 13 complete: Python project foundation.

| Phase | What | Status |
|------:|------|--------|
| 1 | Python foundation | done |
| 2 | Camera foundation | next |
| 3 | Pose estimation (MediaPipe) | |
| 4 | Body model | |
| 5 | Calibration | |
| 6 | Smoothing + motion history | |
| 7 | Squat detector | |
| 8 | Side movement | |
| 9 | Jump | |
| 10 | Movement event system | |
| 11 | Subway game profile | |
| 12 | PyAutoGUI adapter | |
| 13 | Tuning | |

## Requirements

- Python 3.12 (MediaPipe does not reliably ship wheels for newer versions)
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Setup

```bash
uv sync
```

This creates `.venv/`, installs the dependencies from `uv.lock`, and installs
`kinetirun` in editable mode.

## Running the tests

```bash
uv run pytest
```

## Project layout

```
src/kinetirun/    application code
tests/            tests that must run without a webcam
```

Subpackages are added in the phase where they are first needed, not up front.
