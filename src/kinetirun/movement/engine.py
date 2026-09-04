"""The movement engine.

Owns the detectors and runs them all against every frame. Phase 10's job is to
give the rest of the system ONE thing to talk to: callers feed poses in and get
MovementEvents out, without knowing how many detectors exist or how any of them
works.

The detectors stay completely independent of each other. A squat and a jump can
in principle be mid-flight at the same time, and neither is allowed to suppress
the other - if that turns out to cause false positives in practice, arbitration
belongs here, not inside a detector.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from kinetirun.calibration import Baseline
from kinetirun.movement.arm import ArmDetector
from kinetirun.movement.config import (
    ArmConfig,
    JumpConfig,
    LeanConfig,
    SideConfig,
    SquatConfig,
)
from kinetirun.movement.events import MovementEvent
from kinetirun.movement.jump import JumpDetector
from kinetirun.movement.lean import LeanDetector
from kinetirun.movement.side import SideDetector
from kinetirun.movement.squat import SquatDetector
from kinetirun.tracking import BodyPose, MotionHistory


class MovementEngine:
    """Runs every movement detector and collects their events."""

    def __init__(
        self,
        squat: Optional[SquatConfig] = None,
        side: Optional[SideConfig] = None,
        jump: Optional[JumpConfig] = None,
        lean: Optional[LeanConfig] = None,
        arm: Optional[ArmConfig] = None,
        sideways: str = "arm",
    ) -> None:
        """
        Args:
            sideways: "arm", "lean" or "step". All three produce the same
                SIDE_LEFT / SIDE_RIGHT events, so nothing downstream is
                affected by the choice.

                  step - a real sideways displacement with the feet. The most
                         physical, and needs the most floor space.
                  lean - a whole-body sideways tilt. No space needed, but slow
                         to perform and still needs the lower body in frame.
                  arm  - sweep an arm out and back. Fast enough to react with
                         in a game, and the only one that works with the legs
                         out of frame.

                Arm is the default: it is the most reliable to track and the
                quickest to react with, which matters more in practice than
                the extra physical effort of the other two.
        """
        if sideways not in ("arm", "lean", "step"):
            raise ValueError(
                f'sideways must be "arm", "lean" or "step", got {sideways!r}'
            )

        self.sideways = sideways
        self.squat = SquatDetector(squat)
        self.side = {
            "arm": lambda: ArmDetector(arm),
            "lean": lambda: LeanDetector(lean),
            "step": lambda: SideDetector(side),
        }[sideways]()
        self.jump = JumpDetector(jump)
        self._counts: Dict[str, int] = {}
        self._last_event: Optional[MovementEvent] = None

    @property
    def detectors(self):
        return (self.squat, self.side, self.jump)

    def update(
        self,
        pose: Optional[BodyPose],
        baseline: Baseline,
        history: MotionHistory,
    ) -> List[MovementEvent]:
        """Feed one frame. Returns every movement completed on this frame.

        A list rather than a single event: nothing guarantees that two
        detectors cannot complete on the same frame, and silently dropping one
        would be a bug that only shows up as an occasional missed input.
        """
        events = []
        for detector in self.detectors:
            event = detector.update(pose, baseline, history)
            if event is not None:
                events.append(event)
                self._counts[event.type.value] = self._counts.get(event.type.value, 0) + 1
                self._last_event = event
        return events

    def reset(self) -> None:
        """Reset the detectors. Counts and history are kept deliberately -
        recalibrating mid-session should not wipe the session's tally."""
        for detector in self.detectors:
            detector.reset()

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    @property
    def last_event(self) -> Optional[MovementEvent]:
        return self._last_event

    @property
    def rejections(self) -> Dict[str, Optional[str]]:
        """Why each detector last refused to emit. The tuning readout."""
        return {
            "SQUAT": self.squat.last_rejection,
            "SIDE": self.side.last_rejection,
            "JUMP": self.jump.last_rejection,
        }

    @property
    def states(self) -> Dict[str, tuple[str, float]]:
        """Each detector's state and progress, for the debug overlay."""
        return {
            "SQUAT": (self.squat.state.value, self.squat.progress),
            "SIDE": (self.side.state.value, self.side.progress),
            "JUMP": (self.jump.state.value, self.jump.progress),
        }
