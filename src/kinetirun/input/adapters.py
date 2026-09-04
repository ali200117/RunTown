"""Input adapters: the only code that touches the outside world.

An adapter knows how to press a key and nothing else. It has never heard of
squats, jumps, poses or cameras - it receives a GameKey and acts.

That narrowness is what makes the whole pipeline testable: every test in this
project runs against RecordingAdapter, and no test has ever sent a real
keystroke to the machine running it.
"""

from __future__ import annotations

from typing import List, Protocol

from kinetirun.game.profile import GameKey


class InputAdapter(Protocol):
    """Anything that can press a key."""

    def press(self, key: GameKey) -> None: ...


class RecordingAdapter:
    """Records key presses instead of performing them.

    The default, and the one used by every test. Also genuinely useful at
    runtime: it lets the full pipeline run and be watched without a stray
    squat typing into whatever window happens to be focused.
    """

    def __init__(self) -> None:
        self.presses: List[GameKey] = []

    def press(self, key: GameKey) -> None:
        self.presses.append(key)

    def clear(self) -> None:
        self.presses.clear()


class PyAutoGuiAdapter:
    """Sends real keystrokes to whatever window currently has focus.

    Note the risk this carries, which is why nothing enables it implicitly:
    keystrokes go to the FOCUSED window, not to a game specifically. If the
    user alt-tabs to an editor mid-session, their squats start typing there.
    The application layer keeps this disabled until explicitly switched on.
    """

    def __init__(self) -> None:
        # Imported lazily so that the rest of KinetiRun - and its whole test
        # suite - never loads a library whose entire purpose is to move the
        # mouse and press keys on the developer's machine.
        import pyautogui

        # PyAutoGUI's failsafe aborts everything if the mouse is slammed into a
        # screen corner. Keeping it on is worth the occasional surprise: it is
        # the escape hatch when a bug starts spamming keys.
        pyautogui.FAILSAFE = True
        # The default 0.1s pause after every call is dead time we cannot afford
        # in a real-time loop.
        pyautogui.PAUSE = 0.0

        self._pyautogui = pyautogui

    def press(self, key: GameKey) -> None:
        self._pyautogui.press(key.value)
