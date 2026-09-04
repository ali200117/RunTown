"""Input layer.

The only code that touches the operating system's keyboard. Knows what a key
is; has never heard of a squat.
"""

from kinetirun.input.adapters import InputAdapter, PyAutoGuiAdapter, RecordingAdapter
from kinetirun.input.dispatcher import InputDispatcher

__all__ = [
    "InputAdapter",
    "InputDispatcher",
    "PyAutoGuiAdapter",
    "RecordingAdapter",
]
