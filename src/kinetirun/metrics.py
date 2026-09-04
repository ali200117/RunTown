"""Small measurement helpers shared across layers."""

from collections import deque
from typing import Deque, Optional


class RollingAverage:
    """Average of the last `window` values, or None before any value arrives.

    Used for durations (inference latency), where FpsCounter's rate maths does
    not apply: latency is measured per event, not derived from the gaps
    between events.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.window = window
        self._values: Deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def value(self) -> Optional[float]:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values.clear()
