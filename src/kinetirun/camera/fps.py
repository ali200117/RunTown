"""Frame rate measurement.

Deliberately has no dependency on OpenCV or on any camera, so it can be unit
tested with fake timestamps. In Phase 3 we will reuse this to measure pose
inference latency too - at that point it probably moves out of `camera/`.
"""

from collections import deque
from typing import Deque, Optional


class FpsCounter:
    """Rolling-average frames per second.

    Naively computing `1 / (now - previous)` produces a number that jumps
    around far too much to read. Instead we keep the timestamps of the last
    `window` frames and derive the average rate from them.

    Usage:
        counter = FpsCounter()
        while running:
            counter.tick(time.perf_counter())
            print(counter.fps)
    """

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError(
                f"window must be at least 2 to span an interval, got {window}"
            )
        self.window = window
        self._timestamps: Deque[float] = deque(maxlen=window)

    def tick(self, timestamp: float) -> None:
        """Record that a frame happened at `timestamp`.

        The timestamp is passed in rather than read inside this method. That is
        what makes the class testable: a test can feed a perfectly regular
        sequence of fake times and assert an exact result.

        Always use `time.perf_counter()` as the source - never `time.time()`.
        """
        self._timestamps.append(timestamp)

    @property
    def fps(self) -> Optional[float]:
        """Average frames per second over the window, or None if unknown.

        With N timestamps there are N-1 intervals spanning
        (newest - oldest) seconds, so fps = (N - 1) / span.
        """
        if len(self._timestamps) < 2:
            return None

        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return None

        return (len(self._timestamps) - 1) / span

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()
