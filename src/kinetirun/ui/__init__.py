"""Debug UI.

Draws what the system currently believes onto the video feed. Sits at the top
of the dependency stack: it may import from any layer, and no layer imports it.
Nothing here ever feeds back into detection.
"""

from kinetirun.ui.overlay import draw_event_flash, draw_skeleton, draw_text, format_stats

__all__ = ["draw_event_flash", "draw_skeleton", "draw_text", "format_stats"]
