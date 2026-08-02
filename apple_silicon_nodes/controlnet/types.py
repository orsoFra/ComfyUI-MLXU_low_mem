"""ControlNet Union types and constants.

Supports 8 control types:
  pose, depth, soft_edge, line_canny, normal, segment, tile, repaint
"""

from __future__ import annotations

CONTROL_NET_TYPES: dict[str, int] = {
    "pose": 0,
    "depth": 1,
    "soft_edge": 2,
    "line_canny": 3,
    "normal": 4,
    "segment": 5,
    "tile": 6,
    "repaint": 7,
}
