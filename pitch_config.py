"""
Canonical 32-keypoint football pitch model.

The Roboflow pitch keypoint detector (`models/football-pitch-detection.pt`)
returns 32 keypoints per frame in a fixed order.  Each one corresponds to a
landmark on a real-world football pitch in metres.

We use FIFA-standard pitch dimensions:  120m x 70m (length x width).

World coordinate system:
    Origin (0, 0) is the bottom-left corner of the pitch.
    +x runs along the goal-line at y=0 (the "length" axis).
    +y runs from one touchline to the other (the "width"  axis).

This module is a port of `roboflow/sports`'s
`sports.configs.soccer.SoccerPitchConfiguration`, converted from cm to m
so downstream code can speak in metres directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ─── FIFA dimensions (metres) ────────────────────────────────────────────────
PITCH_LENGTH_M             = 120.0
PITCH_WIDTH_M              =  70.0
PENALTY_BOX_LENGTH_M       =  20.15
PENALTY_BOX_WIDTH_M        =  41.0
GOAL_BOX_LENGTH_M          =   5.5
GOAL_BOX_WIDTH_M           =  18.32
CENTRE_CIRCLE_RADIUS_M     =   9.15
PENALTY_SPOT_DISTANCE_M    =  11.0


@dataclass(frozen=True)
class PitchConfig:
    """The 32 canonical keypoints in metres, in the order the YOLO model emits."""

    length:                float = PITCH_LENGTH_M
    width:                 float = PITCH_WIDTH_M
    penalty_box_length:    float = PENALTY_BOX_LENGTH_M
    penalty_box_width:     float = PENALTY_BOX_WIDTH_M
    goal_box_length:       float = GOAL_BOX_LENGTH_M
    goal_box_width:        float = GOAL_BOX_WIDTH_M
    centre_circle_radius:  float = CENTRE_CIRCLE_RADIUS_M
    penalty_spot_distance: float = PENALTY_SPOT_DISTANCE_M

    # ── 32 vertices in (x_m, y_m) ────────────────────────────────────────
    @property
    def vertices_m(self) -> List[Tuple[float, float]]:
        L  = self.length
        W  = self.width
        pl = self.penalty_box_length
        pw = self.penalty_box_width
        gl = self.goal_box_length
        gw = self.goal_box_width
        r  = self.centre_circle_radius
        sp = self.penalty_spot_distance
        return [
            (0,                  0),                  # 1  — left-bottom corner
            (0,                  (W - pw) / 2),       # 2  — left penalty-box bottom corner
            (0,                  (W - gw) / 2),       # 3  — left goal-box bottom corner
            (0,                  (W + gw) / 2),       # 4  — left goal-box top corner
            (0,                  (W + pw) / 2),       # 5  — left penalty-box top corner
            (0,                  W),                  # 6  — left-top corner
            (gl,                 (W - gw) / 2),       # 7  — left goal-box bottom inner
            (gl,                 (W + gw) / 2),       # 8  — left goal-box top inner
            (sp,                 W / 2),              # 9  — left penalty spot
            (pl,                 (W - pw) / 2),       # 10 — left penalty-box bottom inner
            (pl,                 (W - gw) / 2),       # 11 — left penalty arc-bottom intersection
            (pl,                 (W + gw) / 2),       # 12 — left penalty arc-top intersection
            (pl,                 (W + pw) / 2),       # 13 — left penalty-box top inner
            (L / 2,              0),                  # 14 — halfway line bottom touchline
            (L / 2,              W / 2 - r),          # 15 — centre-circle bottom
            (L / 2,              W / 2 + r),          # 16 — centre-circle top
            (L / 2,              W),                  # 17 — halfway line top touchline
            (L - pl,             (W - pw) / 2),       # 18 — right penalty-box bottom inner
            (L - pl,             (W - gw) / 2),       # 19 — right penalty arc-bottom intersection
            (L - pl,             (W + gw) / 2),       # 20 — right penalty arc-top intersection
            (L - pl,             (W + pw) / 2),       # 21 — right penalty-box top inner
            (L - sp,             W / 2),              # 22 — right penalty spot
            (L - gl,             (W - gw) / 2),       # 23 — right goal-box bottom inner
            (L - gl,             (W + gw) / 2),       # 24 — right goal-box top inner
            (L,                  0),                  # 25 — right-bottom corner
            (L,                  (W - pw) / 2),       # 26 — right penalty-box bottom corner
            (L,                  (W - gw) / 2),       # 27 — right goal-box bottom corner
            (L,                  (W + gw) / 2),       # 28 — right goal-box top corner
            (L,                  (W + pw) / 2),       # 29 — right penalty-box top corner
            (L,                  W),                  # 30 — right-top corner
            (L / 2 - r,          W / 2),              # 31 — centre-circle left
            (L / 2 + r,          W / 2),              # 32 — centre-circle right
        ]

    # ── Edges between vertices (1-indexed!) — useful for drawing ─────────
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2),  (2, 3),  (3, 4),  (4, 5),  (5, 6),
        (7, 8),
        (10, 11),(11, 12),(12, 13),
        (14, 15),(15, 16),(16, 17),
        (18, 19),(19, 20),(20, 21),
        (23, 24),
        (25, 26),(26, 27),(27, 28),(28, 29),(29, 30),
        # cross-pitch lines
        (1, 14), (2, 10), (3, 7),  (4, 8),  (5, 13), (6, 17),
        (14, 25),(18, 26),(23, 27),(24, 28),(21, 29),(17, 30),
    ])

    # ── Pitch outline (for drawing the field rectangle as a polygon) ────
    @property
    def outline_m(self) -> List[Tuple[float, float]]:
        return [
            (0,             0),
            (self.length,   0),
            (self.length,   self.width),
            (0,             self.width),
            (0,             0),
        ]

    # ── Convenience accessors ────────────────────────────────────────────
    @property
    def num_keypoints(self) -> int:
        return len(self.vertices_m)


# Module-level singleton for convenience.
DEFAULT_PITCH = PitchConfig()


# ─── Functional zone partitioning (used later by touch detection) ────────────

def world_xy_to_zone(wx: float, wy: float,
                     cfg: PitchConfig = DEFAULT_PITCH) -> str:
    """
    Map a real-world (wx, wy) in metres to one of 9 standard tactical zones.

    Returns one of:
        'def_left',  'def_centre',  'def_right',
        'mid_left',  'mid_centre',  'mid_right',
        'att_left',  'att_centre',  'att_right',
        'oob' if outside the pitch.

    `def`/`att` are interpreted from left → right, i.e. the team attacks
    toward x = pitch_length.
    """
    if not (0 <= wx <= cfg.length and 0 <= wy <= cfg.width):
        return "oob"

    # Lengthwise thirds
    if wx < cfg.length / 3:
        third = "def"
    elif wx < 2 * cfg.length / 3:
        third = "mid"
    else:
        third = "att"

    # Widthwise thirds
    if wy < cfg.width / 3:
        side = "right"        # right wing when attacking left→right
    elif wy < 2 * cfg.width / 3:
        side = "centre"
    else:
        side = "left"

    return f"{third}_{side}"
