"""
Feature 7 — Touch Detection & Touch Map

Phase L3: Detects when the target player touches the ball and records
          the real-world coordinate of each touch.

Logic:
  A touch is registered when ALL three conditions hold:
    1. Ball is detected (is_estimated == False)
    2. Euclidean pixel distance(ball_centre, player_foot) < PROXIMITY_PX
    3. Condition held for ≥ MIN_CONSECUTIVE frames (rejects single-frame noise)

After the full clip is processed, touch locations (in real-world metres) are
rendered as markers on a top-down pitch template and statistics are reported.

Usage:
    from touch_detection import TouchDetector
    td = TouchDetector(proximity_px=50)

    # Per-frame call
    td.update(
        frame_id      = 42,
        ball_result   = {ball_px: ..., ball_py: ..., is_estimated: False},
        player_bbox   = (x, y, w, h),          # target player bbox in pixels
        player_world  = (wx, wy),               # real-world coord from homography
    )

    # After clip
    stats    = td.get_stats()
    td.render_touch_map("touch_map.png")
"""

import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ─── Pitch constants (FIFA standard) ─────────────────────────────────────────
PITCH_W_M = 105.0   # metres
PITCH_H_M = 68.0    # metres

# ─── Touch Detector ───────────────────────────────────────────────────────────

class TouchDetector:
    """
    L3 touch detection.

    Parameters
    ----------
    lower_fraction   : the bottom fraction of the bbox considered the
                       "foot zone" (default 0.5 = lower half). Ball must
                       be in this region to count as a touch.
    margin_px        : how many pixels outside the bbox the ball can be
                       and still count (expands the box on all sides).
    min_consecutive  : minimum consecutive frames the condition must hold
    zones            : custom zone dict; defaults to the 9-zone grid defined
                       in the project spec. Keys = zone name, Values = (x_min,
                       x_max, y_min, y_max) in METRES on the pitch.
    """

    # Default 9-zone grid on a 105 × 68 m pitch
    # Left/Right is relative to the attacking direction (positive x = attack)
    DEFAULT_ZONES = {
        "Defensive Third":   (0,    35,   0,    68),
        "Middle Third":      (35,   70,   0,    68),
        "Final Third":       (70,  105,   0,    68),
        "Left Flank":        (0,   105,   0,    17),
        "Left Half Space":   (0,   105,  17,    27),
        "Centre":            (0,   105,  27,    41),
        "Right Half Space":  (0,   105,  41,    51),
        "Right Flank":       (0,   105,  51,    68),
        "Advanced Wide":     (70,  105,   0,    68),
    }

    def __init__(
        self,
        lower_fraction: float  = 0.5,
        margin_px: float       = 25.0,
        min_consecutive: int   = 2,
        zones: dict | None     = None,
    ):
        self.lower_fraction  = lower_fraction
        self.margin_px       = margin_px
        self.min_consecutive = min_consecutive
        self.zones           = zones or self.DEFAULT_ZONES

        # Internal state
        self._proximity_run  = 0           # consecutive-proximity counter
        self._last_touch_frame = -999      # frame ID of last registered touch

        # Accumulated results
        self.touches: list[dict] = []      # [{frame_id, touch_px, touch_py, touch_wx, touch_wy}]

    # ── Per-frame update ───────────────────────────────────────────────────

    def update(
        self,
        frame_id:    int,
        ball_result: dict,
        player_bbox: tuple[int, int, int, int],  # (x, y, w, h)
        player_world: Optional[tuple[float, float]] = None,
    ) -> bool:
        """
        Process one frame. Returns True if a new touch was registered.

        Parameters
        ----------
        frame_id      : current frame number
        ball_result   : dict from BallDetector.track_ball()
        player_bbox   : (x, y, w, h) of the target player in pixels
        player_world  : (wx, wy) in metres (from homography); can be None if
                        homography is not yet available (touch is still logged
                        at pixel level only).
        """
        # Condition 1: ball must be a real detection (not Kalman estimate)
        if ball_result["ball_px"] is None or ball_result["is_estimated"]:
            self._proximity_run = 0
            return False

        x, y, w, h = player_bbox
        ball_px    = ball_result["ball_px"]
        ball_py    = ball_result["ball_py"]
        m          = self.margin_px

        # Expanded bbox bounds (margin on all sides)
        box_x0 = x - m
        box_x1 = x + w + m
        box_y1 = y + h + m   # bottom edge (expanded down)

        # Foot zone: only the lower `lower_fraction` of the bbox height,
        # expanded by margin. Ball above this (near the head) is not a touch.
        foot_zone_top = y + h * (1.0 - self.lower_fraction) - m

        # Condition 2: ball inside the expanded bbox AND in the foot zone
        in_box       = box_x0 <= ball_px <= box_x1 and ball_py <= box_y1
        in_foot_zone = ball_py >= foot_zone_top

        if in_box and in_foot_zone:
            self._proximity_run += 1
        else:
            self._proximity_run = 0
            return False

        # Condition 3: sustained for min_consecutive frames AND not same touch
        if (self._proximity_run == self.min_consecutive and
                frame_id - self._last_touch_frame > self.min_consecutive):

            self._last_touch_frame = frame_id
            wx, wy = player_world if player_world else (None, None)
            dist   = np.hypot(ball_px - (x + w / 2), ball_py - (y + h))

            self.touches.append({
                "frame_id":     frame_id,
                "touch_px":     int(x + w / 2),
                "touch_py":     int(y + h),
                "touch_wx":     wx,
                "touch_wy":     wy,
                "ball_dist_px": dist,
            })
            return True

        return False

    # ── Statistics ─────────────────────────────────────────────────────────

    def get_stats(self, fps: float = 25.0, total_frames: int = 0) -> dict:
        """
        Returns summary statistics for all recorded touches.

        Parameters
        ----------
        fps           : frame rate of the clip
        total_frames  : total number of processed frames in the clip (used for
                        touches_per_minute). If 0 or not provided, falls back
                        to the last touch frame as a rough estimate.

        Returns
        -------
        {
            touch_count        : int,
            touch_locations    : list of (wx, wy),
            touches_by_zone    : {zone_name: count},
            touches_per_minute : float,
            touches            : full list of touch dicts
        }
        """
        count = len(self.touches)

        # World-coordinate touches (only those with homography data)
        world_locs = [
            (t["touch_wx"], t["touch_wy"])
            for t in self.touches
            if t["touch_wx"] is not None
        ]

        # Zone occupancy for touches
        zone_counts = defaultdict(int)
        for wx, wy in world_locs:
            zone = self._classify_zone(wx, wy)
            zone_counts[zone] += 1

        # Duration: prefer explicit total_frames, fall back to last touch frame
        ref_frame = total_frames if total_frames > 0 else (
            max(t["frame_id"] for t in self.touches) if self.touches else 0
        )
        duration_min = ref_frame / fps / 60.0
        tpm = count / duration_min if duration_min > 0 else 0.0

        return {
            "touch_count":         count,
            "touch_locations":     world_locs,
            "touches_by_zone":     dict(zone_counts),
            "touches_per_minute":  round(tpm, 2),
            "touches":             self.touches,
        }

    def _classify_zone(self, wx: float, wy: float) -> str:
        for zone_name, (x0, x1, y0, y1) in self.zones.items():
            if x0 <= wx <= x1 and y0 <= wy <= y1:
                return zone_name
        return "Unknown"

    # ── Touch Map Rendering ────────────────────────────────────────────────

    def render_touch_map(
        self,
        output_path: str,
        heatmap_array: np.ndarray | None = None,
        pitch_img_path: str | None = None,
        img_w: int = 1050,
        img_h: int = 680,
    ):
        """
        Render touch locations as markers on a top-down pitch image.

        Parameters
        ----------
        output_path    : where to save the output PNG
        heatmap_array  : optional pre-computed heatmap (from Zain's Z7) to
                         blend behind the touch markers
        pitch_img_path : optional path to a custom pitch template image
        img_w, img_h   : output image dimensions in pixels (default 1050×680
                         = 1px per 10 cm at FIFA standard dimensions)
        """
        # ── Build or load pitch canvas ─────────────────────────────────
        if pitch_img_path and Path(pitch_img_path).exists():
            canvas = cv2.imread(str(pitch_img_path))
            canvas = cv2.resize(canvas, (img_w, img_h))
        else:
            canvas = _draw_pitch_template(img_w, img_h)

        # ── Blend in heatmap if provided ───────────────────────────────
        if heatmap_array is not None:
            hm_resized = cv2.resize(heatmap_array, (img_w, img_h))
            if len(hm_resized.shape) == 2:
                hm_colour = cv2.applyColorMap(
                    (hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            else:
                hm_colour = hm_resized
            canvas = cv2.addWeighted(canvas, 0.6, hm_colour, 0.4, 0)

        # ── Scale factors: metres → pixels ─────────────────────────────
        sx = img_w / PITCH_W_M
        sy = img_h / PITCH_H_M

        # ── Draw each touch marker ─────────────────────────────────────
        for i, touch in enumerate(self.touches):
            wx, wy = touch["touch_wx"], touch["touch_wy"]
            if wx is None or wy is None:
                continue

            px = int(wx * sx)
            py = int(wy * sy)

            # Outer ring
            cv2.circle(canvas, (px, py), 10, (0, 255, 255), 2)
            # Inner dot
            cv2.circle(canvas, (px, py), 4, (0, 200, 200), -1)
            # Touch number
            cv2.putText(canvas, str(i + 1), (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── Stats overlay ──────────────────────────────────────────────
        stats = self.get_stats()
        overlay_lines = [
            f"Total Touches: {stats['touch_count']}",
            f"Touches/min: {stats['touches_per_minute']}",
        ]
        for li, line in enumerate(overlay_lines):
            cv2.putText(canvas, line, (15, 30 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imwrite(output_path, canvas)
        print(f"[TouchDetector] Touch map saved → {output_path}")
        return canvas


# ─── Pitch template helper ────────────────────────────────────────────────────

def _draw_pitch_template(w: int, h: int) -> np.ndarray:
    """Draw a minimal top-down football pitch in dark green."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (34, 85, 34)   # dark grass green

    lc  = (255, 255, 255)   # line colour
    lw  = 2                 # line width
    sx  = w / PITCH_W_M
    sy  = h / PITCH_H_M

    def m2p(xm, ym):
        return (int(xm * sx), int(ym * sy))

    # Pitch outline
    cv2.rectangle(img, m2p(0, 0), m2p(PITCH_W_M, PITCH_H_M), lc, lw)

    # Halfway line
    cv2.line(img, m2p(PITCH_W_M / 2, 0), m2p(PITCH_W_M / 2, PITCH_H_M), lc, lw)

    # Centre circle (r = 9.15 m)
    cx, cy = m2p(PITCH_W_M / 2, PITCH_H_M / 2)
    cv2.circle(img, (cx, cy), int(9.15 * sx), lc, lw)
    cv2.circle(img, (cx, cy), 4, lc, -1)

    # Penalty areas (16.5 m × 40.32 m)
    pa_depth = 16.5
    pa_half  = 20.16
    mid_y    = PITCH_H_M / 2
    # Left penalty area
    cv2.rectangle(img, m2p(0, mid_y - pa_half), m2p(pa_depth, mid_y + pa_half), lc, lw)
    # Right penalty area
    cv2.rectangle(img,
                  m2p(PITCH_W_M - pa_depth, mid_y - pa_half),
                  m2p(PITCH_W_M,            mid_y + pa_half), lc, lw)

    # Goal boxes (5.5 m × 18.32 m)
    gb_depth = 5.5
    gb_half  = 9.16
    cv2.rectangle(img, m2p(0, mid_y - gb_half), m2p(gb_depth, mid_y + gb_half), lc, lw)
    cv2.rectangle(img,
                  m2p(PITCH_W_M - gb_depth, mid_y - gb_half),
                  m2p(PITCH_W_M,             mid_y + gb_half), lc, lw)

    return img


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    print("=== TouchDetector self-test with mock data ===")

    td = TouchDetector(lower_fraction=0.5, margin_px=25, min_consecutive=2)

    # Simulate 200 frames with random ball + player positions
    # bbox: x=600, y=300, w=50, h=100  → lower half starts at y=350, foot at y=400
    # For a touch: ball must be inside x=[575,675], y=[325,425] AND y>=325
    random.seed(42)
    for fid in range(1, 201):
        # Fake player bbox (stationary for simplicity)
        bbox = (600, 300, 50, 100)

        # Fake ball: ~15% of frames place it inside the lower bbox region
        near = random.random() < 0.15
        bpx  = 620 + random.uniform(-20, 20) if near else random.uniform(0, 1280)
        bpy  = 375 + random.uniform(-20, 20) if near else random.uniform(0, 720)

        ball_result = {
            "ball_px":    bpx,
            "ball_py":    bpy,
            "confidence": 0.8,
            "is_estimated": False,
        }
        # Fake world coords (just scale from pixel for test)
        world = (bpx / 1280 * PITCH_W_M, bpy / 720 * PITCH_H_M)

        touched = td.update(fid, ball_result, bbox, world)
        if touched:
            print(f"  Touch at frame {fid}!")

    stats = td.get_stats()
    print(f"\nTotal touches : {stats['touch_count']}")
    print(f"Touches/min   : {stats['touches_per_minute']}")
    print(f"By zone       : {stats['touches_by_zone']}")

    td.render_touch_map("touch_map_test.png")
    print("Self-test complete.")