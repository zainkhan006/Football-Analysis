"""
Multi-player touch assignment from per-frame ball detections + player tracks.

State machine
-------------
For every frame we know:
    - ball pixel location (or None if undetected)
    - a list of (player_id, bbox_pixel) tuples for tracked players

We compute the player whose foot-point (bottom-centre of bbox) is closest
to the ball. If that closest distance, scaled by the bbox height (a proxy
for how big a player should appear at that depth in the scene), is below a
threshold, that player is the "candidate possessor" for this frame.

A TOUCH is emitted on the FIRST frame of a confirmed new possession run:
    - "Confirmed" = the candidate possessor remains the same for at least
      `min_hold_frames` consecutive frames (debouncing).
    - "New run" = either the previous frame had no possessor (ball in
      flight) or the previous possessor was a different player.
    - Per-player cooldown: the same player can't be credited a second
      touch within `cooldown_frames` frames after their last touch
      (handles a player jogging with the ball whose distance jitters
      across the threshold).

Foot-zone gate
--------------
Only balls in the bottom `foot_zone_fraction` of a player's bbox are
considered touches. A ball near the head/torso is rejected even if it
is close in pixel distance. This prevents crediting aerial duels or
shots flying past the upper body.

Why foot-point rather than centroid?
------------------------------------
The ball is on the ground. A player's bbox centroid sits around chest
height. Their feet are at the bottom-centre of the bbox, which is where
ball contact actually happens.

Why bbox-height-scaled threshold?
---------------------------------
A near-camera player has a ~200 px tall bbox; a far-camera player has a
~50 px tall bbox. A constant pixel threshold would either miss touches
in the far field or double-count in the near field. Using bbox height as
the unit normalises the threshold across the depth of the scene.

Output
------
Each `update()` call returns at most one `TouchEvent` (when a new
possession is confirmed on this frame). The full list is also kept in
`self.events`. Per-player counts are in `self.touches_per_player`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlayerBox:
    pid: int
    x: int
    y: int
    w: int
    h: int

    @property
    def foot_xy(self) -> Tuple[float, float]:
        """Bottom-centre of the bbox in pixel space."""
        return (self.x + self.w / 2.0, self.y + self.h)


@dataclass(frozen=True)
class TouchEvent:
    frame_id:    int
    player_id:   int
    ball_px:     float
    ball_py:     float
    world_xy:    Optional[Tuple[float, float]]   # metres on pitch, or None
    zone:        Optional[str]
    h_supported: bool                             # was world coord inside H inlier bbox?
    foot_dist_px: float
    bbox_h_px:   int                              # bbox height of the possessor


# ──────────────────────────────────────────────────────────────────────────────
#  Assigner
# ──────────────────────────────────────────────────────────────────────────────

class MultiPlayerTouchAssigner:
    """
    Multi-player touch / possession state machine.

    Parameters
    ----------
    proximity_factor
        Threshold for "ball is close enough to player" expressed in
        units of the player's bbox height. e.g. 1.0 means the ball must
        be within `1 * bbox_height` pixels of the player's foot point.
        Increase if the detector tends to miss touches; decrease if it
        over-credits touches to nearby players.
    min_hold_frames
        Number of consecutive frames a candidate possessor must remain
        the same before they're confirmed as the new possessor. Filters
        out single-frame jitter.
    cooldown_frames
        After a touch is emitted for player P, no further touches can
        be emitted for P until this many frames have elapsed. Prevents
        double-counting when the ball stays at one player's feet.
    foot_zone_fraction
        Only the bottom this-fraction of a player's bbox is considered
        the foot zone. Ball must be vertically inside this zone to
        count as a touch. Default 0.4 (bottom 40%). Set to 1.0 to
        disable the gate.
    foot_zone_margin_px
        Extra pixels of vertical slack below the bbox bottom edge,
        so a ball slightly beneath the visible foot still registers.
    """

    def __init__(
        self,
        proximity_factor:    float = 1.0,
        min_hold_frames:     int   = 3,
        cooldown_frames:     int   = 8,
        foot_zone_fraction:  float = 0.4,
        foot_zone_margin_px: float = 15.0,
    ):
        self.proximity_factor = float(proximity_factor)
        self.min_hold_frames     = int(min_hold_frames)
        self.cooldown_frames     = int(cooldown_frames)
        self.foot_zone_fraction  = float(foot_zone_fraction)
        self.foot_zone_margin_px = float(foot_zone_margin_px)

        # State
        self._current_possessor: Optional[int] = None       # confirmed
        self._candidate:         Optional[int] = None       # awaiting confirmation
        self._candidate_run:     int           = 0
        self._frames_no_candidate: int         = 0
        self._last_touch_frame:  dict[int, int] = {}

        # Outputs
        self.events: list[TouchEvent] = []
        self.touches_per_player: dict[int, int] = defaultdict(int)
        # Frame-by-frame possessor log (for diagnostics / overlay).
        # Keys are frame_id, values are confirmed_possessor (int) or None.
        self.possessor_per_frame: dict[int, Optional[int]] = {}

    # ── Per-frame update ────────────────────────────────────────────────

    def update(
        self,
        frame_id: int,
        ball_xy:  Optional[Tuple[float, float]],
        players:  Sequence[PlayerBox],
        homography=None,
    ) -> Optional[TouchEvent]:
        """
        Process one frame. Returns a TouchEvent if a new possession was
        confirmed on this frame, else None.

        ball_xy
            (px, py) of the ball centre, or None if ball not detected.
        players
            Sequence of PlayerBox for all tracked players in this frame.
        homography
            Optional `PitchHomography`. If provided, the touch's
            world coords + zone label are filled in.
        """
        # ── 1. Ball not detected → no possessor this frame
        if ball_xy is None:
            self._reset_candidate()
            self._tick_no_candidate()
            self.possessor_per_frame[frame_id] = self._current_possessor
            return None

        if not players:
            self._reset_candidate()
            self._tick_no_candidate()
            self.possessor_per_frame[frame_id] = self._current_possessor
            return None

        # ── 2. Find closest player (foot-point distance, normalised by bbox height)
        #        PLUS foot-zone vertical gate: ball must be in the bottom
        #        `foot_zone_fraction` of the player's bbox to count.
        bx, by = ball_xy
        best_pid:    Optional[int]   = None
        best_norm:   float           = float("inf")
        best_dist:   float           = float("inf")
        best_bbox_h: int             = 0
        for pb in players:
            if pb.h <= 0:
                continue

            # Foot-zone gate: the ball's y must be >= the top of the foot zone
            # foot_zone_top = bbox_bottom - foot_zone_fraction * bbox_height
            foot_zone_top = pb.y + pb.h * (1.0 - self.foot_zone_fraction)
            foot_zone_bot = pb.y + pb.h + self.foot_zone_margin_px
            if by < foot_zone_top or by > foot_zone_bot:
                continue  # ball is above the foot zone or way below — skip

            # Also check horizontal containment (with some margin)
            horiz_margin = pb.w * 0.3  # 30% width margin on each side
            if bx < pb.x - horiz_margin or bx > pb.x + pb.w + horiz_margin:
                continue  # ball is too far left/right of this player

            fx, fy = pb.foot_xy
            d = float(np.hypot(bx - fx, by - fy))
            norm = d / float(pb.h)          # how many bbox-heights away
            if norm < best_norm:
                best_norm   = norm
                best_dist   = d
                best_pid    = pb.pid
                best_bbox_h = pb.h

        # ── 3. Candidate possessor for this frame (if close enough)
        candidate = best_pid if best_norm <= self.proximity_factor else None

        if candidate is None:
            # Ball in flight / between players.
            self._reset_candidate()
            self._tick_no_candidate()
            self.possessor_per_frame[frame_id] = self._current_possessor
            return None

        # We have a candidate this frame, so reset the no-candidate counter.
        self._frames_no_candidate = 0

        # ── 4. Update candidate run length
        if candidate == self._candidate:
            self._candidate_run += 1
        else:
            self._candidate     = candidate
            self._candidate_run = 1

        # ── 5. Confirm and possibly emit a touch
        emitted: Optional[TouchEvent] = None
        if (self._candidate_run >= self.min_hold_frames
            and candidate != self._current_possessor):

            # Cooldown check (per-player)
            last = self._last_touch_frame.get(candidate, -10**9)
            if frame_id - last >= self.cooldown_frames:
                emitted = self._make_event(
                    frame_id   = frame_id,
                    player_id  = candidate,
                    ball_px    = bx,
                    ball_py    = by,
                    foot_dist  = best_dist,
                    bbox_h     = best_bbox_h,
                    homography = homography,
                )
                self.events.append(emitted)
                self.touches_per_player[candidate] += 1
                self._last_touch_frame[candidate]  = frame_id

            # Whether or not a touch was emitted (cooldown), mark as
            # confirmed possessor so subsequent frames don't keep
            # re-confirming the same person.
            self._current_possessor = candidate

        self.possessor_per_frame[frame_id] = self._current_possessor
        return emitted

    # ── Helpers ─────────────────────────────────────────────────────────

    def _reset_candidate(self) -> None:
        self._candidate     = None
        self._candidate_run = 0

    def _tick_no_candidate(self) -> None:
        """
        Track how many consecutive frames have had no candidate possessor.
        Once that exceeds `cooldown_frames`, drop the current possessor
        so that if the same player gets the ball back later it's counted
        as a NEW touch (e.g. A kicks ball, ball bounces back, A controls
        it again -> two touches, not one).
        """
        self._frames_no_candidate += 1
        if self._frames_no_candidate >= self.cooldown_frames:
            self._current_possessor = None

    def _make_event(
        self,
        frame_id:   int,
        player_id:  int,
        ball_px:    float,
        ball_py:    float,
        foot_dist:  float,
        bbox_h:     int,
        homography,
    ) -> TouchEvent:
        world_xy:    Optional[Tuple[float, float]] = None
        zone:        Optional[str]                 = None
        h_supported: bool                          = False

        if homography is not None:
            try:
                wx, wy = homography.pixel_to_world(ball_px, ball_py)
                world_xy = (float(wx), float(wy))
                # Use the project's canonical zoning if available.
                try:
                    from pitch_config import world_xy_to_zone
                    zone = world_xy_to_zone(wx, wy)
                except Exception:
                    zone = None
                # Honest "is this projection backed by evidence?" check.
                if hasattr(homography, "is_within_support"):
                    h_supported = bool(homography.is_within_support(wx, wy))
                else:
                    h_supported = True
            except Exception:
                pass

        return TouchEvent(
            frame_id     = frame_id,
            player_id    = player_id,
            ball_px      = ball_px,
            ball_py      = ball_py,
            world_xy     = world_xy,
            zone         = zone,
            h_supported  = h_supported,
            foot_dist_px = foot_dist,
            bbox_h_px    = bbox_h,
        )

    # ── Reporting ───────────────────────────────────────────────────────

    def summary_table(self) -> list[dict]:
        """
        Per-player aggregate. Returns list of dicts sorted by touch_count desc.
        """
        zones_per_pid: dict[int, set] = defaultdict(set)
        first_per_pid: dict[int, int] = {}
        last_per_pid:  dict[int, int] = {}
        approx_per_pid: dict[int, int] = defaultdict(int)
        for ev in self.events:
            if ev.zone:
                zones_per_pid[ev.player_id].add(ev.zone)
            first_per_pid.setdefault(ev.player_id, ev.frame_id)
            last_per_pid[ev.player_id] = ev.frame_id
            if not ev.h_supported:
                approx_per_pid[ev.player_id] += 1

        rows = []
        for pid, count in self.touches_per_player.items():
            rows.append({
                "player_id":          pid,
                "touch_count":        count,
                "first_touch_frame":  first_per_pid.get(pid, -1),
                "last_touch_frame":   last_per_pid.get(pid, -1),
                "zones":              ",".join(sorted(zones_per_pid.get(pid, set()))),
                "touches_outside_H_support": approx_per_pid.get(pid, 0),
            })
        rows.sort(key=lambda r: r["touch_count"], reverse=True)
        return rows

    # ── Touch map rendering ────────────────────────────────────────────

    def render_touch_map(
        self,
        output_path: str,
        img_w: int = 1050,
        img_h: int = 680,
    ) -> Optional[np.ndarray]:
        """
        Render per-player touch markers on a top-down pitch template.
        Each player gets a unique colour. Returns the canvas (or None
        if there are no world-coordinate touches).
        """
        PITCH_W_M = 120.0  # match pitch_config
        PITCH_H_M = 70.0

        world_events = [e for e in self.events if e.world_xy is not None]
        if not world_events:
            print("[TouchMap] No world-coordinate touches to render.")
            return None

        canvas = _draw_pitch_template(img_w, img_h, PITCH_W_M, PITCH_H_M)
        sx = img_w / PITCH_W_M
        sy = img_h / PITCH_H_M

        # Assign a unique colour per player
        pids = sorted(set(e.player_id for e in world_events))
        palette = _generate_palette(len(pids))
        pid_colour = {pid: palette[i] for i, pid in enumerate(pids)}

        for ev in world_events:
            wx, wy = ev.world_xy
            px = int(wx * sx)
            py = int(wy * sy)
            colour = pid_colour[ev.player_id]

            # Outer ring + inner dot
            cv2.circle(canvas, (px, py), 10, colour, 2)
            cv2.circle(canvas, (px, py), 4, colour, -1)
            # Player ID label
            cv2.putText(canvas, f"#{ev.player_id}",
                        (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Legend in top-left corner
        for i, pid in enumerate(pids):
            y_pos = 30 + i * 22
            colour = pid_colour[pid]
            count = self.touches_per_player.get(pid, 0)
            cv2.circle(canvas, (20, y_pos - 4), 6, colour, -1)
            cv2.putText(canvas, f"Player #{pid}: {count} touches",
                        (32, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imwrite(output_path, canvas)
        print(f"[TouchMap] Saved → {output_path}")
        return canvas


# ──────────────────────────────────────────────────────────────────────────────
#  Pitch template + colour helpers (module-level)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_pitch_template(w: int, h: int,
                         pitch_w_m: float = 120.0,
                         pitch_h_m: float = 70.0) -> np.ndarray:
    """Draw a minimal top-down football pitch in dark green."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (34, 85, 34)   # dark grass green

    lc  = (255, 255, 255)
    lw  = 2
    sx  = w / pitch_w_m
    sy  = h / pitch_h_m

    def m2p(xm, ym):
        return (int(xm * sx), int(ym * sy))

    cv2.rectangle(img, m2p(0, 0), m2p(pitch_w_m, pitch_h_m), lc, lw)
    cv2.line(img, m2p(pitch_w_m / 2, 0), m2p(pitch_w_m / 2, pitch_h_m), lc, lw)

    cx, cy = m2p(pitch_w_m / 2, pitch_h_m / 2)
    cv2.circle(img, (cx, cy), int(9.15 * sx), lc, lw)
    cv2.circle(img, (cx, cy), 4, lc, -1)

    pa_depth = 16.5
    pa_half  = 20.16
    mid_y    = pitch_h_m / 2
    cv2.rectangle(img, m2p(0, mid_y - pa_half), m2p(pa_depth, mid_y + pa_half), lc, lw)
    cv2.rectangle(img, m2p(pitch_w_m - pa_depth, mid_y - pa_half),
                  m2p(pitch_w_m, mid_y + pa_half), lc, lw)

    gb_depth = 5.5
    gb_half  = 9.16
    cv2.rectangle(img, m2p(0, mid_y - gb_half), m2p(gb_depth, mid_y + gb_half), lc, lw)
    cv2.rectangle(img, m2p(pitch_w_m - gb_depth, mid_y - gb_half),
                  m2p(pitch_w_m, mid_y + gb_half), lc, lw)
    return img


def _generate_palette(n: int) -> list[tuple]:
    """Generate n visually distinct BGR colours."""
    if n == 0:
        return []
    colours = []
    for i in range(n):
        hue = int(180 * i / n)  # spread evenly across hue wheel
        hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colours.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colours
