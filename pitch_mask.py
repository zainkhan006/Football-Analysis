"""
Pitch-interior mask: a tight binary mask of the green football-pitch region.

Used to gate the line-mask / Hough so we don't pick up ad-board, scoreboard,
crowd, or stadium edges as fake pitch lines.

Pipeline
--------
1. HSV-gate broadly for "green-ish" pixels (loose threshold).
2. Morphological close to fill the white-line gaps inside the pitch.
3. Keep the largest connected component (the pitch itself).
4. Fill its interior holes.
5. Optionally erode a few pixels so the mask doesn't include the pitch
   boundary line that touches the ad-board.

Public API
----------
    build_pitch_mask(frame_bgr) -> np.ndarray   # uint8 {0, 255}
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def build_pitch_mask(
    frame_bgr: np.ndarray,
    green_hsv_lo: Tuple[int, int, int] = (35, 60, 30),
    green_hsv_hi: Tuple[int, int, int] = (85, 255, 220),
    close_px: int = 0,
    erode_px: int = 5,
) -> np.ndarray:
    """
    Return a uint8 {0, 255} mask of the pitch-interior region.

    Strategy:
      1. Tight HSV green gate (high saturation required) so ad-boards with
         dim green backgrounds don't pass.
      2. Morph-close to fill in white-line gaps.
      3. Keep the largest connected component that *touches the bottom edge*
         (the pitch always reaches the bottom in broadcast football).
         This prevents stadium roof / crowd green pixels from being picked.
      4. Floodfill internal holes.
      5. Light erosion so the mask edge sits inside the pitch, not on the
         pitch-to-ad-board boundary.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array(green_hsv_lo, dtype=np.uint8),
        np.array(green_hsv_hi, dtype=np.uint8),
    )

    # Optionally close to fill the white-line gaps inside the pitch. Keep
    # this kernel small: large kernels bridge the green ad-board banners
    # above the pitch, which we specifically want to exclude.
    if close_px > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
        closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k_close)
    else:
        closed = green

    # Connected components -- pitch is the largest CC that touches bottom row.
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_lbl <= 1:
        return np.zeros_like(green)
    h, w = closed.shape
    bottom_row = lbl[h - 1, :]
    bottom_labels = np.unique(bottom_row)
    bottom_labels = bottom_labels[bottom_labels > 0]
    if len(bottom_labels) == 0:
        # Fallback: just largest CC overall
        areas = stats[1:, cv2.CC_STAT_AREA]
        chosen = 1 + int(np.argmax(areas))
    else:
        # Pick the one with the largest area among bottom-touching CCs.
        bl_areas = [(int(lab), int(stats[lab, cv2.CC_STAT_AREA]))
                    for lab in bottom_labels]
        chosen = max(bl_areas, key=lambda x: x[1])[0]
    pitch = np.where(lbl == chosen, 255, 0).astype(np.uint8)

    # Fill interior holes via floodfill from outside.
    ff = pitch.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    pitch = cv2.bitwise_or(pitch, holes)

    if erode_px > 0:
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_px, erode_px))
        pitch = cv2.erode(pitch, k_erode)

    return pitch
