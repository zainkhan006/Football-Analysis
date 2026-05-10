"""
Distance-transform refinement of a pitch homography using detected pitch lines.

Motivation
----------
YOLO keypoints give us ~8-15 sparse landmarks per frame. Fitting H from those
alone is internally consistent (1-2px reprojection) but extrapolates badly to
pitch regions where the YOLO didn't see keypoints. The visible pitch LINES are
a much denser free signal: every metre of touchline, halfway line, penalty
box, centre circle etc. that shows up white-on-green in the frame is a
constraint on H.

This module:
    1. Extracts a binary line-mask from a frame (green ∩ white, thinned).
    2. Densely samples the canonical pitch edges in world metres.
    3. Optimises the 8 DOF of H so that canonical-pitch samples, when
       projected through H, land as close as possible to detected line
       pixels (measured via a distance transform of the line mask).

The optimisation is LOCAL -- it needs a reasonable seed H (from the YOLO
keypoint fit). Samples that land outside the frame or far from any line are
Huber-capped, so extrapolated pitch regions don't pull the fit.

Public API
----------
    build_line_mask(frame_bgr) -> np.ndarray      # uint8 binary line mask
    sample_canonical_pitch(cfg, step_m) -> np.ndarray  # (N, 2) world xy
    refine_homography(H_init, line_mask,
                      canonical_samples, max_px=30.0) -> (H_refined, info)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from pitch_config import PitchConfig, DEFAULT_PITCH


# ──────────────────────────────────────────────────────────────────────────────
#  Line-mask extraction
# ──────────────────────────────────────────────────────────────────────────────

def build_line_mask(
    frame_bgr: np.ndarray,
    green_hsv_lo: Tuple[int, int, int] = (30, 25, 25),
    green_hsv_hi: Tuple[int, int, int] = (90, 255, 230),
    white_hsv_lo: Tuple[int, int, int] = (0, 0, 170),
    white_hsv_hi: Tuple[int, int, int] = (180, 70, 255),
    green_dilate_px: int = 25,
) -> np.ndarray:
    """
    Return a uint8 {0, 255} mask of likely pitch-line pixels.

    Steps:
      1. Find the green pitch region (broad HSV gate), and dilate it so the
         white lines that sit on top of the grass are included.
      2. Find bright, desaturated pixels → candidate white line pixels.
      3. Intersect (1) ∩ (2) and clean up with a light morph open.

    Player shirts that are white get included, but since we only use this for
    distance-transform matching (not direct correspondence), small amounts of
    noise don't hurt the fit.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array(green_hsv_lo, dtype=np.uint8),
                             np.array(green_hsv_hi, dtype=np.uint8))
    # Dilate green so the white lines on it are inside the region.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (green_dilate_px, green_dilate_px))
    green_dil = cv2.dilate(green, kernel)

    white = cv2.inRange(hsv, np.array(white_hsv_lo, dtype=np.uint8),
                             np.array(white_hsv_hi, dtype=np.uint8))

    lines = cv2.bitwise_and(white, green_dil)

    # Light clean-up: remove isolated specks, keep thin line structure.
    lines = cv2.morphologyEx(
        lines, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    return lines


# ──────────────────────────────────────────────────────────────────────────────
#  Canonical pitch sampling (dense, world metres)
# ──────────────────────────────────────────────────────────────────────────────

def _sample_segment(a: Tuple[float, float], b: Tuple[float, float],
                    step_m: float) -> np.ndarray:
    ax, ay = a
    bx, by = b
    L = float(np.hypot(bx - ax, by - ay))
    n = max(2, int(np.ceil(L / step_m)) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = ax + t * (bx - ax)
    ys = ay + t * (by - ay)
    return np.stack([xs, ys], axis=1)


def sample_canonical_pitch(cfg: PitchConfig = DEFAULT_PITCH,
                           step_m: float = 1.0) -> np.ndarray:
    """
    Dense (N, 2) samples along every canonical pitch line and arc.
    Step = 1 metre by default → roughly 1k points for a full 120×70 pitch.
    """
    L, W = cfg.length, cfg.width
    pl, pw = cfg.penalty_box_length, cfg.penalty_box_width
    gl, gw = cfg.goal_box_length,    cfg.goal_box_width
    r = cfg.centre_circle_radius

    segs: list[np.ndarray] = []

    # Outer rectangle
    segs.append(_sample_segment((0, 0),        (L, 0),        step_m))
    segs.append(_sample_segment((L, 0),        (L, W),        step_m))
    segs.append(_sample_segment((L, W),        (0, W),        step_m))
    segs.append(_sample_segment((0, W),        (0, 0),        step_m))

    # Halfway line
    segs.append(_sample_segment((L / 2, 0),    (L / 2, W),    step_m))

    # Left penalty box
    segs.append(_sample_segment((0,  (W - pw) / 2), (pl, (W - pw) / 2), step_m))
    segs.append(_sample_segment((pl, (W - pw) / 2), (pl, (W + pw) / 2), step_m))
    segs.append(_sample_segment((pl, (W + pw) / 2), (0,  (W + pw) / 2), step_m))

    # Left goal box
    segs.append(_sample_segment((0,  (W - gw) / 2), (gl, (W - gw) / 2), step_m))
    segs.append(_sample_segment((gl, (W - gw) / 2), (gl, (W + gw) / 2), step_m))
    segs.append(_sample_segment((gl, (W + gw) / 2), (0,  (W + gw) / 2), step_m))

    # Right penalty box
    segs.append(_sample_segment((L,     (W - pw) / 2),
                                (L-pl,  (W - pw) / 2), step_m))
    segs.append(_sample_segment((L-pl,  (W - pw) / 2),
                                (L-pl,  (W + pw) / 2), step_m))
    segs.append(_sample_segment((L-pl,  (W + pw) / 2),
                                (L,     (W + pw) / 2), step_m))

    # Right goal box
    segs.append(_sample_segment((L,     (W - gw) / 2),
                                (L-gl,  (W - gw) / 2), step_m))
    segs.append(_sample_segment((L-gl,  (W - gw) / 2),
                                (L-gl,  (W + gw) / 2), step_m))
    segs.append(_sample_segment((L-gl,  (W + gw) / 2),
                                (L,     (W + gw) / 2), step_m))

    # Centre circle
    angles = np.linspace(0, 2 * np.pi,
                         max(36, int(2 * np.pi * r / step_m)),
                         endpoint=False)
    cx = L / 2 + r * np.cos(angles)
    cy = W / 2 + r * np.sin(angles)
    segs.append(np.stack([cx, cy], axis=1))

    return np.concatenate(segs, axis=0).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
#  Refinement via distance-transform minimisation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RefineInfo:
    cost_before: float          # mean capped dist (px) before refinement
    cost_after:  float          # mean capped dist (px) after refinement
    n_active:    int            # samples inside frame & within max_px after fit
    iterations:  int


def _project(H: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """H @ [x y 1]^T  →  (u, v) with perspective divide."""
    ones = np.ones((pts_world.shape[0], 1), dtype=np.float64)
    homog = np.hstack([pts_world, ones]).T         # (3, N)
    proj = H @ homog                                # (3, N)
    w = proj[2, :]
    # Guard against singular projections.
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return np.stack([proj[0, :] / w, proj[1, :] / w], axis=1)


def _sample_dt_bilinear(dt: np.ndarray, xs: np.ndarray,
                         ys: np.ndarray) -> np.ndarray:
    """Bilinear sampling of a float32 distance-transform at (xs, ys).

    Any NaN/Inf projections or values outside [0, W)x[0, H) get assigned
    a huge distance so the Huber cap zeroes them out.
    """
    H, W = dt.shape
    # Guard: NaN / Inf / wildly out-of-range projections → treat as out-of-bounds.
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs_safe = np.where(finite, xs, -1.0)
    ys_safe = np.where(finite, ys, -1.0)
    # Also clip wildly-large values so int cast doesn't overflow.
    xs_safe = np.clip(xs_safe, -1e6, 1e6)
    ys_safe = np.clip(ys_safe, -1e6, 1e6)

    x0 = np.floor(xs_safe).astype(np.int64)
    y0 = np.floor(ys_safe).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    in_bounds = finite & (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)

    # Clip for safe indexing; we'll mask with in_bounds below.
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)

    Ia = dt[y0c, x0c]; Ib = dt[y0c, x1c]
    Ic = dt[y1c, x0c]; Id = dt[y1c, x1c]

    wa = (x1 - xs_safe) * (y1 - ys_safe)
    wb = (xs_safe - x0) * (y1 - ys_safe)
    wc = (x1 - xs_safe) * (ys_safe - y0)
    wd = (xs_safe - x0) * (ys_safe - y0)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    # For out-of-frame samples, just use the max distance (cap will zero them).
    out = np.where(in_bounds, out, 1e6)
    return out.astype(np.float64)


def _is_H_sane(H: np.ndarray, canonical_samples: np.ndarray,
               frame_shape: Tuple[int, int]) -> bool:
    """
    Reject degenerate Hs that collapse the pitch onto a point/line or flip it.

    Heuristics:
      - All pitch corners must project to finite values.
      - The projected pitch quadrilateral must have reasonable area.
      - The orientation (sign of cross product of edges) must be consistent.
    """
    if not np.all(np.isfinite(H)):
        return False
    if abs(H[2, 2]) < 1e-9:
        return False

    # Use bounding box of canonical samples as pitch corners.
    x_min, y_min = canonical_samples.min(axis=0)
    x_max, y_max = canonical_samples.max(axis=0)
    corners = np.array([
        [x_min, y_min], [x_max, y_min],
        [x_max, y_max], [x_min, y_max]
    ], dtype=np.float64)
    proj = _project(H, corners)
    if not np.all(np.isfinite(proj)):
        return False

    # Signed area via shoelace; positive => quad is ordered CCW in pixel space.
    x = proj[:, 0]; y = proj[:, 1]
    area = 0.5 * (x[0] * (y[1] - y[3]) + x[1] * (y[2] - y[0])
                 + x[2] * (y[3] - y[1]) + x[3] * (y[0] - y[2]))
    h_img, w_img = frame_shape[:2]
    # The pitch should cover a meaningful fraction of the frame (not
    # collapsed to a speck, not blown up to astronomical scale).
    if abs(area) < 0.01 * h_img * w_img:
        return False
    if abs(area) > 100.0 * h_img * w_img:
        return False

    # Don't let corners go lightyears off-screen.
    if (np.abs(proj[:, 0]) > 20 * w_img).any() \
       or (np.abs(proj[:, 1]) > 20 * h_img).any():
        return False

    return True


def refine_homography(
    H_init: np.ndarray,
    line_mask: np.ndarray,
    canonical_samples: np.ndarray,
    max_px: float = 30.0,
    max_iter: int = 200,
    prior_weight: float = 0.0,
) -> Tuple[np.ndarray, RefineInfo]:
    """
    Locally refine a 3×3 homography by aligning canonical pitch samples with
    the detected line mask.

    Parameters
    ----------
    H_init
        Seed world→pixel homography (3×3 float). Typically the output of
        the YOLO-keypoint RANSAC fit or the previous frame's H.
    line_mask
        uint8 {0, 255} image of detected white pitch lines (same resolution
        as the calibration frame).
    canonical_samples
        (N, 2) world-metre points sampled along canonical pitch edges.
    max_px
        Huber cap: any sample whose projection lands further than this
        many pixels from the nearest line is clipped. This prevents
        extrapolated regions from dominating the cost.
    max_iter
        LM iteration budget.
    prior_weight
        If > 0, adds a squared penalty   prior_weight * (h - h_init)^2
        to the cost vector. Keeps H from straying far from the seed,
        which is useful when tracking frame-to-frame (the previous
        frame's H is a strong prior). 0 disables the prior.

    Returns
    -------
    H_refined : (3, 3) float64
    info      : RefineInfo

    If refinement produces a degenerate H, the seed H is returned
    unchanged.
    """
    if line_mask.dtype != np.uint8:
        raise ValueError("line_mask must be uint8 {0, 255}")
    # distanceTransform expects foreground = non-zero; we want "distance to
    # the nearest line pixel", so invert.
    inv = cv2.bitwise_not(line_mask)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)

    # Parametrise H with 8 params (force H[2,2] = 1). Seed is normalised.
    H0 = H_init.astype(np.float64) / H_init[2, 2]
    h8_init = H0.flatten()[:8].copy()

    sqrt_prior = float(np.sqrt(prior_weight)) if prior_weight > 0 else 0.0

    def _cost_vector(h8: np.ndarray) -> np.ndarray:
        H = np.append(h8, 1.0).reshape(3, 3)
        proj = _project(H, canonical_samples)
        d = _sample_dt_bilinear(dt, proj[:, 0], proj[:, 1])
        # Huber-style cap: quadratic up to max_px, flat beyond.
        res = np.minimum(d, max_px)
        if sqrt_prior > 0.0:
            prior_res = sqrt_prior * (h8 - h8_init)
            res = np.concatenate([res, prior_res])
        return res

    # Pre-refinement cost (for reporting)
    d_before = _sample_dt_bilinear(
        dt,
        _project(np.append(h8_init, 1.0).reshape(3, 3), canonical_samples)[:, 0],
        _project(np.append(h8_init, 1.0).reshape(3, 3), canonical_samples)[:, 1],
    )
    cost_before = float(np.mean(np.minimum(d_before, max_px)))

    result = least_squares(
        _cost_vector, h8_init,
        method="lm",
        max_nfev=max_iter,
    )

    h8_ref = result.x
    H_ref = np.append(h8_ref, 1.0).reshape(3, 3).astype(np.float64)

    # Reject degenerate Hs — fall back to the seed.
    if not _is_H_sane(H_ref, canonical_samples, line_mask.shape):
        H_ref = np.append(h8_init, 1.0).reshape(3, 3).astype(np.float64)

    # Cost after (always without prior term, for honest reporting)
    proj_after = _project(H_ref, canonical_samples)
    d_after = _sample_dt_bilinear(dt, proj_after[:, 0], proj_after[:, 1])
    d_after_capped = np.minimum(d_after, max_px)
    cost_after = float(np.mean(d_after_capped))
    n_active  = int(np.sum(d_after_capped < max_px - 0.5))

    return H_ref, RefineInfo(
        cost_before=cost_before,
        cost_after=cost_after,
        n_active=n_active,
        iterations=int(result.nfev),
    )
