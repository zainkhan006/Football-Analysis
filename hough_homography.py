"""
Hough-augmented per-frame homography solver.

Given a frame and a *seed* homography (e.g. from a nearby YOLO-sane frame),
this module:
    1. Builds a clean line mask (pitch-interior gated, players blanked).
    2. Runs probabilistic Hough on it → straight pitch-line segments.
    3. Projects canonical pitch lines through the seed H → predicted
       image-space lines.
    4. Matches each Hough segment to its closest canonical line by
       angle + perpendicular distance.
    5. From each matched (Hough segment, canonical line) pair, generates
       N point correspondences (segment endpoints in pixels ↔ canonical
       line interpolated points in world metres).
    6. Combines those point pairs with the YOLO keypoints (if any) and
       re-fits H via RANSAC DLT.
    7. Sanity-gates the result.

Public API
----------
    canonical_pitch_lines(cfg) -> list[CanonicalLine]
    fit_H_with_hough(frame_bgr, kp_result, H_seed, gt_bboxes=None,
                     conf_thresh=0.5) -> (H or None, info_dict)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pitch_config import DEFAULT_PITCH, PitchConfig
from pitch_mask import build_pitch_mask
from refine_homography import build_line_mask
from test_per_frame_yolo_h import _is_H_sane


# ──────────────────────────────────────────────────────────────────────────────
#  Canonical pitch lines (straight segments only)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CanonicalLine:
    name: str
    p0_m: Tuple[float, float]
    p1_m: Tuple[float, float]


def canonical_pitch_lines(cfg: PitchConfig = DEFAULT_PITCH) -> List[CanonicalLine]:
    """The set of straight pitch lines we attempt to match Hough segments to."""
    L, W = cfg.length, cfg.width
    pl, pw = cfg.penalty_box_length, cfg.penalty_box_width
    gl, gw = cfg.goal_box_length, cfg.goal_box_width

    return [
        # Outer rectangle
        CanonicalLine("touch_bot",      (0,        0),         (L,        0)),
        CanonicalLine("touch_top",      (0,        W),         (L,        W)),
        CanonicalLine("goal_left",      (0,        0),         (0,        W)),
        CanonicalLine("goal_right",     (L,        0),         (L,        W)),
        # Halfway
        CanonicalLine("halfway",        (L / 2,    0),         (L / 2,    W)),
        # Left penalty box (3 sides; the 4th side is the goal line itself)
        CanonicalLine("L_pbox_bot",     (0,        (W - pw) / 2), (pl, (W - pw) / 2)),
        CanonicalLine("L_pbox_top",     (0,        (W + pw) / 2), (pl, (W + pw) / 2)),
        CanonicalLine("L_pbox_front",   (pl,       (W - pw) / 2), (pl, (W + pw) / 2)),
        # Left goal box
        CanonicalLine("L_gbox_bot",     (0,        (W - gw) / 2), (gl, (W - gw) / 2)),
        CanonicalLine("L_gbox_top",     (0,        (W + gw) / 2), (gl, (W + gw) / 2)),
        CanonicalLine("L_gbox_front",   (gl,       (W - gw) / 2), (gl, (W + gw) / 2)),
        # Right penalty box
        CanonicalLine("R_pbox_bot",     (L - pl,   (W - pw) / 2), (L,  (W - pw) / 2)),
        CanonicalLine("R_pbox_top",     (L - pl,   (W + pw) / 2), (L,  (W + pw) / 2)),
        CanonicalLine("R_pbox_front",   (L - pl,   (W - pw) / 2), (L - pl, (W + pw) / 2)),
        # Right goal box
        CanonicalLine("R_gbox_bot",     (L - gl,   (W - gw) / 2), (L,  (W - gw) / 2)),
        CanonicalLine("R_gbox_top",     (L - gl,   (W + gw) / 2), (L,  (W + gw) / 2)),
        CanonicalLine("R_gbox_front",   (L - gl,   (W - gw) / 2), (L - gl, (W + gw) / 2)),
    ]


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _seg_angle(seg: np.ndarray) -> float:
    """Return segment angle in [0, pi) (orientation, not direction)."""
    dx = seg[2] - seg[0]
    dy = seg[3] - seg[1]
    a = np.arctan2(dy, dx)
    if a < 0:
        a += np.pi
    if a >= np.pi:
        a -= np.pi
    return float(a)


def _seg_length(seg: np.ndarray) -> float:
    return float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))


def _project_world_pts(H: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    """(N,2) world → (N,2) image, via H. Drops infeasible points (w near zero)."""
    homog = np.hstack([pts_m, np.ones((len(pts_m), 1))]).T          # (3, N)
    proj = H @ homog                                                # (3, N)
    w = proj[2, :]
    safe = np.abs(w) > 1e-9
    out = np.full((len(pts_m), 2), np.nan, dtype=np.float64)
    out[safe] = (proj[:2, safe] / w[safe]).T
    return out


def _angle_diff(a: float, b: float) -> float:
    """Smallest angle difference in [0, pi/2] (orientation only)."""
    d = abs(a - b) % np.pi
    if d > np.pi / 2:
        d = np.pi - d
    return float(d)


def _point_to_line_dist(pt: np.ndarray, line_p0: np.ndarray,
                         line_p1: np.ndarray) -> float:
    """Perpendicular distance from `pt` to infinite line through p0, p1."""
    d = line_p1 - line_p0
    n = np.array([-d[1], d[0]])
    nrm = np.linalg.norm(n)
    if nrm < 1e-9:
        return float(np.linalg.norm(pt - line_p0))
    n = n / nrm
    return float(abs(np.dot(pt - line_p0, n)))


def _segments_in_pitch(segs: np.ndarray, pitch_mask: np.ndarray,
                        min_inside_frac: float = 0.7) -> np.ndarray:
    """Keep only segments where >= min_inside_frac of sampled points lie in pitch_mask."""
    if len(segs) == 0:
        return segs
    h, w = pitch_mask.shape[:2]
    keep = []
    for s in segs:
        n = max(8, int(_seg_length(s) / 5))
        ts = np.linspace(0, 1, n)
        xs = (s[0] + ts * (s[2] - s[0])).astype(np.int32)
        ys = (s[1] + ts * (s[3] - s[1])).astype(np.int32)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        in_pitch = pitch_mask[ys, xs] > 0
        if in_pitch.mean() >= min_inside_frac:
            keep.append(s)
    if not keep:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(keep)


# ──────────────────────────────────────────────────────────────────────────────
#  Line-mask preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _subtract_bboxes(mask: np.ndarray,
                      bboxes: Sequence[Tuple[int, int, int, int]],
                      pad_px: int = 6) -> np.ndarray:
    if not bboxes:
        return mask
    out = mask.copy()
    h, w = mask.shape[:2]
    for x, y, bw, bh in bboxes:
        x0 = max(0, int(x) - pad_px)
        y0 = max(0, int(y) - pad_px)
        x1 = min(w, int(x + bw) + pad_px)
        y1 = min(h, int(y + bh) + pad_px)
        out[y0:y1, x0:x1] = 0
    return out


def _suppress_topleft_overlay(mask: np.ndarray,
                               frac_w: float = 0.20,
                               frac_h: float = 0.10) -> np.ndarray:
    """Zero a top-left rectangle to suppress broadcast scoreboards."""
    out = mask.copy()
    h, w = mask.shape[:2]
    out[: int(h * frac_h), : int(w * frac_w)] = 0
    return out


def extract_hough_segments(
    frame_bgr: np.ndarray,
    gt_bboxes: Sequence[Tuple[int, int, int, int]] = (),
    min_seg_len_px: int = 60,
    rho:        float = 1.0,
    theta:      float = np.pi / 360.0,
    threshold:  int   = 60,
    max_line_gap: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (segments, pitch_mask).
    segments is (N, 4) float32 [[x1,y1,x2,y2],...] in pixels.
    """
    pitch = build_pitch_mask(frame_bgr)
    line_raw = build_line_mask(frame_bgr)
    gated = cv2.bitwise_and(line_raw, pitch)
    gated = _subtract_bboxes(gated, gt_bboxes, pad_px=6)
    gated = _suppress_topleft_overlay(gated)

    segs = cv2.HoughLinesP(
        gated, rho, theta, threshold,
        minLineLength=min_seg_len_px,
        maxLineGap=max_line_gap,
    )
    if segs is None:
        return np.zeros((0, 4), dtype=np.float32), pitch
    segs = segs.reshape(-1, 4).astype(np.float32)
    # Restrict to pitch interior
    segs = _segments_in_pitch(segs, pitch, min_inside_frac=0.7)
    # Length filter (Hough may already have applied via minLineLength but be strict)
    if len(segs) > 0:
        L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        segs = segs[L >= min_seg_len_px]
    return segs, pitch


# ──────────────────────────────────────────────────────────────────────────────
#  Match Hough segments to canonical pitch lines via seed H
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchedLine:
    canonical: CanonicalLine
    seg_px:    np.ndarray         # (4,) image-space segment [x1,y1,x2,y2]
    proj_p0:   np.ndarray         # (2,) seed-H-projected canonical p0 (image)
    proj_p1:   np.ndarray         # (2,) seed-H-projected canonical p1 (image)
    angle_err_deg: float
    perp_err_px:   float


def match_segments_to_canonical(
    segments_px: np.ndarray,
    H_seed: np.ndarray,
    cfg: PitchConfig = DEFAULT_PITCH,
    angle_tol_deg: float = 6.0,
    perp_tol_px:   float = 25.0,
    frame_shape:   Optional[Tuple[int, int]] = None,
) -> List[MatchedLine]:
    """
    For each Hough segment, find the canonical line whose seed-H projection
    is closest (in angle and perpendicular distance). Returns the list of
    accepted matches.
    """
    if len(segments_px) == 0:
        return []
    canon = canonical_pitch_lines(cfg)

    # Project canonical line endpoints with H_seed
    canon_proj: list[tuple[CanonicalLine, np.ndarray, np.ndarray, float]] = []
    for cl in canon:
        p = _project_world_pts(H_seed, np.array([cl.p0_m, cl.p1_m]))
        if not np.all(np.isfinite(p)):
            continue
        # Skip canonical lines whose projection is fully off-screen / tiny
        if frame_shape is not None:
            h_img, w_img = frame_shape[:2]
            if (np.abs(p[:, 0]) > 3 * w_img).any() \
               or (np.abs(p[:, 1]) > 3 * h_img).any():
                continue
        seg_len = float(np.linalg.norm(p[1] - p[0]))
        if seg_len < 5.0:
            continue
        ang = _seg_angle(np.array([p[0, 0], p[0, 1], p[1, 0], p[1, 1]]))
        canon_proj.append((cl, p[0], p[1], ang))

    if not canon_proj:
        return []

    matches: list[MatchedLine] = []
    used_canon = set()
    for seg in segments_px:
        seg_a = _seg_angle(seg)
        seg_mid = np.array([(seg[0] + seg[2]) * 0.5, (seg[1] + seg[3]) * 0.5])

        best = None
        best_score = float("inf")
        for cl, p0, p1, ang in canon_proj:
            ad = np.degrees(_angle_diff(seg_a, ang))
            if ad > angle_tol_deg:
                continue
            d = _point_to_line_dist(seg_mid, p0, p1)
            if d > perp_tol_px:
                continue
            # Score: combine perp distance (px) + angle penalty (px-equivalent)
            score = d + 2.0 * ad
            if score < best_score:
                best_score = score
                best = (cl, p0, p1, ad, d)

        if best is None:
            continue
        cl, p0, p1, ad, d = best
        # Soft constraint: prefer one match per canonical line, keep the best
        # If a previous match used this canonical line with a worse (higher) score,
        # we'd want to replace it. For simplicity here we allow many segments per
        # canonical (they reinforce each other in DLT).
        matches.append(MatchedLine(
            canonical = cl,
            seg_px    = seg.astype(np.float64),
            proj_p0   = p0,
            proj_p1   = p1,
            angle_err_deg = float(ad),
            perp_err_px   = float(d),
        ))
        used_canon.add(cl.name)

    return matches


# ──────────────────────────────────────────────────────────────────────────────
#  Build point correspondences from matched lines
# ──────────────────────────────────────────────────────────────────────────────

def correspondences_from_matches(
    matches: List[MatchedLine],
    samples_per_segment: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert matched lines into point correspondences for DLT.

    For each match we need to know the parameter t∈[0,1] along the canonical
    line that maps to the segment endpoints. We obtain it by projecting the
    Hough-segment endpoints onto the seed-projected canonical line, computing
    its t-parameter, and interpolating the canonical line in world space at
    those same t values.

    Returns (world_pts (N,2), pixel_pts (N,2)).
    """
    if not matches:
        return np.zeros((0, 2)), np.zeros((0, 2))

    world_pts: list[Tuple[float, float]] = []
    pixel_pts: list[Tuple[float, float]] = []

    for m in matches:
        # Project Hough endpoints onto the seed-projected canonical line in image
        # to find t-parameters along the canonical line.
        proj = m.proj_p1 - m.proj_p0
        L2 = float(np.dot(proj, proj))
        if L2 < 1e-9:
            continue
        e0 = np.array([m.seg_px[0], m.seg_px[1]])
        e1 = np.array([m.seg_px[2], m.seg_px[3]])
        t0 = float(np.dot(e0 - m.proj_p0, proj) / L2)
        t1 = float(np.dot(e1 - m.proj_p0, proj) / L2)
        # Clamp to a slightly extended range; if both t are far out, skip.
        if min(t0, t1) > 1.3 or max(t0, t1) < -0.3:
            continue

        # Sample N points uniformly along the actual Hough segment in image,
        # and map each to its corresponding world-line point via t.
        ts_seg = np.linspace(0.0, 1.0, samples_per_segment)
        # Image-space samples along the Hough segment itself
        px_x = e0[0] + ts_seg * (e1[0] - e0[0])
        px_y = e0[1] + ts_seg * (e1[1] - e0[1])

        # The same image points correspond to t-values along the canonical
        # projected line that vary linearly between t0 and t1.
        ts_canon = t0 + ts_seg * (t1 - t0)

        cx0, cy0 = m.canonical.p0_m
        cx1, cy1 = m.canonical.p1_m
        wx = cx0 + ts_canon * (cx1 - cx0)
        wy = cy0 + ts_canon * (cy1 - cy0)

        for i in range(samples_per_segment):
            pixel_pts.append((float(px_x[i]), float(px_y[i])))
            world_pts.append((float(wx[i]),   float(wy[i])))

    return (np.asarray(world_pts, dtype=np.float64),
            np.asarray(pixel_pts, dtype=np.float64))


# ──────────────────────────────────────────────────────────────────────────────
#  Public solver
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HoughFitInfo:
    n_yolo_kp:        int
    n_hough_segs:     int
    n_matched_lines:  int
    n_combined_pts:   int
    n_ransac_inliers: int
    mean_reproj_px:   float
    sane:             bool
    fail_reason:      str


def fit_H_with_hough(
    frame_bgr: np.ndarray,
    kp_result,
    H_seed: np.ndarray,
    gt_bboxes: Sequence[Tuple[int, int, int, int]] = (),
    conf_thresh: float = 0.5,
    cfg: PitchConfig = DEFAULT_PITCH,
    ransac_thresh_px: float = 8.0,
    min_corresp: int = 6,
) -> Tuple[Optional[np.ndarray], HoughFitInfo]:
    """
    Fit H using YOLO keypoints AUGMENTED with Hough-line correspondences.

    Args:
        frame_bgr   – the frame (H, W, 3)
        kp_result   – PitchKeypointDetector output
        H_seed      – prior H from a nearby sane frame, used only for matching
        gt_bboxes   – optional player bboxes to blank from the line mask
        conf_thresh – YOLO keypoint confidence threshold

    Returns:
        (H or None, HoughFitInfo)
    """
    info = HoughFitInfo(
        n_yolo_kp        = 0,
        n_hough_segs     = 0,
        n_matched_lines  = 0,
        n_combined_pts   = 0,
        n_ransac_inliers = 0,
        mean_reproj_px   = float("nan"),
        sane             = False,
        fail_reason      = "",
    )

    # 1. YOLO keypoints
    vertices = cfg.vertices_m
    yolo_world: list[Tuple[float, float]] = []
    yolo_pixel: list[Tuple[float, float]] = []
    for kp in kp_result.keypoints:
        if kp.confidence < conf_thresh:
            continue
        x, y = kp.pixel_xy
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        wx, wy = vertices[kp.index]
        yolo_world.append((wx, wy))
        yolo_pixel.append((x, y))
    info.n_yolo_kp = len(yolo_world)

    # 2. Extract Hough segments
    segs, _ = extract_hough_segments(frame_bgr, gt_bboxes=gt_bboxes)
    info.n_hough_segs = len(segs)

    # 3. Match to canonical pitch lines using seed H
    matches: List[MatchedLine] = []
    if len(segs) > 0 and H_seed is not None and np.all(np.isfinite(H_seed)):
        matches = match_segments_to_canonical(
            segs, H_seed, cfg=cfg, frame_shape=frame_bgr.shape)
    info.n_matched_lines = len(matches)

    # 4. Correspondences from matches + YOLO kps
    line_world, line_pixel = correspondences_from_matches(
        matches, samples_per_segment=5)

    if len(yolo_world) > 0:
        all_world = np.vstack([np.array(yolo_world, dtype=np.float64), line_world]) \
                     if len(line_world) else np.array(yolo_world, dtype=np.float64)
        all_pixel = np.vstack([np.array(yolo_pixel, dtype=np.float64), line_pixel]) \
                     if len(line_pixel) else np.array(yolo_pixel, dtype=np.float64)
    else:
        all_world = line_world
        all_pixel = line_pixel
    info.n_combined_pts = len(all_world)

    if len(all_world) < min_corresp:
        info.fail_reason = f"only {len(all_world)} combined correspondences"
        return None, info

    # Degeneracy guard: bbox of pixel points
    px_bbox = (all_pixel.max(axis=0) - all_pixel.min(axis=0))
    if px_bbox[0] < 60 or px_bbox[1] < 60:
        info.fail_reason = f"correspondence bbox too small: {px_bbox}"
        return None, info

    # 5. RANSAC DLT
    H, mask = cv2.findHomography(
        srcPoints=all_world,
        dstPoints=all_pixel,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh_px,
        maxIters=2000,
        confidence=0.999,
    )
    if H is None:
        info.fail_reason = "findHomography returned None"
        return None, info

    mask = mask.ravel().astype(bool)
    info.n_ransac_inliers = int(mask.sum())

    if info.n_ransac_inliers < min_corresp:
        info.fail_reason = f"only {info.n_ransac_inliers} RANSAC inliers"
        return None, info

    homog = np.hstack([all_world[mask], np.ones((mask.sum(), 1))]).T
    proj = H @ homog
    proj_xy = (proj[:2, :] / proj[2, :]).T
    diffs = proj_xy - all_pixel[mask]
    info.mean_reproj_px = float(np.hypot(diffs[:, 0], diffs[:, 1]).mean())

    # 6. Sanity gate
    if not _is_H_sane(H, frame_bgr.shape):
        info.fail_reason = "H failed sanity check"
        return None, info

    info.sane = True
    return H.astype(np.float64), info
