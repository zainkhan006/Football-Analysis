"""
Per-frame homography tracking for a full clip.

Why this exists
---------------
Broadcast cameras pan and zoom. A single homography fitted on one
"calibration frame" is only accurate near that frame -- it drifts
as soon as the camera moves. Real pitch analytics needs an H per frame.

How it works (the simplest thing that works)
--------------------------------------------
Frame 0:
    H_0 = refined calibration H for the clip (already saved by
          `compute_homographies.py`).

Frame t > 0 (sequential):
    1. Build a line-mask from frame t (HSV green-dilated ∩ white).
    2. Predict H_t = H_{t-1}   (identity motion model -- pan between
                                adjacent frames is small at 25-30fps).
    3. Refine H_t via distance-transform line alignment, starting
       from the prediction. Only a handful of LM iterations needed
       because the seed is close.
    4. If the refinement cost spikes above `reseed_cost_px` (camera
       cut, big occlusion, lost track), fall back to the YOLO
       keypoint detector + full refinement on this frame.

Output
------
`homographies/<seq>_track.npz` with keys:
    H_per_frame   (T, 3, 3) float64  -- world→pixel per frame
    costs         (T,)      float32  -- DT cost per frame (px, -1 = reseed failed)
    reseeded      (T,)      bool     -- True where YOLO fallback kicked in
    frame_ids     (T,)      int32    -- frame numbers (stem of img filename)

Usage
-----
    python track_homography.py                  # all clips
    python track_homography.py --seq v_...      # one clip
    python track_homography.py --preview        # show live overlay as it tracks
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from homography import PitchHomography
from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector
from refine_homography import (build_line_mask, refine_homography,
                               sample_canonical_pitch)


DATASET_TRAIN_DIR  = Path("dataset/train")
HOMOGRAPHY_DIR     = Path("homographies")


# ──────────────────────────────────────────────────────────────────────────────
#  YOLO reseed
# ──────────────────────────────────────────────────────────────────────────────

def _yolo_reseed(frame_bgr: np.ndarray,
                 detector: PitchKeypointDetector,
                 ransac_thresh: float = 8.0,
                 conf: float = 0.5) -> Optional[np.ndarray]:
    """Try to recover H for a frame from scratch using the keypoint detector.
    Returns a 3x3 matrix or None if recovery failed."""
    res = detector.detect(frame_bgr, conf_threshold=conf)
    confident = res.confident(conf)
    if len(confident) < 4:
        return None
    pitch_xy = DEFAULT_PITCH.vertices_m
    world_pts = [pitch_xy[kp.index] for kp in confident]
    pixel_pts = [kp.pixel_xy         for kp in confident]
    try:
        H_obj = PitchHomography.from_correspondences(
            world_xy_m          = world_pts,
            pixel_xy            = pixel_pts,
            ransac_threshold_px = ransac_thresh,
        )
    except ValueError:
        return None
    return H_obj.H_world_to_pixel


# ──────────────────────────────────────────────────────────────────────────────
#  Per-clip tracker
# ──────────────────────────────────────────────────────────────────────────────

def track_clip(
    seq_dir: Path,
    detector: PitchKeypointDetector,
    *,
    max_px: float = 30.0,
    reseed_cost_px: float = 28.0,
    max_iter_per_frame: int = 40,
    preview: bool = False,
    verbose: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Track H across a full clip.

    Returns (H_per_frame, costs, reseeded, frame_ids) or None on failure.
    """
    calib_path = HOMOGRAPHY_DIR / f"{seq_dir.name}.npz"
    if not calib_path.exists():
        if verbose:
            print(f"  [skip] no calibration H at {calib_path} - "
                  f"run compute_homographies.py first")
        return None

    calib = PitchHomography.load(calib_path)
    samples = sample_canonical_pitch(step_m=1.0)

    frames = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frames:
        print(f"  [skip] no images under {seq_dir}")
        return None

    T = len(frames)
    H_per_frame = np.zeros((T, 3, 3), dtype=np.float64)
    costs       = np.full(T, -1.0, dtype=np.float32)
    reseeded    = np.zeros(T, dtype=bool)
    frame_ids   = np.array([int(f.stem) for f in frames], dtype=np.int32)

    H_prev = calib.H_world_to_pixel.copy()
    t_start = time.perf_counter()
    n_reseed = 0
    n_fail   = 0

    for i, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        if frame is None:
            H_per_frame[i] = H_prev
            continue

        mask = build_line_mask(frame)

        # Step 1: refine starting from previous H with a motion prior that
        # prevents H from drifting too far between frames. Camera pan between
        # adjacent frames is tiny at 25-30fps, so a reasonably strong prior
        # is valid and guards against degenerate fits.
        try:
            H_t, info = refine_homography(
                H_prev, mask, samples,
                max_px=max_px,
                max_iter=max_iter_per_frame,
                prior_weight=50.0,      # ~7 px-equivalent per param
            )
            cost = info.cost_after
        except Exception:
            H_t = H_prev.copy()
            cost = float("inf")

        # Step 2: if cost is too high, reseed with YOLO (camera cut /
        # lost track). No prior on the reseed — we want to drop the
        # previous H entirely.
        if cost > reseed_cost_px:
            H_seed = _yolo_reseed(frame, detector)
            if H_seed is not None:
                try:
                    H_re, info_re = refine_homography(
                        H_seed, mask, samples,
                        max_px=max_px,
                        max_iter=max_iter_per_frame * 2,
                        prior_weight=0.0,
                    )
                    if info_re.cost_after < cost:
                        H_t = H_re
                        cost = info_re.cost_after
                        reseeded[i] = True
                        n_reseed += 1
                except Exception:
                    pass

        if not np.all(np.isfinite(H_t)) or abs(H_t[2, 2]) < 1e-9:
            H_t = H_prev.copy()
            n_fail += 1

        H_per_frame[i] = H_t
        costs[i]       = cost if np.isfinite(cost) else -1.0
        H_prev = H_t

        if verbose and (i % max(1, T // 20) == 0):
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / max(elapsed, 1e-3)
            tag = " (reseed)" if reseeded[i] else ""
            print(f"    [{i + 1:>4}/{T}]  frame={fp.stem}  "
                  f"cost={cost:.2f}px  {fps:4.1f}fps{tag}")

        if preview:
            vis = frame.copy()
            _draw_overlay(vis, H_t,
                          colour=(0, 255, 0) if not reseeded[i] else (0, 165, 255))
            cv2.putText(vis,
                        f"frame {fp.stem}  cost={cost:.1f}px"
                        + ("  RESEED" if reseeded[i] else ""),
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)
            cv2.imshow("track_homography", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    if preview:
        cv2.destroyAllWindows()

    dt = time.perf_counter() - t_start
    if verbose:
        mean_cost = float(costs[costs >= 0].mean()) if (costs >= 0).any() else -1
        print(f"  done {T} frames in {dt:.1f}s "
              f"({T / max(dt, 1e-3):.1f}fps)  "
              f"mean_cost={mean_cost:.2f}px  "
              f"reseeded={n_reseed}  failures={n_fail}")

    return H_per_frame, costs, reseeded, frame_ids


# ──────────────────────────────────────────────────────────────────────────────
#  Minimal overlay for the preview mode
# ──────────────────────────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, H: np.ndarray, colour) -> None:
    cfg = DEFAULT_PITCH
    L, W = cfg.length, cfg.width
    pl, pw = cfg.penalty_box_length, cfg.penalty_box_width
    r = cfg.centre_circle_radius

    def _prj(w):
        pt = np.array([[[w[0], w[1]]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, H)[0, 0]
        return int(round(out[0])), int(round(out[1]))

    def ln(a, b, t=2):
        p0, p1 = _prj(a), _prj(b)
        h, w = frame.shape[:2]
        if (-w <= p0[0] <= 2 * w and -h <= p0[1] <= 2 * h
            and -w <= p1[0] <= 2 * w and -h <= p1[1] <= 2 * h):
            cv2.line(frame, p0, p1, colour, t, cv2.LINE_AA)

    for seg in [((0, 0), (L, 0)), ((L, 0), (L, W)),
                ((L, W), (0, W)), ((0, W), (0, 0)),
                ((L / 2, 0), (L / 2, W))]:
        ln(*seg, t=2)
    for box in [
        [(0, (W - pw) / 2), (pl, (W - pw) / 2),
         (pl, (W + pw) / 2), (0, (W + pw) / 2)],
        [(L, (W - pw) / 2), (L - pl, (W - pw) / 2),
         (L - pl, (W + pw) / 2), (L, (W + pw) / 2)],
    ]:
        for a, b in zip(box[:-1], box[1:]):
            ln(a, b, t=1)
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=True)
    pts = np.stack([L / 2 + r * np.cos(ang), W / 2 + r * np.sin(ang)], 1)
    for i in range(len(pts) - 1):
        ln(pts[i], pts[i + 1], t=1)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    ap.add_argument("--max-iter", type=int, default=40,
                    help="LM iterations per frame (default 40)")
    ap.add_argument("--reseed-cost", type=float, default=28.0,
                    help="Cost threshold above which we reseed with YOLO")
    ap.add_argument("--preview", action="store_true",
                    help="Show a live window of the overlay while tracking")
    args = ap.parse_args()

    detector = PitchKeypointDetector()

    if args.seq:
        clip_dirs: List[Path] = [DATASET_TRAIN_DIR / args.seq]
    else:
        clip_dirs = sorted(p for p in DATASET_TRAIN_DIR.iterdir()
                           if p.is_dir() and (p / "img1").exists())

    print(f"[track] tracking H across {len(clip_dirs)} clip(s)...")
    for seq_dir in clip_dirs:
        print(f"\n[track] === {seq_dir.name} ===")
        out = track_clip(
            seq_dir, detector,
            max_iter_per_frame=args.max_iter,
            reseed_cost_px=args.reseed_cost,
            preview=args.preview,
        )
        if out is None:
            continue
        H_per_frame, costs, reseeded, frame_ids = out
        out_path = HOMOGRAPHY_DIR / f"{seq_dir.name}_track.npz"
        np.savez(out_path,
                 H_per_frame = H_per_frame,
                 costs       = costs,
                 reseeded    = reseeded,
                 frame_ids   = frame_ids)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
