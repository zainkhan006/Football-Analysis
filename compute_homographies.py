"""
Per-clip homography computation.

For each SportsMOT sequence:

1. Sample N candidate frames spread across the clip.
2. Run the pitch keypoint detector on each.
3. Pick the frame with the most high-confidence keypoints
   (tie-broken by highest mean confidence).
4. Use the keypoints + their canonical world coords (from `pitch_config`)
   to fit a homography with RANSAC.
5. Save H to `homographies/<seq>.npz`.

Caveats this code is honest about:
- These are broadcast clips with panning cameras. A single H is only
  approximately correct across the whole clip — accuracy degrades as
  the camera pans away from the calibration frame.
- Future upgrade (Phase F3b): recompute H every K frames where landmarks
  allow, and interpolate / step-function between keyframes.

Usage:
    # Compute for every clip under dataset/train/
    python compute_homographies.py

    # Just one clip
    python compute_homographies.py --seq v_gQNyhv8y0QY_c013

    # Tune sampling
    python compute_homographies.py --num-candidates 16 --conf 0.5
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
from pitch_keypoint_detector import PitchKeypointDetector, PitchKeypointResult
from refine_homography import (
    build_line_mask, refine_homography, sample_canonical_pitch,
)


DATASET_TRAIN_DIR = Path("dataset/train")
HOMOGRAPHY_OUT_DIR = Path("homographies")


# Pitch x-axis thirds (length=120m). A frame is "good" if it has detected
# keypoints in multiple thirds — that prevents the homography from being
# fit on only one half of the pitch and extrapolating wildly to the other.
THIRD_BOUNDARIES = (40.0, 80.0)


def _spread_score(world_xys: np.ndarray) -> float:
    """
    Score a candidate frame's keypoints for homography fitting.

    Higher is better. Composed of:
      * thirds_seen   * 1000   — primary signal (0, 1, 2 or 3 thirds covered)
      * x_span                  — secondary tiebreaker
      * 0.5 * count             — minor tiebreaker
    """
    if len(world_xys) == 0:
        return -1.0
    xs = world_xys[:, 0]
    n_left  = (xs <  THIRD_BOUNDARIES[0]).any()
    n_mid   = ((xs >= THIRD_BOUNDARIES[0]) & (xs <= THIRD_BOUNDARIES[1])).any()
    n_right = (xs >  THIRD_BOUNDARIES[1]).any()
    thirds  = int(n_left) + int(n_mid) + int(n_right)
    x_span  = float(xs.max() - xs.min())
    return thirds * 1000.0 + x_span + 0.5 * len(world_xys)


# ─── Frame sampling ──────────────────────────────────────────────────────────

def list_frames(seq_dir: Path) -> List[Path]:
    return sorted((seq_dir / "img1").glob("*.jpg"))


def evenly_spaced(items: List, n: int) -> List:
    """Return at most n items, evenly spread through `items`."""
    if not items:
        return []
    if n >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]


# ─── Best-frame picker ───────────────────────────────────────────────────────

def find_best_calibration_frame(
    seq_dir: Path,
    detector: PitchKeypointDetector,
    num_candidates: int,
    conf_threshold: float,
    verbose: bool = True,
) -> Tuple[Optional[Path], Optional[PitchKeypointResult]]:
    """
    Sample candidates and pick the frame with the best WORLD-SPACE keypoint
    spread (thirds covered first, x-span next, count last).

    A frame with fewer keypoints spread across the whole pitch is far better
    than a frame with many keypoints clustered on one half — the latter
    makes the homography extrapolate wildly to the unseen half.
    """
    all_frames = list_frames(seq_dir)
    candidates = evenly_spaced(all_frames, num_candidates)
    if not candidates:
        return None, None

    pitch_xy = np.array(DEFAULT_PITCH.vertices_m)

    best_frame: Optional[Path] = None
    best_res:   Optional[PitchKeypointResult] = None
    best_score = -1.0

    if verbose:
        print(f"  evaluating {len(candidates)} candidate frames...")
        print(f"    {'frame':>6} {'n':>3} {'thirds':>6} "
              f"{'x_min':>5} {'x_max':>5} {'score':>7}")

    for fp in candidates:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        res       = detector.detect(img, conf_threshold=conf_threshold)
        confident = res.confident(conf_threshold)
        if not confident:
            if verbose:
                print(f"    {int(fp.stem):>6} {0:>3}")
            continue

        world = pitch_xy[[kp.index for kp in confident]]
        score = _spread_score(world)

        # Decode for printing
        xs = world[:, 0]
        thirds = (
            int((xs <  THIRD_BOUNDARIES[0]).any())
          + int(((xs >= THIRD_BOUNDARIES[0]) & (xs <= THIRD_BOUNDARIES[1])).any())
          + int((xs >  THIRD_BOUNDARIES[1]).any())
        )
        if verbose:
            print(f"    {int(fp.stem):>6} {len(confident):>3} {thirds:>6} "
                  f"{xs.min():>5.1f} {xs.max():>5.1f} {score:>7.1f}")

        if score > best_score:
            best_score = score
            best_frame = fp
            best_res   = res

    return best_frame, best_res


# ─── Per-clip driver ─────────────────────────────────────────────────────────

def compute_for_clip(
    seq_dir: Path,
    detector: PitchKeypointDetector,
    *,
    num_candidates: int = 12,
    conf_threshold: float = 0.5,
    min_keypoints:  int   = 4,
    ransac_thresh:  float = 8.0,
    refine: bool = True,
    refine_max_px: float = 30.0,
    out_dir: Path = HOMOGRAPHY_OUT_DIR,
    verbose: bool = True,
) -> Optional[PitchHomography]:
    """
    Run the full pipeline for one clip and write the resulting H file.

    Returns the PitchHomography on success, None on failure (e.g. not
    enough keypoints anywhere in the clip).
    """
    if verbose:
        print(f"\n[homo] === {seq_dir.name} ===")

    best_frame, best_res = find_best_calibration_frame(
        seq_dir, detector, num_candidates, conf_threshold, verbose,
    )

    if best_res is None:
        print(f"  [skip] no readable frames in {seq_dir.name}")
        return None

    confident = best_res.confident(conf_threshold)
    if len(confident) < min_keypoints:
        print(f"  [fail] best frame {best_frame.stem} only had "
              f"{len(confident)} confident keypoints "
              f"(need >= {min_keypoints})")
        return None

    if verbose:
        print(f"  best frame: {best_frame.stem}  "
              f"({len(confident)} confident keypoints)")

    # Build correspondences:  vertex i -> (world_xy_m, pixel_xy)
    pitch_xy = DEFAULT_PITCH.vertices_m
    world_pts: List[Tuple[float, float]] = []
    pixel_pts: List[Tuple[float, float]] = []
    for kp in confident:
        wx, wy = pitch_xy[kp.index]
        world_pts.append((wx, wy))
        pixel_pts.append(kp.pixel_xy)

    # Fit
    try:
        H = PitchHomography.from_correspondences(
            world_xy_m       = world_pts,
            pixel_xy         = pixel_pts,
            ransac_threshold_px = ransac_thresh,
            source_frame_id  = int(best_frame.stem),
        )
    except ValueError as e:
        print(f"  [fail] homography fit failed: {e}")
        return None

    if verbose:
        n_in = int(H.fit_inliers.sum()) if H.fit_inliers is not None else "?"
        print(f"  [seed] RANSAC inliers={n_in}/{len(confident)}  "
              f"reproj_err={H.fit_error_px:.2f}px")

    # Optional line-refinement: polishes H by aligning canonical pitch
    # samples to the detected white-line mask. Starts from the YOLO
    # keypoint seed so it only does local correction.
    if refine:
        frame_bgr = cv2.imread(str(best_frame))
        mask      = build_line_mask(frame_bgr)
        samples   = sample_canonical_pitch(step_m=1.0)
        try:
            H_ref_mat, info = refine_homography(
                H.H_world_to_pixel, mask, samples,
                max_px=refine_max_px,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [warn] line-refinement failed: {e}; keeping seed H")
        else:
            # Replace H matrices but keep RANSAC-inlier bbox (evidence region)
            # and the seed's source_frame_id / n_correspondences.
            # PitchHomography is frozen, so rebuild it.
            H_ref_inv = np.linalg.inv(H_ref_mat)
            H = PitchHomography(
                H_world_to_pixel  = H_ref_mat.astype(np.float64),
                H_pixel_to_world  = H_ref_inv.astype(np.float64),
                fit_inliers       = H.fit_inliers,
                fit_error_px      = H.fit_error_px,
                n_correspondences = H.n_correspondences,
                source_frame_id   = H.source_frame_id,
                inlier_world_bbox = H.inlier_world_bbox,
            )
            if verbose:
                print(f"  [refine] line-DT cost "
                      f"{info.cost_before:.2f}px -> {info.cost_after:.2f}px  "
                      f"(n_active={info.n_active}/{len(samples)}  "
                      f"iters={info.iterations})")

    out_path = out_dir / f"{seq_dir.name}.npz"
    H.save(out_path)
    if verbose:
        print(f"  [ok]   saved -> {out_path}")
    return H


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None,
                    help="Sequence folder name; default = all under dataset/train/")
    ap.add_argument("--num-candidates", type=int, default=12,
                    help="How many frames to sample per clip when searching "
                         "for the best calibration frame.")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="Confidence threshold for accepting a keypoint.")
    ap.add_argument("--min-kp", type=int, default=4,
                    help="Minimum confident keypoints required to fit H.")
    ap.add_argument("--ransac-thresh", type=float, default=8.0,
                    help="RANSAC reprojection threshold in pixels.")
    ap.add_argument("--out-dir", default=str(HOMOGRAPHY_OUT_DIR))
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip the distance-transform line refinement stage.")
    ap.add_argument("--refine-max-px", type=float, default=30.0,
                    help="Huber cap for the refinement cost (px).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PitchKeypointDetector()

    if args.seq:
        seq_dir = DATASET_TRAIN_DIR / args.seq
        if not seq_dir.exists():
            raise SystemExit(f"sequence {seq_dir} not found")
        clip_dirs = [seq_dir]
    else:
        clip_dirs = sorted(p for p in DATASET_TRAIN_DIR.iterdir()
                           if p.is_dir() and (p / "img1").exists())

    print(f"[homo] processing {len(clip_dirs)} clip(s)...")
    t0 = time.perf_counter()
    n_ok = 0
    for seq_dir in clip_dirs:
        H = compute_for_clip(
            seq_dir, detector,
            num_candidates = args.num_candidates,
            conf_threshold = args.conf,
            min_keypoints  = args.min_kp,
            ransac_thresh  = args.ransac_thresh,
            refine         = not args.no_refine,
            refine_max_px  = args.refine_max_px,
            out_dir        = out_dir,
        )
        if H is not None:
            n_ok += 1
    dt = time.perf_counter() - t0
    print(f"\n[homo] done. {n_ok}/{len(clip_dirs)} clips succeeded "
          f"in {dt:.1f}s.")


if __name__ == "__main__":
    main()
