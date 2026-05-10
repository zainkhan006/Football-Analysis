"""
A/B viewer for homography: seed (YOLO-keypoint RANSAC) vs refined
(seed + distance-transform line alignment).

Runs the pipeline twice per clip -- once with --no-refine semantics, once
with refinement -- and dumps side-by-side JPEGs to `debug_refine_compare/`.

Usage:
    python compare_refine.py              # all clips
    python compare_refine.py --seq v_...  # one clip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from homography import PitchHomography
from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector
from refine_homography import (build_line_mask, refine_homography,
                               sample_canonical_pitch)
from compute_homographies import find_best_calibration_frame


DATASET_TRAIN_DIR = Path("dataset/train")
OUT_DIR           = Path("debug_refine_compare")


def _safe(p, shape):
    Hh, Ww = shape[:2]
    x, y = p
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if x < -Ww or x > 2 * Ww or y < -Hh or y > 2 * Hh:
        return None
    return int(round(x)), int(round(y))


def _project_H(H, w):
    pt = np.array([[[w[0], w[1]]]], dtype=np.float64)
    return tuple(cv2.perspectiveTransform(pt, H)[0, 0].tolist())


def draw_overlay(frame: np.ndarray, H: np.ndarray, colour) -> None:
    cfg = DEFAULT_PITCH
    L, W = cfg.length, cfg.width
    pl, pw = cfg.penalty_box_length, cfg.penalty_box_width
    gl, gw = cfg.goal_box_length, cfg.goal_box_width
    r = cfg.centre_circle_radius

    def ln(a, b, t=2):
        p0 = _safe(_project_H(H, a), frame.shape)
        p1 = _safe(_project_H(H, b), frame.shape)
        if p0 and p1:
            cv2.line(frame, p0, p1, colour, t, cv2.LINE_AA)

    # Outline + halfway
    for seg in [((0,0),(L,0)), ((L,0),(L,W)), ((L,W),(0,W)),
                ((0,W),(0,0)), ((L/2,0),(L/2,W))]:
        ln(*seg, t=2)
    # Penalty + goal boxes, both sides
    for box in [
        [(0,(W-pw)/2),(pl,(W-pw)/2),(pl,(W+pw)/2),(0,(W+pw)/2)],
        [(0,(W-gw)/2),(gl,(W-gw)/2),(gl,(W+gw)/2),(0,(W+gw)/2)],
        [(L,(W-pw)/2),(L-pl,(W-pw)/2),(L-pl,(W+pw)/2),(L,(W+pw)/2)],
        [(L,(W-gw)/2),(L-gl,(W-gw)/2),(L-gl,(W+gw)/2),(L,(W+gw)/2)],
    ]:
        for a, b in zip(box[:-1], box[1:]):
            ln(a, b, t=1)
    # Centre circle
    ang = np.linspace(0, 2*np.pi, 60, endpoint=True)
    pts = np.stack([L/2 + r*np.cos(ang), W/2 + r*np.sin(ang)], 1)
    for i in range(len(pts)-1):
        ln(pts[i], pts[i+1], t=1)


def process_clip(seq_dir: Path, detector: PitchKeypointDetector) -> None:
    best_frame, best_res = find_best_calibration_frame(
        seq_dir, detector, num_candidates=20, conf_threshold=0.5,
        verbose=False,
    )
    if best_res is None:
        print(f"[compare] {seq_dir.name}: no readable frames")
        return
    confident = best_res.confident(0.5)
    if len(confident) < 4:
        print(f"[compare] {seq_dir.name}: too few keypoints")
        return

    frame = cv2.imread(str(best_frame))
    pitch_xy = DEFAULT_PITCH.vertices_m
    world_pts = [pitch_xy[kp.index]  for kp in confident]
    pixel_pts = [kp.pixel_xy          for kp in confident]

    H_seed_obj = PitchHomography.from_correspondences(
        world_xy_m=world_pts, pixel_xy=pixel_pts,
        ransac_threshold_px=8.0,
        source_frame_id=int(best_frame.stem),
    )
    H_seed = H_seed_obj.H_world_to_pixel

    mask = build_line_mask(frame)
    samples = sample_canonical_pitch(step_m=1.0)
    H_ref, info = refine_homography(H_seed, mask, samples, max_px=30.0)

    seed_img = frame.copy()
    ref_img  = frame.copy()
    draw_overlay(seed_img, H_seed, (0, 0, 255))     # RED  = seed
    draw_overlay(ref_img,  H_ref,  (0, 255, 0))     # GREEN = refined

    side = np.hstack([seed_img, ref_img])
    label = np.zeros((36, side.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, f"SEED  reproj_err={H_seed_obj.fit_error_px:.2f}px  "
                       f"N={len(confident)}  "
                       f"frame={best_frame.stem}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(label, f"REFINED  line-DT {info.cost_before:.1f}px -> "
                       f"{info.cost_after:.1f}px  "
                       f"active={info.n_active}/{len(samples)}",
                (frame.shape[1] + 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    side = np.vstack([label, side])

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{seq_dir.name}.jpg"
    cv2.imwrite(str(out_path), side)
    print(f"[compare] {seq_dir.name}  "
          f"cost {info.cost_before:.2f} -> {info.cost_after:.2f}px  "
          f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    args = ap.parse_args()

    detector = PitchKeypointDetector()
    if args.seq:
        clip_dirs = [DATASET_TRAIN_DIR / args.seq]
    else:
        clip_dirs = sorted(p for p in DATASET_TRAIN_DIR.iterdir()
                           if p.is_dir() and (p / "img1").exists())
    for d in clip_dirs:
        process_clip(d, detector)


if __name__ == "__main__":
    main()
