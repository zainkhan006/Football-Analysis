"""
Find out WHY the homography is wrong.

For a given clip:
  1. Run keypoint detection on the calibration frame.
  2. Pair each detected keypoint with its assumed world coord
     (from `pitch_config.vertices_m`).
  3. Fit H with RANSAC.
  4. For each correspondence, compute |projected pixel - detected pixel|.
  5. Print a table sorted by residual.  Inliers should be small; outliers
     are correspondences where the model and our pitch_config disagree
     about which vertex this is.

Also overlays the result so you can see which keypoints fit and which don't.

Usage:
    python debug_homography_fit.py --seq v_dw7LOz17Omg_c053 --frame 79
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--ransac", type=float, default=8.0)
    args = ap.parse_args()

    img_path = Path("dataset/train") / args.seq / "img1" / f"{args.frame:06d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"frame not found: {img_path}")

    det = PitchKeypointDetector()
    res = det.detect(img, conf_threshold=args.conf)
    confident = res.confident(args.conf)
    print(f"\n{len(confident)} confident KPs (>= {args.conf}):")

    pitch_xy = DEFAULT_PITCH.vertices_m
    world  = np.array([pitch_xy[kp.index] for kp in confident], dtype=np.float32)
    pixel  = np.array([kp.pixel_xy        for kp in confident], dtype=np.float32)

    H, mask = cv2.findHomography(world, pixel, cv2.RANSAC, args.ransac)
    inlier  = mask.flatten().astype(bool)

    proj = cv2.perspectiveTransform(
        world.reshape(-1, 1, 2).astype(np.float64),
        H.astype(np.float64)
    ).reshape(-1, 2)
    err  = np.linalg.norm(proj - pixel, axis=1)

    # Print sorted table
    rows = sorted(
        zip(confident, world, pixel, proj, err, inlier),
        key=lambda r: r[4]
    )
    print(f"\n{'v#':>3} {'world (x,y) m':>16} "
          f"{'detected px':>14} {'projected px':>14} "
          f"{'err':>7}  {'in?':>4}")
    print("-" * 70)
    for kp, w, p, q, e, ok in rows:
        v = kp.index + 1
        print(f"{v:>3} {f'({w[0]:5.1f}, {w[1]:5.1f})':>16} "
              f"{f'({p[0]:6.1f}, {p[1]:6.1f})':>14} "
              f"{f'({q[0]:6.1f}, {q[1]:6.1f})':>14} "
              f"{e:7.2f}  {'IN' if ok else 'out':>4}")

    print(f"\nRANSAC inliers: {inlier.sum()}/{len(world)}")
    print(f"Mean inlier err: {err[inlier].mean():.2f}px")
    print(f"Mean outlier err: {err[~inlier].mean() if (~inlier).any() else 0:.2f}px")

    # Project ALL canonical keypoints through H, draw on image
    out = img.copy()
    pts_world_all = np.array(DEFAULT_PITCH.vertices_m, dtype=np.float64)
    pts_proj_all  = cv2.perspectiveTransform(
        pts_world_all.reshape(-1, 1, 2), H.astype(np.float64)
    ).reshape(-1, 2)

    # Draw projected canonical positions (BLUE crosses)
    for i, (px, py) in enumerate(pts_proj_all):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        x, y = int(round(px)), int(round(py))
        cv2.drawMarker(out, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(out, f"v{i+1}", (x + 8, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    # Draw detected (in green) and connect each detected to its claimed
    # vertex projection (line). Long lines = bad correspondences.
    for kp, ok in zip(confident, inlier):
        x, y = int(round(kp.pixel_xy[0])), int(round(kp.pixel_xy[1]))
        col_dot = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(out, (x, y), 6, col_dot, -1)
        cv2.circle(out, (x, y), 8, (0, 0, 0),  1)

        # Line to projected canonical pos
        qx, qy = pts_proj_all[kp.index]
        if np.isfinite(qx) and np.isfinite(qy):
            cv2.line(out, (x, y),
                     (int(round(qx)), int(round(qy))),
                     (255, 255, 0), 1, cv2.LINE_AA)

    out_path = f"debug_fit_{args.seq}_{args.frame}.png"
    cv2.imwrite(out_path, out)
    print(f"\n[saved] {out_path}")
    print("Legend: green dot = inlier detected   red dot = outlier detected")
    print("        blue cross = projected canonical position")
    print("        cyan line  = residual between detected and projected")


if __name__ == "__main__":
    main()
