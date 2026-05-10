"""
Diagnostic: how WIDELY do the detected keypoints spread across the pitch
in different frames of a clip?  A homography fit only on (say) the left
half will extrapolate badly when drawing the right half — that's our
current failure mode.

Usage:
    python debug_keypoint_spread.py --seq v_dw7LOz17Omg_c053
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
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    det = PitchKeypointDetector(verbose=False)
    frames = sorted(Path(f"dataset/train/{args.seq}/img1").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in dataset/train/{args.seq}/img1")
    pitch_xy = np.array(DEFAULT_PITCH.vertices_m)

    samples = np.linspace(0, len(frames) - 1, args.samples).astype(int)

    print(f"{'frame':>6}  {'n_kp':>4}  {'x_min':>5}  {'x_max':>5}  "
          f"{'y_min':>5}  {'y_max':>5}  {'left':>4}  {'mid':>3}  "
          f"{'right':>5}")
    print("-" * 60)

    best_spread = -1.0
    best_frame  = None

    for i in samples:
        fp = frames[i]
        img = cv2.imread(str(fp))
        res = det.detect(img, conf_threshold=args.conf)
        confident = res.confident(args.conf)
        if not confident:
            print(f"{int(fp.stem):>6}  {0:>4}")
            continue
        world = pitch_xy[[kp.index for kp in confident]]
        xs, ys = world[:, 0], world[:, 1]

        n_left  = int((xs < 40).sum())
        n_mid   = int(((xs >= 40) & (xs <= 80)).sum())
        n_right = int((xs > 80).sum())

        # "Spread" score = how much of the pitch x-axis is represented.
        # Reward frames that see all 3 thirds.
        thirds_seen = (n_left > 0) + (n_mid > 0) + (n_right > 0)
        spread = thirds_seen * 1000.0 + (xs.max() - xs.min())

        if spread > best_spread:
            best_spread = spread
            best_frame  = fp

        print(f"{int(fp.stem):>6}  {len(confident):>4}  "
              f"{xs.min():>5.1f}  {xs.max():>5.1f}  "
              f"{ys.min():>5.1f}  {ys.max():>5.1f}  "
              f"{n_left:>4}  {n_mid:>3}  {n_right:>5}")

    print()
    if best_frame is not None:
        print(f"[best spread] frame {int(best_frame.stem)}  "
              f"score={best_spread:.1f}")


if __name__ == "__main__":
    main()
