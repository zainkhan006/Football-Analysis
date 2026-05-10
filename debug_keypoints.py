"""
Diagnostic: visualise the raw 32 YOLO keypoint outputs on a frame, labelled
with both interpretations of which vertex each output position corresponds
to.

Usage:
    python debug_keypoints.py --seq v_dw7LOz17Omg_c053 --frame 79

Outputs `debug_keypoints_<seq>_<frame>.png` showing:
  RED label  = YOLO output position 0..31 (raw, no permutation)
  GREEN label = vertex number under the "labels-ordering" hypothesis
                (what the Roboflow `labels` list says)

If GREEN labels match the actual landmark ('centre-circle bottom = 15',
'halfway-line bottom touchline = 14', etc.) the labels permutation is
correct.  If RED labels match instead, the model uses `vertices` order
directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Same permutation as in pitch_keypoint_detector
ROBOFLOW_KP_TO_VERTEX_IDX = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12,
    14, 15, 16, 17,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    13, 18,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img_path = Path("dataset/train") / args.seq / "img1" / f"{args.frame:06d}.jpg"
    if not img_path.exists():
        raise SystemExit(f"frame not found: {img_path}")
    img = cv2.imread(str(img_path))

    model = YOLO("models/football-pitch-detection.pt")
    res = model(img, imgsz=640, conf=args.conf, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.xy) == 0:
        raise SystemExit("no keypoints detected")

    xy   = res.keypoints.xy.cpu().numpy()[0]
    conf = res.keypoints.conf.cpu().numpy()[0]

    # Sort YOLO output indices by ascending pixel x, just for table readability
    order = np.argsort(conf)[::-1]   # by confidence descending
    print(f"\nFrame {args.frame} of {args.seq}:  detected {(conf >= 0.5).sum()} "
          f"high-conf KPs out of 32")
    print()
    print(f"{'yolo_idx':>8}  {'px':>6}  {'py':>6}  {'conf':>5}  "
          f"{'vertex (labels-perm)':>22}  {'vertex (raw)':>13}")
    print("-" * 80)
    for yolo_idx in order:
        px, py = xy[yolo_idx]
        c      = conf[yolo_idx]
        if c < 0.3:
            continue
        v_perm = ROBOFLOW_KP_TO_VERTEX_IDX[yolo_idx] + 1   # 1-indexed for clarity
        v_raw  = yolo_idx + 1
        print(f"{yolo_idx:>8}  {px:6.1f}  {py:6.1f}  {c:.2f}  "
              f"{v_perm:>22}  {v_raw:>13}")

    # Draw both interpretations on the image
    out = img.copy()
    H, W = out.shape[:2]
    for yolo_idx in range(32):
        c = conf[yolo_idx]
        if c < 0.3:
            continue
        px, py = float(xy[yolo_idx, 0]), float(xy[yolo_idx, 1])
        if not (0 <= px < W and 0 <= py < H):
            continue

        # Dot
        cv2.circle(out, (int(px), int(py)), 5, (0, 255, 255), -1)

        # Raw YOLO position (red)
        cv2.putText(out, f"r{yolo_idx + 1}",
                    (int(px) + 6, int(py) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Label-permuted vertex (green)
        v_perm = ROBOFLOW_KP_TO_VERTEX_IDX[yolo_idx] + 1
        cv2.putText(out, f"g{v_perm}",
                    (int(px) + 6, int(py) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)

    out_path = args.out or f"debug_keypoints_{args.seq}_{args.frame}.png"
    cv2.imwrite(out_path, out)
    print(f"\n[saved] {out_path}")
    print("Open the image and compare:")
    print("  - red 'r##' labels = raw YOLO output position (1-indexed)")
    print("  - green 'g##' labels = vertex under labels permutation")
    print("Find a clearly-identifiable landmark (centre-circle bottom,")
    print("halfway-line bottom touchline, corner flags) and check which")
    print("interpretation matches reality.")


if __name__ == "__main__":
    main()
