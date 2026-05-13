"""
For each frame in the clip, compute the KEYPOINT SPREAD in pixel space
(convex-hull area) and correlate with sanity success/failure.

Hypothesis: sanity-failing frames have keypoints concentrated in a small
pixel region. If that's true, the fix is clear: require a minimum
pixel spread before even attempting H.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector

SEQ = "v_gQNyhv8y0QY_c013"


def convex_hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull))


def main():
    detector = PitchKeypointDetector(verbose=False)
    frames = sorted(Path(f"dataset/train/{SEQ}/img1").glob("*.jpg"))
    rows = list(csv.DictReader(open(f"debug_per_frame_yolo/{SEQ}_stats.csv")))
    stats_map = {int(r["frame_id"]): r for r in rows}

    # Re-run YOLO on the stepped frames to get pixel coords
    print("frame,n_kp,hull_area_px,hull_frac_img,bbox_w_px,bbox_h_px,sane")
    for fp in frames[::5]:
        fid = int(fp.stem)
        if fid not in stats_map:
            continue
        row = stats_map[fid]
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        h_img, w_img = frame.shape[:2]
        kp = detector.detect(frame, conf_threshold=0.30)
        pts = []
        for k in kp.keypoints:
            if k.confidence < 0.5:
                continue
            x, y = k.pixel_xy
            if np.isfinite(x) and np.isfinite(y):
                pts.append((x, y))
        if not pts:
            print(f"{fid},0,0,0.00,0,0,0")
            continue
        pts_np = np.array(pts, dtype=np.float32)
        hull_a = convex_hull_area(pts_np)
        bbox = pts_np.max(axis=0) - pts_np.min(axis=0)
        print(f"{fid},{len(pts)},{hull_a:.0f},"
              f"{hull_a/(h_img*w_img):.3f},"
              f"{bbox[0]:.0f},{bbox[1]:.0f},{row['sane']}")


if __name__ == "__main__":
    main()
