"""
For a handful of failing-sanity frames from the per-frame-yolo test,
render a diagnostic panel showing:
  - frame with confident YOLO keypoints marked
  - the H that was computed (even though it failed sanity)
  - the reason it failed

This tells us WHERE the keypoints are concentrated on these frames,
so we know what shape of fix is needed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector
from render_track_video import draw_overlay
from test_per_frame_yolo_h import fit_H_from_keypoints, _is_H_sane


SEQ = "v_gQNyhv8y0QY_c013"

# Pick a few failing frames across the clip
SAMPLE_FIDS = [1, 66, 126, 156, 226, 351, 451, 551, 651]

OUT = Path("debug_per_frame_yolo") / f"{SEQ}_failures_detail.jpg"

def annotate_fail_reason(H):
    """Return human-readable reason H failed sanity, or 'OK' if it passed."""
    if H is None:
        return "H is None"
    if not np.all(np.isfinite(H)):
        return "H has NaN/Inf"
    L, W = DEFAULT_PITCH.length, DEFAULT_PITCH.width
    corners = np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=np.float64)
    homog = np.hstack([corners, np.ones((4, 1))]).T
    proj = H @ homog
    w = proj[2, :]
    if np.any(np.abs(w) < 1e-9):
        return "H near-singular (w==0)"
    xy = (proj[:2, :] / w).T
    if not np.all(np.isfinite(xy)):
        return "projected corners non-finite"

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    signs = [np.sign(cross(xy[i], xy[(i+1)%4], xy[(i+2)%4])) for i in range(4)]
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return "projected pitch is self-intersecting"

    area = 0.5 * abs(xy[0,0]*(xy[1,1]-xy[3,1]) + xy[1,0]*(xy[2,1]-xy[0,1])
                   + xy[2,0]*(xy[3,1]-xy[1,1]) + xy[3,0]*(xy[0,1]-xy[2,1]))
    return f"OK (proj-area={area:.0f}px2)"


def main():
    detector = PitchKeypointDetector(verbose=False)
    panels = []
    for fid in SAMPLE_FIDS:
        path = Path(f"dataset/train/{SEQ}/img1/{fid:06d}.jpg")
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        kp = detector.detect(frame, conf_threshold=0.30)
        H, info = fit_H_from_keypoints(kp, frame.shape)

        # Draw all confident kps
        img = frame.copy()
        for k in kp.keypoints:
            if k.confidence < 0.5:
                continue
            x, y = k.pixel_xy
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cv2.circle(img, (int(x), int(y)), 7, (0, 255, 0), -1)
            cv2.circle(img, (int(x), int(y)), 9, (0, 0, 0), 1)

        # Compute H (even if sanity-failed) to visualise what RANSAC wants to do.
        # We re-run findHomography here because fit_H_from_keypoints returns None
        # on sanity failure.
        world = []
        pixel = []
        vertices = DEFAULT_PITCH.vertices_m
        for k in kp.keypoints:
            if k.confidence < 0.5:
                continue
            x, y = k.pixel_xy
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            world.append(vertices[k.index])
            pixel.append((x, y))
        H_raw = None
        if len(world) >= 4:
            world_np = np.array(world, dtype=np.float64)
            pixel_np = np.array(pixel, dtype=np.float64)
            H_raw, _ = cv2.findHomography(world_np, pixel_np,
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=8.0)

        if H_raw is not None:
            draw_overlay(img, H_raw, colour=(0, 255, 255))

        fr = annotate_fail_reason(H_raw)
        status = "OK (sane)" if info.get("sane") else f"FAIL: {info.get('fail_reason')}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 66), (0, 0, 0), -1)
        cv2.putText(img,
                    f"frame {fid}   kp>=0.5: {info['n_confident']}   "
                    f"ransac_in: {info['n_ransac_inliers']}   "
                    f"reproj: {info['mean_reproj_px']:.2f}px",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img,
                    f"sanity: {status}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255) if info.get("sane") else (0, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"corner-proj: {fr}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        panels.append(cv2.resize(img, None, fx=0.5, fy=0.5))

    # Grid 3x3
    n = len(panels)
    rows_n = 3
    cols_n = 3
    while len(panels) < rows_n * cols_n:
        panels.append(np.zeros_like(panels[0]))
    h, w = panels[0].shape[:2]
    rows = [np.hstack(panels[r*cols_n:(r+1)*cols_n]) for r in range(rows_n)]
    grid = np.vstack(rows)
    cv2.imwrite(str(OUT), grid)
    print(f"wrote {OUT}  shape={grid.shape}")

if __name__ == "__main__":
    main()
