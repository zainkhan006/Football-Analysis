"""
Render an overlay video from a saved homographies_v2/*.npz file.

Independent of the solver pipeline: only needs the per-frame Hs on disk
and the original image files.

Usage:
    python render_solved_video.py --seq v_gQNyhv8y0QY_c013
    python render_solved_video.py --seq v_gQNyhv8y0QY_c013 --step 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from render_track_video import draw_overlay


DATASET_TRAIN = Path("dataset/train")
HOM_DIR       = Path("homographies_v2")

SRC_COLOUR = {
    "yolo":   (0, 255,   0),    # green
    "hough":  (0, 200, 255),    # amber
    "interp": (0, 120, 255),    # orange
    "copy":   (200, 80, 200),   # purple
    "fail":   (0,   0, 255),    # red
    "":       (0,   0, 255),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    npz_path = HOM_DIR / f"{args.seq}_per_frame_H.npz"
    data = np.load(npz_path, allow_pickle=False)
    frame_ids = data["frame_ids"]
    H_all     = data["H_all"]
    sane      = data["sane"]
    sources   = data["source"]
    n_yolo    = data["n_yolo_kp"]
    n_hough_s = data["n_hough_segs"]
    n_matched = data["n_matched_lines"]
    reproj    = data["reproj_px"]

    img_dir = DATASET_TRAIN / args.seq / "img1"
    first_path = img_dir / f"{int(frame_ids[0]):06d}.jpg"
    first = cv2.imread(str(first_path))
    if first is None:
        raise SystemExit(f"cannot read {first_path}")
    h_img, w_img = first.shape[:2]

    out_path = Path(args.out) if args.out else \
               HOM_DIR / f"{args.seq}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w_img, h_img))

    print(f"[render] {args.seq}: {len(frame_ids)} frames -> {out_path}")
    for i, fid in enumerate(frame_ids):
        if (i % args.step) != 0:
            continue
        fp = img_dir / f"{int(fid):06d}.jpg"
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        src = str(sources[i])
        col = SRC_COLOUR.get(src, (0, 0, 255))
        if sane[i] and np.all(np.isfinite(H_all[i])):
            draw_overlay(frame, H_all[i], colour=col)
            tag = (f"f{int(fid):>6}  src={src:<6}  "
                   f"kp={int(n_yolo[i]):>2}  "
                   f"hough={int(n_hough_s[i]):>3}  "
                   f"matched={int(n_matched[i]):>2}  "
                   f"reproj={float(reproj[i]):5.1f}px"
                   if np.isfinite(reproj[i]) else
                   f"f{int(fid):>6}  src={src:<6}  (no reproj metric)")
        else:
            tag = f"f{int(fid):>6}  UNSOLVED"
        cv2.rectangle(frame, (0, 0), (w_img, 32), (0, 0, 0), -1)
        cv2.putText(frame, tag, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[render] wrote {out_path}")


if __name__ == "__main__":
    main()
