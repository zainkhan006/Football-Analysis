"""
Render the per-frame H as a video overlay so you can watch the tracker
work without stepping through 800 frames manually.

Usage:
    python render_track_video.py --seq v_gQNyhv8y0QY_c013
    python render_track_video.py                       # all clips that have _track.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pitch_config import DEFAULT_PITCH


DATASET_TRAIN_DIR = Path("dataset/train")
HOMOGRAPHY_DIR    = Path("homographies")
OUT_DIR           = Path("debug_track_videos")


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

    for seg in [((0, 0), (L, 0)), ((L, 0), (L, W)),
                ((L, W), (0, W)), ((0, W), (0, 0)),
                ((L / 2, 0), (L / 2, W))]:
        ln(*seg, t=2)
    for box in [
        [(0, (W - pw) / 2), (pl, (W - pw) / 2),
         (pl, (W + pw) / 2), (0, (W + pw) / 2)],
        [(0, (W - gw) / 2), (gl, (W - gw) / 2),
         (gl, (W + gw) / 2), (0, (W + gw) / 2)],
        [(L, (W - pw) / 2), (L - pl, (W - pw) / 2),
         (L - pl, (W + pw) / 2), (L, (W + pw) / 2)],
        [(L, (W - gw) / 2), (L - gl, (W - gw) / 2),
         (L - gl, (W + gw) / 2), (L, (W + gw) / 2)],
    ]:
        for a, b in zip(box[:-1], box[1:]):
            ln(a, b, t=1)
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=True)
    pts = np.stack([L / 2 + r * np.cos(ang), W / 2 + r * np.sin(ang)], 1)
    for i in range(len(pts) - 1):
        ln(pts[i], pts[i + 1], t=1)


def render_clip(seq_name: str, fps: float = 30.0) -> None:
    track_path = HOMOGRAPHY_DIR / f"{seq_name}_track.npz"
    if not track_path.exists():
        print(f"[render] no track file for {seq_name}; skip")
        return
    seq_dir = DATASET_TRAIN_DIR / seq_name
    frames  = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frames:
        print(f"[render] no frames for {seq_name}; skip")
        return

    data = np.load(track_path)
    H_per = data["H_per_frame"]
    costs = data["costs"]
    rese  = data["reseeded"]
    fids  = data["frame_ids"]
    fid_to_idx = {int(fid): i for i, fid in enumerate(fids)}

    OUT_DIR.mkdir(exist_ok=True)
    first = cv2.imread(str(frames[0]))
    h_img, w_img = first.shape[:2]

    out_path = OUT_DIR / f"{seq_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_img, h_img))

    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        j = fid_to_idx.get(int(fp.stem))
        if j is None:
            writer.write(img)
            continue
        H = H_per[j]
        cost = float(costs[j])
        is_reseed = bool(rese[j])
        # green by default; orange where the tracker reseeded that frame
        colour = (0, 165, 255) if is_reseed else (0, 255, 0)
        draw_overlay(img, H, colour)
        cv2.putText(img, f"frame {fp.stem}  cost={cost:.1f}px"
                    + ("  RESEED" if is_reseed else ""),
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        writer.write(img)

    writer.release()
    n = len(frames)
    n_re = int(rese.sum())
    mean_cost = float(costs[costs >= 0].mean()) if (costs >= 0).any() else -1
    print(f"[render] {seq_name}  {n} frames  "
          f"mean_cost={mean_cost:.2f}px  reseeded={n_re}  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.seq:
        render_clip(args.seq, fps=args.fps)
    else:
        for p in HOMOGRAPHY_DIR.glob("*_track.npz"):
            render_clip(p.stem.replace("_track", ""), fps=args.fps)


if __name__ == "__main__":
    main()
