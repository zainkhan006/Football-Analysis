"""
Debug harness for the distance-transform homography refinement.

For one clip:
  1. Loads the existing (YOLO-keypoint) homography.
  2. Builds a line mask from the calibration frame.
  3. Projects the canonical pitch with the OLD H → saves "before.jpg".
  4. Refines H via `refine_homography` → saves "after.jpg".
  5. Saves the line mask as "mask.jpg" so you can see what the refiner saw.

Run:
    python debug_line_refine.py v_dw7LOz17Omg_c053
    python debug_line_refine.py v_gQNyhv8y0QY_c013
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from homography import PitchHomography
from pitch_config import DEFAULT_PITCH
from refine_homography import (build_line_mask, refine_homography,
                               sample_canonical_pitch)


DATASET_TRAIN_DIR = Path("dataset/train")
HOMOGRAPHY_DIR    = Path("homographies")
OUT_DIR           = Path("debug_refine")


# ── Drawing ─────────────────────────────────────────────────────────────────
def _safe(p, shape):
    Hh, Ww = shape[:2]
    x, y = p
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if x < -Ww or x > 2 * Ww or y < -Hh or y > 2 * Hh:
        return None
    return int(round(x)), int(round(y))


def _project_H(H: np.ndarray, w):
    pt = np.array([[[w[0], w[1]]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def draw_pitch_with_H(frame: np.ndarray, H: np.ndarray, colour) -> None:
    """Minimal pitch overlay using a raw 3x3 H matrix."""
    cfg = DEFAULT_PITCH
    L, W = cfg.length, cfg.width
    pl, pw = cfg.penalty_box_length, cfg.penalty_box_width
    gl, gw = cfg.goal_box_length, cfg.goal_box_width
    r = cfg.centre_circle_radius

    def line(a, b, col, thk=2):
        p0 = _safe(_project_H(H, a), frame.shape)
        p1 = _safe(_project_H(H, b), frame.shape)
        if p0 and p1:
            cv2.line(frame, p0, p1, col, thk, cv2.LINE_AA)

    # Outline
    outline = [(0, 0), (L, 0), (L, W), (0, W), (0, 0)]
    for a, b in zip(outline[:-1], outline[1:]):
        line(a, b, colour, 2)
    # Halfway
    line((L / 2, 0), (L / 2, W), colour, 1)
    # Left penalty
    for a, b in zip(
        [(0, (W - pw) / 2), (pl, (W - pw) / 2),
         (pl, (W + pw) / 2), (0, (W + pw) / 2)],
        [(pl, (W - pw) / 2), (pl, (W + pw) / 2),
         (0, (W + pw) / 2)]
    ):
        line(a, b, colour, 1)
    # Right penalty
    for a, b in zip(
        [(L, (W - pw) / 2), (L - pl, (W - pw) / 2),
         (L - pl, (W + pw) / 2), (L, (W + pw) / 2)],
        [(L - pl, (W - pw) / 2), (L - pl, (W + pw) / 2),
         (L, (W + pw) / 2)]
    ):
        line(a, b, colour, 1)
    # Left goal box
    for a, b in zip(
        [(0, (W - gw) / 2), (gl, (W - gw) / 2),
         (gl, (W + gw) / 2), (0, (W + gw) / 2)],
        [(gl, (W - gw) / 2), (gl, (W + gw) / 2),
         (0, (W + gw) / 2)]
    ):
        line(a, b, colour, 1)
    # Right goal box
    for a, b in zip(
        [(L, (W - gw) / 2), (L - gl, (W - gw) / 2),
         (L - gl, (W + gw) / 2), (L, (W + gw) / 2)],
        [(L - gl, (W - gw) / 2), (L - gl, (W + gw) / 2),
         (L, (W + gw) / 2)]
    ):
        line(a, b, colour, 1)
    # Centre circle
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=True)
    pts = np.stack([L / 2 + r * np.cos(angles), W / 2 + r * np.sin(angles)], 1)
    for i in range(len(pts) - 1):
        p0 = _safe(_project_H(H, pts[i]), frame.shape)
        p1 = _safe(_project_H(H, pts[i + 1]), frame.shape)
        if p0 and p1:
            cv2.line(frame, p0, p1, colour, 1, cv2.LINE_AA)


# ── Main ────────────────────────────────────────────────────────────────────
def main(seq_name: str) -> None:
    homo_path = HOMOGRAPHY_DIR / f"{seq_name}.npz"
    if not homo_path.exists():
        sys.exit(f"no homography at {homo_path}")

    H_obj = PitchHomography.load(homo_path)
    frame_id = H_obj.source_frame_id
    if frame_id is None:
        sys.exit("homography has no source_frame_id")

    frame_path = DATASET_TRAIN_DIR / seq_name / "img1" / f"{frame_id:06d}.jpg"
    if not frame_path.exists():
        sys.exit(f"missing frame {frame_path}")

    frame = cv2.imread(str(frame_path))
    print(f"[debug] {seq_name}  frame={frame_id}  shape={frame.shape}")
    print(f"[debug] seed H: reproj_err={H_obj.fit_error_px:.2f}px  "
          f"N={H_obj.n_correspondences}")

    # 1. Line mask
    mask = build_line_mask(frame)
    print(f"[debug] line mask: {int((mask > 0).sum())} white px "
          f"({100 * (mask > 0).mean():.2f}% of frame)")

    # 2. Canonical samples
    samples = sample_canonical_pitch(step_m=1.0)
    print(f"[debug] canonical pitch samples: {len(samples)}")

    # 3. Refine
    H_refined, info = refine_homography(
        H_obj.H_world_to_pixel, mask, samples, max_px=30.0
    )
    print(f"[debug] refine: cost {info.cost_before:.2f}px -> "
          f"{info.cost_after:.2f}px   "
          f"(n_active={info.n_active}/{len(samples)}  "
          f"iters={info.iterations})")

    # 4. Render overlays
    OUT_DIR.mkdir(exist_ok=True)
    before = frame.copy()
    after  = frame.copy()
    draw_pitch_with_H(before, H_obj.H_world_to_pixel, (0, 0, 255))     # RED = before
    draw_pitch_with_H(after,  H_refined,              (0, 255, 0))     # GREEN = after

    # Side-by-side for easy A/B
    side = np.hstack([before, after])
    label = np.zeros((40, side.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, f"BEFORE (seed H, cost={info.cost_before:.2f}px)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(label, f"AFTER  (refined H, cost={info.cost_after:.2f}px)",
                (frame.shape[1] + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    side = np.vstack([label, side])

    cv2.imwrite(str(OUT_DIR / f"{seq_name}_before.jpg"), before)
    cv2.imwrite(str(OUT_DIR / f"{seq_name}_after.jpg"),  after)
    cv2.imwrite(str(OUT_DIR / f"{seq_name}_mask.jpg"),   mask)
    cv2.imwrite(str(OUT_DIR / f"{seq_name}_side.jpg"),   side)
    print(f"[debug] wrote {OUT_DIR}/{seq_name}_{{before,after,mask,side}}.jpg")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python debug_line_refine.py <seq_name>")
    main(sys.argv[1])
