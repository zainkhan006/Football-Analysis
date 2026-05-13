"""
Diagnostic for the Hough-line fallback idea.

For each (seq, frame_id) pair in SAMPLES:
    1. Run YOLO pitch-keypoint detection -> note how many confident keypoints
    2. Build the white-on-green line mask
    3. Subtract player bboxes (from SportsMOT gt.txt) so jerseys don't leak in
    4. Run Probabilistic Hough on the cleaned mask
    5. Save a 2x2 panel:
         [raw frame + yolo keypoints] | [line mask with player boxes blanked]
         [hough line segments overlaid on frame] | [hough on black canvas]

Purpose: verify that Hough actually finds sensible pitch-line segments on
these SportsMOT frames before we commit to building the full matching + H-fit
pipeline. If the Hough output is dominated by noise (ad boards, stadium
structure, grass stripes), we need to rethink before scaling.

Output: debug_hough/<seq>_<frame>.jpg -- one panel per sampled frame.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from pitch_keypoint_detector import PitchKeypointDetector
from pitch_mask import build_pitch_mask
from refine_homography import build_line_mask


DATASET_TRAIN = Path("dataset/train")
OUT_DIR       = Path("debug_hough")


# ──────────────────────────────────────────────────────────────────────────────
#  Sample frames to diagnose
# ──────────────────────────────────────────────────────────────────────────────
#
# Mix of:
#   - calibration-frame candidates (likely easier)
#   - mid-clip frames where the per-frame tracker drifted (likely harder)
#   - clips with Wembley-style tight zooms
#
SAMPLES: list[tuple[str, int]] = [
    # Stage-A v2: with pitch-interior gating. Cover all 3 representative clips.
    ("v_gQNyhv8y0QY_c013",   1),
    ("v_gQNyhv8y0QY_c013", 266),
    ("v_gQNyhv8y0QY_c013", 446),
    ("v_gQNyhv8y0QY_c013", 626),
    ("v_dw7LOz17Omg_c053",   1),
    ("v_dw7LOz17Omg_c053", 160),
    ("v_dw7LOz17Omg_c053", 240),
    ("v_dw7LOz17Omg_c053", 400),
    ("v_1yHWGw8DH4A_c047",   1),
    ("v_1yHWGw8DH4A_c047", 240),
    ("v_1yHWGw8DH4A_c047", 320),
    ("v_1yHWGw8DH4A_c047", 400),
]


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_bboxes(seq_dir: Path) -> dict[int, list[tuple[int, int, int, int]]]:
    """Return {frame_id: [(x, y, w, h), ...]} of player bboxes."""
    out: dict = defaultdict(list)
    gt = seq_dir / "gt" / "gt.txt"
    if not gt.exists():
        return {}
    for line in gt.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            fid = int(parts[0])
            x   = int(float(parts[2]))
            y   = int(float(parts[3]))
            w   = int(float(parts[4]))
            h   = int(float(parts[5]))
        except ValueError:
            continue
        out[fid].append((x, y, w, h))
    return dict(out)


def subtract_player_bboxes(
    mask: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    pad_px: int = 4,
) -> np.ndarray:
    """Zero out rectangles around each player bbox (padded slightly)."""
    out = mask.copy()
    H, W = mask.shape[:2]
    for x, y, w, h in bboxes:
        x0 = max(0, x - pad_px)
        y0 = max(0, y - pad_px)
        x1 = min(W, x + w + pad_px)
        y1 = min(H, y + h + pad_px)
        out[y0:y1, x0:x1] = 0
    return out


def run_hough(mask: np.ndarray,
              rho:            float = 1.0,
              theta:          float = np.pi / 360.0,   # 0.5 deg
              threshold:      int   = 60,
              min_line_len:   int   = 40,
              max_line_gap:   int   = 12,
              ) -> np.ndarray:
    """
    Probabilistic Hough on a binary line mask.
    Returns (N, 4) float32 [[x1, y1, x2, y2], ...] or empty array.
    """
    segs = cv2.HoughLinesP(
        mask, rho, theta, threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if segs is None:
        return np.zeros((0, 4), dtype=np.float32)
    return segs.reshape(-1, 4).astype(np.float32)


def draw_segments(
    img: np.ndarray,
    segs: np.ndarray,
    colour: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    for x1, y1, x2, y2 in segs:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 colour, thickness, cv2.LINE_AA)


def panelise(panels: list[tuple[str, np.ndarray]],
             grid: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Stack panels into a grid image, each panel labelled at top-left."""
    rows, cols = grid
    assert rows * cols == len(panels), f"need {rows*cols} panels, got {len(panels)}"

    # Force all panels to same shape (use first panel's shape)
    H, W = panels[0][1].shape[:2]
    for i in range(len(panels)):
        title, img = panels[i]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        # stamp title
        cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(img, title, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        panels[i] = (title, img)

    row_imgs = []
    for r in range(rows):
        strip = np.hstack([panels[r * cols + c][1] for c in range(cols)])
        row_imgs.append(strip)
    return np.vstack(row_imgs)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def process_frame(seq: str, fid: int, detector: PitchKeypointDetector) -> None:
    seq_dir = DATASET_TRAIN / seq
    frame_path = seq_dir / "img1" / f"{fid:06d}.jpg"
    if not frame_path.exists():
        print(f"  [skip] {seq} frame {fid}: file not found")
        return
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"  [skip] {seq} frame {fid}: could not read")
        return

    H, W = frame.shape[:2]

    # 1. YOLO keypoints
    kp_res = detector.detect(frame, conf_threshold=0.30)
    confident = kp_res.confident(threshold=0.5)
    kp_img = frame.copy()
    for kp in kp_res.keypoints:
        x, y = kp.pixel_xy
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        # colour by confidence: red (low) -> green (high)
        c = max(0.0, min(1.0, kp.confidence))
        col = (0, int(255 * c), int(255 * (1 - c)))
        cv2.circle(kp_img, (int(x), int(y)), 5, col, -1)
        cv2.circle(kp_img, (int(x), int(y)), 7, (0, 0, 0), 1)
    cv2.putText(kp_img,
                f"YOLO kp >=0.5: {len(confident)}/32",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Line mask + pitch-interior gate
    mask_raw    = build_line_mask(frame)
    pitch_mask  = build_pitch_mask(frame)
    mask_gated  = cv2.bitwise_and(mask_raw, pitch_mask)

    # 3. Subtract player bboxes (post-pitch-gate, so we keep mask thin)
    gt = load_gt_bboxes(seq_dir)
    bboxes = gt.get(fid, [])
    mask_clean = subtract_player_bboxes(mask_gated, bboxes, pad_px=6)

    # 4. Hough on the pitch-gated mask
    segs = run_hough(mask_clean)

    # 5. Render panels
    frame_with_segs = frame.copy()
    draw_segments(frame_with_segs, segs, colour=(0, 255, 255), thickness=2)
    cv2.putText(frame_with_segs,
                f"Hough segments (gated): {len(segs)}",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)

    # Pitch-mask overlay panel
    pitch_overlay = frame.copy()
    pitch_color = np.zeros_like(frame)
    pitch_color[pitch_mask > 0] = (0, 180, 0)
    pitch_overlay = cv2.addWeighted(pitch_overlay, 0.6, pitch_color, 0.4, 0)
    cv2.putText(pitch_overlay,
                "pitch-interior mask (HSV+CC+fill)",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2, cv2.LINE_AA)

    # Render cleaned mask in BGR
    mask_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
    # draw player-bbox subtraction regions in dim red for context
    for x, y, w, h in bboxes:
        cv2.rectangle(mask_vis, (x, y), (x + w, y + h), (0, 0, 120), 1)

    panel = panelise([
        (f"{seq}  frame {fid}   (raw + YOLO keypoints)", kp_img),
        ("pitch-interior mask",                          pitch_overlay),
        ("white-line mask  (pitch-gated + players blanked)", mask_vis),
        ("Hough segments overlaid on frame",             frame_with_segs),
    ], grid=(2, 2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{seq}_f{fid:06d}.jpg"
    cv2.imwrite(str(out_path), panel)
    print(f"  [ok] {seq} f{fid:<6}  yolo_conf={len(confident):>2}  "
          f"hough_segs={len(segs):>3}  -> {out_path}")


def main():
    print(f"[hough-test] processing {len(SAMPLES)} sample frames...")
    detector = PitchKeypointDetector(verbose=False)
    for seq, fid in SAMPLES:
        try:
            process_frame(seq, fid, detector)
        except Exception as e:
            print(f"  [err] {seq} f{fid}: {e}")
    print(f"[hough-test] done. panels saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
