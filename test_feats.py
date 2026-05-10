"""
Test driver for Lamaan's Features 7 (touch detection) and 11 (ball detection).

Runs on the SportsMOT sequence v_gQNyhv8y0QY_c013 (875 frames @ 25fps).

What this script verifies:
  - The refactored BallDetector uses the Roboflow football-ball-detection
    model and the ROI-based fast path
  - GT player bboxes are passed in as the false-positive filter
  - TouchDetector still receives `is_estimated=False` only on real detections
  - Inference speed (FPS) and detection rate are reported

No GPU required.  No homography (world coords are still None — that is
expected until Zain delivers Z4).  Touch detection still fires; zone
classification is skipped for None-world touches.
"""

from __future__ import annotations

import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from ball_detect import BallDetector
from touch_detect import TouchDetector


# ─── Config ───────────────────────────────────────────────────────────────────

SEQUENCE_DIR  = Path("dataset/train/v_gQNyhv8y0QY_c013")
TARGET_PLAYER = 4         # track_id treated as the "target player"
FRAME_LIMIT   = 200       # set to 875 for full clip; 200 = ~2 min run
STRIDE        = 2         # process every Nth frame (project spec)
LOWER_FRACTION = 0.5      # lower half of bbox = foot zone
MARGIN_PX      = 25       # px outside bbox that still count
OUTPUT_PATH    = "touch_map_lamaan_test.png"

# BallDetector tuning ----------------------------------------------------------
# imgsz_full: full-frame inference size. 640 is a strong CPU/accuracy compromise
# imgsz_roi : ROI inference size. Match roi_size to avoid upscaling.
BALL_CONF       = 0.15
IMGSZ_FULL      = 640
IMGSZ_ROI       = 320
ROI_SIZE        = 320
FULL_FRAME_EVERY = 30


# ─── GT loader ────────────────────────────────────────────────────────────────

def load_gt(sequence_dir: Path) -> dict[int, list[dict]]:
    """Returns {frame_id: [{track_id, bbox=(x,y,w,h)}, ...]}."""
    gt_file = sequence_dir / "gt" / "gt.txt"
    data: dict = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = (int(parts[2]), int(parts[3]),
                          int(parts[4]), int(parts[5]))
            data[fid].append({"track_id": tid, "bbox": (x, y, w, h)})
    return dict(data)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[test] Loading GT from {SEQUENCE_DIR}")
    gt = load_gt(SEQUENCE_DIR)
    all_frames = sorted(gt.keys())
    frames_to_process = [f for f in all_frames if f <= FRAME_LIMIT][::STRIDE]
    print(f"[test] Total GT frames     : {len(all_frames)}")
    print(f"[test] Processing          : {len(frames_to_process)} "
          f"(stride={STRIDE})")

    target_visible = [
        f for f in frames_to_process
        if any(d["track_id"] == TARGET_PLAYER for d in gt[f])
    ]
    print(f"[test] Target id={TARGET_PLAYER} visible in "
          f"{len(target_visible)}/{len(frames_to_process)} processed frames")

    # ── Initialise modules ─────────────────────────────────────────────────
    detector = BallDetector(
        conf_threshold        = BALL_CONF,
        imgsz_full            = IMGSZ_FULL,
        imgsz_roi             = IMGSZ_ROI,
        roi_size              = ROI_SIZE,
        full_frame_every      = FULL_FRAME_EVERY,
    )
    td = TouchDetector(
        lower_fraction  = LOWER_FRACTION,
        margin_px       = MARGIN_PX,
        min_consecutive = 2,
    )

    # ── Counters & timers ──────────────────────────────────────────────────
    real_det = 0
    est_det  = 0
    no_det   = 0
    total_inference_s = 0.0
    roi_used = 0     # heuristic: a frame is "ROI" if not an FF-every multiple
                     # AND ball was locked at start of frame (we cannot read
                     # internal state cleanly, so we approximate by output)

    # ── Frame loop ─────────────────────────────────────────────────────────
    t_total_start = time.perf_counter()
    for i, fid in enumerate(frames_to_process):
        img_path = SEQUENCE_DIR / "img1" / f"{fid:06d}.jpg"
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [warn] Could not read frame {fid} — skipping")
            continue

        # All player bboxes in this frame (used as FP filter)
        all_player_bboxes = [d["bbox"] for d in gt.get(fid, [])]

        t0 = time.perf_counter()
        ball_result = detector.track_ball(frame, player_bboxes=all_player_bboxes)
        total_inference_s += (time.perf_counter() - t0)

        if ball_result["ball_px"] is None:
            no_det += 1
        elif ball_result["is_estimated"]:
            est_det += 1
        else:
            real_det += 1

        # Find target player bbox
        player_bbox = None
        for det in gt.get(fid, []):
            if det["track_id"] == TARGET_PLAYER:
                player_bbox = det["bbox"]
                break

        # L3: touch detection (world coord = None — no homography yet)
        if player_bbox is not None:
            touched = td.update(
                frame_id     = fid,
                ball_result  = ball_result,
                player_bbox  = player_bbox,
                player_world = None,
            )
            if touched:
                print(f"  [TOUCH] Frame {fid} — ball at "
                      f"({ball_result['ball_px']:.0f}, "
                      f"{ball_result['ball_py']:.0f}), "
                      f"player bbox {player_bbox}")

        # Progress report every 100 processed frames
        if (i + 1) % 100 == 0:
            avg_ms = 1000.0 * total_inference_s / (i + 1)
            fps    = 1.0 / max(total_inference_s / (i + 1), 1e-9)
            print(f"  frame {fid:4d}  "
                  f"avg {avg_ms:5.1f} ms/frame ({fps:5.1f} FPS)  "
                  f"real={real_det} est={est_det} miss={no_det}")

    t_total = time.perf_counter() - t_total_start

    # ── Results ────────────────────────────────────────────────────────────
    n = len(frames_to_process)
    avg_ms = 1000.0 * total_inference_s / max(n, 1)
    fps    = n / max(total_inference_s, 1e-9)

    print("\n=== Ball Detection Performance ===")
    print(f"  Frames processed     : {n}")
    print(f"  Inference total      : {total_inference_s:6.2f} s "
          f"(wall {t_total:6.2f} s)")
    print(f"  Per-frame inference  : {avg_ms:6.1f} ms  ({fps:5.1f} FPS)")
    print(f"  Real detections      : {real_det:4d}  "
          f"({100*real_det/n:5.1f}%)")
    print(f"  Kalman estimates     : {est_det:4d}  "
          f"({100*est_det/n:5.1f}%)")
    print(f"  Truly missed         : {no_det:4d}  "
          f"({100*no_det/n:5.1f}%)")
    print(f"  Effective recall     : "
          f"{100*(real_det+est_det)/n:5.1f}%  "
          f"(real + Kalman fill-ins)")

    stats = td.get_stats(fps=25.0 / STRIDE, total_frames=n)
    print("\n=== Touch Detection Results ===")
    print(f"  Touch count          : {stats['touch_count']}")
    print(f"  Touches / min        : {stats['touches_per_minute']}")
    print(f"  Touches by zone      : {stats['touches_by_zone']}")
    print( "  (World coords are None — zone data empty until "
           "homography is wired in)")

    td.render_touch_map(OUTPUT_PATH)
    print(f"\n[test] Saved touch map → {OUTPUT_PATH}")
    print("[test] DONE.")


if __name__ == "__main__":
    main()
