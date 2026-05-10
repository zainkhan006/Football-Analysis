"""
Ball-detection benchmark — quantifies the speedup from the refactor.

Compares three configurations on the same SportsMOT subset:

    A. Baseline           – COCO YOLOv8n, full frame, imgsz=640, no FP filter
                            (≈ what the old ball_detect.py defaulted to)
    B. Heavy + full-frame – Roboflow model, full frame every time, imgsz=640
                            (the old behaviour with the heavy model)
    C. Optimised          – Roboflow model + ROI tracking + FP filter
                            (the new default)

Reports for each:
  - Avg ms/frame, FPS
  - Real detections, Kalman estimates, true misses
  - Effective recall  = (real + Kalman) / N

Run:
    python benchmark_ball.py
    python benchmark_ball.py --frames 200    # quick smoke test
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2

from ball_detect import BallDetector


# ─── Config ───────────────────────────────────────────────────────────────────

SEQUENCE_DIR = Path("dataset/train/v_gQNyhv8y0QY_c013")


def load_gt(sequence_dir: Path) -> dict[int, list[tuple]]:
    """Returns {frame_id: [(x,y,w,h), ...]}."""
    gt_file = sequence_dir / "gt" / "gt.txt"
    out: dict = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            x, y, w, h = (int(parts[2]), int(parts[3]),
                          int(parts[4]), int(parts[5]))
            out[fid].append((x, y, w, h))
    return dict(out)


def run_config(
    label: str,
    detector: BallDetector,
    frames: list[Path],
    gt: dict,
    use_player_filter: bool,
):
    real = est = miss = 0
    inf_s = 0.0
    for p in frames:
        fid = int(p.stem)
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        bboxes = gt.get(fid) if use_player_filter else None
        t0 = time.perf_counter()
        r  = detector.track_ball(frame, player_bboxes=bboxes)
        inf_s += time.perf_counter() - t0

        if r["ball_px"] is None:
            miss += 1
        elif r["is_estimated"]:
            est += 1
        else:
            real += 1

    n = len(frames)
    avg_ms = 1000.0 * inf_s / max(n, 1)
    fps    = n / max(inf_s, 1e-9)
    eff_recall = (real + est) / max(n, 1) * 100.0
    real_recall = real / max(n, 1) * 100.0

    print(f"\n--- {label} ---")
    print(f"  Frames           : {n}")
    print(f"  Inference total  : {inf_s:6.2f} s")
    print(f"  Per-frame        : {avg_ms:6.1f} ms  ({fps:5.1f} FPS)")
    print(f"  Real detections  : {real:4d}  ({real_recall:5.1f}%)")
    print(f"  Kalman estimates : {est:4d}")
    print(f"  Misses           : {miss:4d}")
    print(f"  Effective recall : {eff_recall:5.1f}%")
    return {
        "label": label, "n": n, "ms_per_frame": avg_ms, "fps": fps,
        "real": real, "est": est, "miss": miss,
        "real_recall": real_recall, "eff_recall": eff_recall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300,
                    help="number of frames to process per config "
                         "(default 300; full clip = 875)")
    ap.add_argument("--stride", type=int, default=2,
                    help="frame stride (default 2)")
    args = ap.parse_args()

    img_dir = SEQUENCE_DIR / "img1"
    all_imgs = sorted(img_dir.glob("*.jpg"))
    if not all_imgs:
        print(f"[error] No frames in {img_dir}")
        return
    frames = all_imgs[:args.frames][::args.stride]
    print(f"[bench] Sequence  : {SEQUENCE_DIR}")
    print(f"[bench] Frames    : {len(frames)} "
          f"(first {args.frames}, stride {args.stride})")

    gt = load_gt(SEQUENCE_DIR)

    summary = []

    # A. Baseline — COCO YOLOv8n, no FP filter, no ROI (full_frame_every=1)
    print("\n[bench] A. Baseline COCO YOLOv8n (sports ball class)")
    det_a = BallDetector(
        model_path       = "yolov8n.pt",
        conf_threshold   = 0.25,
        imgsz_full       = 640,
        imgsz_roi        = 640,
        roi_size         = 9999,        # effectively full frame
        full_frame_every = 1,           # always full frame
        verbose          = True,
    )
    summary.append(run_config(
        "A. COCO YOLOv8n, full frame, no FP filter",
        det_a, frames, gt, use_player_filter=False))
    del det_a

    # B. Heavy model, full-frame each call — pre-refactor behaviour
    print("\n[bench] B. Roboflow model, full frame every call (no ROI)")
    det_b = BallDetector(
        conf_threshold   = 0.15,
        imgsz_full       = 640,
        imgsz_roi        = 640,
        roi_size         = 9999,
        full_frame_every = 1,
        verbose          = True,
    )
    summary.append(run_config(
        "B. Roboflow heavy, full frame, no FP filter",
        det_b, frames, gt, use_player_filter=False))
    del det_b

    # C. Optimised — Roboflow + ROI tracking + FP filter
    print("\n[bench] C. Roboflow model + ROI tracking + FP filter (new default)")
    det_c = BallDetector(
        conf_threshold   = 0.15,
        imgsz_full       = 640,
        imgsz_roi        = 320,
        roi_size         = 320,
        full_frame_every = 30,
        verbose          = True,
    )
    summary.append(run_config(
        "C. Roboflow + ROI + FP filter (NEW)",
        det_c, frames, gt, use_player_filter=True))
    del det_c

    # ── Final table ────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'Config':<48} {'ms/frame':>10} {'FPS':>7} "
          f"{'real':>6} {'eff':>7}")
    for s in summary:
        print(f"{s['label']:<48} {s['ms_per_frame']:>10.1f} "
              f"{s['fps']:>7.1f} {s['real_recall']:>5.1f}% "
              f"{s['eff_recall']:>6.1f}%")

    if len(summary) >= 3:
        speedup = summary[1]["ms_per_frame"] / summary[2]["ms_per_frame"]
        print(f"\n[bench] Speedup (B → C) on heavy model: {speedup:.2f}x")


if __name__ == "__main__":
    main()
