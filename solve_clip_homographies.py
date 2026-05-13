"""
Per-clip homography solver — YOLO + Hough + temporal-fill.

Three-pass pipeline:

  Pass 1 — YOLO-only:
      Run PitchKeypointDetector on every frame, fit H via RANSAC DLT,
      sanity-gate. Frames where this succeeds are "anchors".

  Pass 2 — Hough-augmented:
      For each non-anchor frame, find the nearest anchor in time and use
      its H as a SEED. Run hough_homography.fit_H_with_hough(...) which
      matches detected line segments to canonical pitch lines via the seed
      and refits H using the augmented correspondences. Sanity-gate.

  Pass 3 — Temporal fill:
      Any frame still unsolved is interpolated between its two nearest
      sane neighbours by element-wise linear interpolation of H (after
      normalising H[2,2]=1). For huge gaps (> max_gap), the nearest
      neighbour H is copied unchanged.

Outputs:
    out_dir/<seq>_per_frame_H.npz
        frame_ids        (N,)   int
        H_all            (N,3,3) float64 (NaN where not solved)
        sane             (N,)   bool
        source           (N,)   str  ('yolo','hough','interp','copy','fail')
        n_yolo_kp        (N,)   int
        n_hough_segs     (N,)   int
        n_matched_lines  (N,)   int
        reproj_px        (N,)   float

Usage:
    python solve_clip_homographies.py --seq v_gQNyhv8y0QY_c013
    python solve_clip_homographies.py --seq v_gQNyhv8y0QY_c013 --step 5 --no-video
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hough_homography import fit_H_with_hough
from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector
from render_track_video import draw_overlay
from test_per_frame_yolo_h import fit_H_from_keypoints


DATASET_TRAIN = Path("dataset/train")
OUT_DIR       = Path("homographies_v2")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_bboxes(seq_dir: Path) -> Dict[int, List[Tuple[int, int, int, int]]]:
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


def _normalise_H(H: np.ndarray) -> np.ndarray:
    if H is None:
        return H
    if abs(H[2, 2]) < 1e-12:
        return H
    return H / H[2, 2]


def _interp_H(H_a: np.ndarray, H_b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation of normalised Hs (good enough for short gaps)."""
    return (1.0 - t) * H_a + t * H_b


# ──────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_clip(seq: str,
                 step: int = 1,
                 save_video: bool = True,
                 max_interp_gap: int = 60) -> dict:
    seq_dir = DATASET_TRAIN / seq
    if not seq_dir.exists():
        raise SystemExit(f"sequence not found: {seq_dir}")

    frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if step > 1:
        frame_paths = frame_paths[::step]
    if not frame_paths:
        raise SystemExit(f"no frames in {seq_dir}/img1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt_bboxes(seq_dir)
    detector = PitchKeypointDetector(verbose=False)

    n = len(frame_paths)
    records: List[dict] = [None] * n        # type: ignore[list-item]

    print(f"[solve] {seq}  pass 1 (YOLO-only) over {n} frames")
    t0 = time.perf_counter()

    # ── Pass 1: YOLO ────────────────────────────────────────────────────
    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            records[i] = dict(frame_id=int(fp.stem), path=str(fp), H=None,
                              sane=False, source="fail",
                              n_yolo_kp=0, n_hough_segs=0,
                              n_matched_lines=0, reproj_px=float("nan"),
                              fail_reason="cv2.imread failed")
            continue
        kp = detector.detect(frame, conf_threshold=0.30)
        H, info = fit_H_from_keypoints(kp, frame.shape)
        records[i] = dict(
            frame_id        = int(fp.stem),
            path            = str(fp),
            frame_shape     = frame.shape,
            kp_result       = kp,
            H               = H,
            sane            = bool(info["sane"]),
            source          = "yolo" if info["sane"] else "",
            n_yolo_kp       = int(info["n_confident"]),
            n_hough_segs    = 0,
            n_matched_lines = 0,
            reproj_px       = float(info["mean_reproj_px"]),
            fail_reason     = info["fail_reason"],
        )
        if (i + 1) % 50 == 0 or (i + 1) == n:
            dt = time.perf_counter() - t0
            print(f"  [{i+1:>4}/{n}]  {(i+1)/dt:4.1f}fps")

    n_yolo_ok = sum(1 for r in records if r["sane"])
    print(f"  YOLO-only sane: {n_yolo_ok}/{n}  ({100*n_yolo_ok/n:.1f}%)")

    # ── Pass 2: Hough on YOLO-failures, using nearest-anchor seed H ─────
    anchors = [i for i, r in enumerate(records) if r["sane"]]
    if not anchors:
        print("  [warn] zero YOLO anchors; skipping Hough pass")
    else:
        print(f"[solve] pass 2 (Hough on {n - len(anchors)} YOLO-failures)")
        t0 = time.perf_counter()
        n_hough_ok = 0
        for i, r in enumerate(records):
            if r["sane"]:
                continue
            # nearest anchor
            j = min(anchors, key=lambda a: abs(a - i))
            H_seed = records[j]["H"]
            frame = cv2.imread(r["path"])
            if frame is None:
                continue
            kp = r["kp_result"]
            bboxes = gt.get(r["frame_id"], [])
            H_h, info_h = fit_H_with_hough(
                frame_bgr=frame,
                kp_result=kp,
                H_seed=H_seed,
                gt_bboxes=bboxes,
            )
            r["n_hough_segs"]    = info_h.n_hough_segs
            r["n_matched_lines"] = info_h.n_matched_lines
            if info_h.sane and H_h is not None:
                r["H"]          = H_h
                r["sane"]       = True
                r["source"]     = "hough"
                r["reproj_px"]  = info_h.mean_reproj_px
                r["fail_reason"] = ""
                n_hough_ok += 1
            else:
                r["fail_reason"] = info_h.fail_reason or r["fail_reason"]
        dt = time.perf_counter() - t0
        print(f"  Hough recovered: {n_hough_ok}  "
              f"(now sane: {n_yolo_ok + n_hough_ok}/{n} "
              f"= {100*(n_yolo_ok+n_hough_ok)/n:.1f}%)   "
              f"[{dt:.1f}s]")

    # ── Pass 3: temporal fill ───────────────────────────────────────────
    sane_idx = [i for i, r in enumerate(records) if r["sane"]]
    if not sane_idx:
        print("  [warn] no sane frames at all; cannot temporal-fill")
    else:
        n_interp = 0
        n_copy   = 0
        for i, r in enumerate(records):
            if r["sane"]:
                continue
            # Find nearest sane neighbours on each side
            left  = max((j for j in sane_idx if j < i), default=None)
            right = min((j for j in sane_idx if j > i), default=None)
            if left is not None and right is not None:
                gap = right - left
                if gap <= max_interp_gap:
                    t = (i - left) / float(gap)
                    Ha = _normalise_H(records[left]["H"])
                    Hb = _normalise_H(records[right]["H"])
                    r["H"] = _interp_H(Ha, Hb, t)
                    r["sane"] = True
                    r["source"] = "interp"
                    r["fail_reason"] = ""
                    n_interp += 1
                    continue
            # Copy nearest neighbour for huge gaps
            j = left if left is not None else right
            if j is not None:
                r["H"] = _normalise_H(records[j]["H"])
                r["sane"] = True
                r["source"] = "copy"
                r["fail_reason"] = ""
                n_copy += 1
        n_done = sum(1 for r in records if r["sane"])
        print(f"  temporal fill: {n_interp} interp + {n_copy} copy  "
              f"(now sane: {n_done}/{n} = {100*n_done/n:.1f}%)")

    # ── Save NPZ + CSV ──────────────────────────────────────────────────
    frame_ids = np.array([r["frame_id"]    for r in records], dtype=np.int64)
    sane      = np.array([r["sane"]        for r in records], dtype=bool)
    sources   = np.array([r["source"]      for r in records], dtype=object)
    n_yolo    = np.array([r["n_yolo_kp"]   for r in records], dtype=np.int32)
    n_hough_s = np.array([r["n_hough_segs"] for r in records], dtype=np.int32)
    n_matched = np.array([r["n_matched_lines"] for r in records], dtype=np.int32)
    reproj    = np.array([r["reproj_px"]   for r in records], dtype=np.float64)

    H_all = np.full((n, 3, 3), np.nan, dtype=np.float64)
    for i, r in enumerate(records):
        if r["H"] is not None:
            H_all[i] = r["H"]

    npz_path = OUT_DIR / f"{seq}_per_frame_H.npz"
    np.savez_compressed(
        npz_path,
        frame_ids       = frame_ids,
        H_all           = H_all,
        sane            = sane,
        source          = sources.astype("<U8"),
        n_yolo_kp       = n_yolo,
        n_hough_segs    = n_hough_s,
        n_matched_lines = n_matched,
        reproj_px       = reproj,
    )
    print(f"  wrote: {npz_path}")

    csv_path = OUT_DIR / f"{seq}_per_frame_H.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "sane", "source", "n_yolo_kp",
                    "n_hough_segs", "n_matched_lines", "reproj_px",
                    "fail_reason"])
        for r in records:
            w.writerow([
                r["frame_id"], int(r["sane"]), r["source"],
                r["n_yolo_kp"], r["n_hough_segs"], r["n_matched_lines"],
                f"{r['reproj_px']:.3f}" if np.isfinite(r["reproj_px"]) else "",
                r["fail_reason"],
            ])
    print(f"  wrote: {csv_path}")

    # ── Source breakdown summary ────────────────────────────────────────
    src_counts = {"yolo": 0, "hough": 0, "interp": 0, "copy": 0, "fail": 0, "": 0}
    for r in records:
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
    print("=" * 64)
    print(f"  {seq}  source breakdown ({n} frames):")
    for k in ("yolo", "hough", "interp", "copy", "fail", ""):
        c = src_counts.get(k, 0)
        if c:
            label = k or "unsolved"
            print(f"    {label:>10s}: {c:>4d}  ({100*c/n:5.1f}%)")
    print("=" * 64)

    # ── Render overlay video ────────────────────────────────────────────
    if save_video:
        video_path = OUT_DIR / f"{seq}_overlay.mp4"
        render_overlay_video(records, video_path)
        print(f"  wrote: {video_path}")

    return {
        "n":            n,
        "n_yolo_ok":    n_yolo_ok,
        "n_sane_final": int(sane.sum()),
        "src_counts":   src_counts,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

_SRC_COLOUR = {
    "yolo":   (0, 255, 0),       # green
    "hough":  (0, 200, 255),     # amber
    "interp": (0, 120, 255),     # orange
    "copy":   (200, 80, 200),    # purple
    "fail":   (0, 0, 255),       # red
    "":       (0, 0, 255),       # red
}


def render_overlay_video(records: List[dict], out_path: Path,
                          fps: float = 30.0) -> None:
    if not records:
        return
    first = cv2.imread(records[0]["path"])
    if first is None:
        return
    h_img, w_img = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_img, h_img))
    for r in records:
        frame = cv2.imread(r["path"])
        if frame is None:
            continue
        col = _SRC_COLOUR.get(r["source"], (0, 0, 255))
        if r["H"] is not None:
            draw_overlay(frame, r["H"], colour=col)
            tag = (f"f{r['frame_id']:>6}  src={r['source']:<6}  "
                   f"kp={r['n_yolo_kp']:>2}  "
                   f"hough={r['n_hough_segs']:>3}  "
                   f"matched={r['n_matched_lines']:>2}  "
                   f"reproj={r['reproj_px']:5.1f}px"
                   if np.isfinite(r["reproj_px"]) else
                   f"f{r['frame_id']:>6}  src={r['source']:<6}  "
                   f"kp={r['n_yolo_kp']:>2}  "
                   f"hough={r['n_hough_segs']:>3}  "
                   f"matched={r['n_matched_lines']:>2}")
        else:
            tag = (f"f{r['frame_id']:>6}  UNSOLVED   {r['fail_reason']}")
        cv2.rectangle(frame, (0, 0), (w_img, 32), (0, 0, 0), -1)
        cv2.putText(frame, tag, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--max-interp-gap", type=int, default=60)
    args = ap.parse_args()

    process_clip(
        args.seq,
        step           = args.step,
        save_video     = not args.no_video,
        max_interp_gap = args.max_interp_gap,
    )


if __name__ == "__main__":
    main()
