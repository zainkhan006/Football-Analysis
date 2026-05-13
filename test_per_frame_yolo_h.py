"""
Independent per-frame H test: YOLO keypoints -> RANSAC DLT -> sanity check.

No temporal propagation, no DT refinement, no Hough. Each frame stands alone.

For the given clip, we:
    1. Run PitchKeypointDetector on every frame.
    2. Take keypoints with confidence >= CONF_THRESH.
    3. Fit H via cv2.findHomography(..., RANSAC) from world<->pixel pairs.
    4. Sanity-check the H (reprojection error, projected-pitch area / convexity).
    5. Record success / failure and reprojection error for each frame.

Outputs:
    debug_per_frame_yolo/<seq>.mp4          -- overlay video across whole clip
    debug_per_frame_yolo/<seq>_samples.jpg  -- 3x3 grid of sample frames
    debug_per_frame_yolo/<seq>_stats.csv    -- per-frame stats
    stdout: summary (success rate, mean/median reprojection error)

Usage:
    python test_per_frame_yolo_h.py --seq v_gQNyhv8y0QY_c013
    python test_per_frame_yolo_h.py --seq v_gQNyhv8y0QY_c013 --step 5      # every 5th frame
    python test_per_frame_yolo_h.py --seq v_gQNyhv8y0QY_c013 --no-video    # skip mp4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector
from render_track_video import draw_overlay


DATASET_TRAIN = Path("dataset/train")
OUT_DIR       = Path("debug_per_frame_yolo")

CONF_THRESH        = 0.5
MIN_KP_FOR_FIT     = 4
RANSAC_THRESH_PX   = 8.0


# ──────────────────────────────────────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────────────────────────────────────

def _is_H_sane(H: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
    """
    Reject degenerate / clearly-wrong Hs. Uses the canonical 120x70 pitch.
    The H must:
      - be finite, non-singular
      - project the 4 pitch corners to a convex quadrilateral
      - produce a projected pitch that covers a reasonable area of the frame
      - have corners not absurdly far off-screen
    """
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-9:
        return False

    L, W = DEFAULT_PITCH.length, DEFAULT_PITCH.width
    corners = np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=np.float64)
    homog = np.hstack([corners, np.ones((4, 1))]).T
    proj = H @ homog
    w = proj[2, :]
    if np.any(np.abs(w) < 1e-9):
        return False
    proj_xy = (proj[:2, :] / w).T                             # (4, 2)

    if not np.all(np.isfinite(proj_xy)):
        return False

    # Convexity: signed area via shoelace; all cross products should share sign
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    signs = [np.sign(cross(proj_xy[i], proj_xy[(i+1)%4], proj_xy[(i+2)%4]))
             for i in range(4)]
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return False

    # Area check
    x, y = proj_xy[:, 0], proj_xy[:, 1]
    area = 0.5 * abs(x[0]*(y[1]-y[3]) + x[1]*(y[2]-y[0])
                   + x[2]*(y[3]-y[1]) + x[3]*(y[0]-y[2]))
    h_img, w_img = frame_shape[:2]
    if area < 0.02 * h_img * w_img:
        return False
    if area > 50.0 * h_img * w_img:
        return False

    # Corners not absurdly far offscreen
    if (np.abs(proj_xy[:, 0]) > 10 * w_img).any() \
            or (np.abs(proj_xy[:, 1]) > 10 * h_img).any():
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
#  Per-frame fitter
# ──────────────────────────────────────────────────────────────────────────────

def fit_H_from_keypoints(
    kp_result,
    frame_shape: Tuple[int, int],
    conf_thresh: float = CONF_THRESH,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Fit H from confident YOLO keypoints using RANSAC DLT.
    Returns (H or None, info_dict).
    """
    info = {
        "n_confident":      0,
        "n_ransac_inliers": 0,
        "mean_reproj_px":   float("nan"),
        "sane":             False,
        "fail_reason":      "",
    }

    vertices = DEFAULT_PITCH.vertices_m
    world_pts = []
    pixel_pts = []
    for kp in kp_result.keypoints:
        if kp.confidence < conf_thresh:
            continue
        x, y = kp.pixel_xy
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        wx, wy = vertices[kp.index]
        world_pts.append((wx, wy))
        pixel_pts.append((x, y))

    info["n_confident"] = len(world_pts)

    if len(world_pts) < MIN_KP_FOR_FIT:
        info["fail_reason"] = f"only {len(world_pts)} confident keypoints"
        return None, info

    world_np = np.array(world_pts, dtype=np.float64)
    pixel_np = np.array(pixel_pts, dtype=np.float64)

    # Degeneracy guard: if keypoints are ~collinear in pixel space, RANSAC
    # will still "succeed" but H is ill-conditioned. Check bounding box.
    px_bbox = (pixel_np.max(axis=0) - pixel_np.min(axis=0))
    if px_bbox[0] < 30 or px_bbox[1] < 30:
        info["fail_reason"] = f"keypoint pixel bbox too small: {px_bbox}"
        return None, info

    H, mask = cv2.findHomography(
        srcPoints    = world_np,
        dstPoints    = pixel_np,
        method       = cv2.RANSAC,
        ransacReprojThreshold = RANSAC_THRESH_PX,
        maxIters     = 2000,
        confidence   = 0.999,
    )
    if H is None:
        info["fail_reason"] = "findHomography returned None"
        return None, info

    mask = mask.ravel().astype(bool)
    info["n_ransac_inliers"] = int(mask.sum())

    # Reprojection error on the inliers
    homog = np.hstack([world_np[mask], np.ones((mask.sum(), 1))]).T
    proj = H @ homog
    proj_xy = (proj[:2, :] / proj[2, :]).T
    diffs = proj_xy - pixel_np[mask]
    info["mean_reproj_px"] = float(np.hypot(diffs[:, 0], diffs[:, 1]).mean())

    if info["n_ransac_inliers"] < MIN_KP_FOR_FIT:
        info["fail_reason"] = f"only {info['n_ransac_inliers']} RANSAC inliers"
        return None, info

    if not _is_H_sane(H, frame_shape):
        info["fail_reason"] = "H failed sanity check"
        return None, info

    info["sane"] = True
    return H.astype(np.float64), info


# ──────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def process_clip(seq: str,
                 step: int = 1,
                 save_video: bool = True,
                 save_samples: bool = True) -> None:
    seq_dir = DATASET_TRAIN / seq
    if not seq_dir.exists():
        raise SystemExit(f"sequence not found: {seq_dir}")

    frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"no frames in {seq_dir}/img1")
    if step > 1:
        frame_paths = frame_paths[::step]

    print(f"[per-frame-yolo] {seq}  processing {len(frame_paths)} frames "
          f"(step={step})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = PitchKeypointDetector(verbose=False)

    # Pass 1: fit H on every frame
    records = []
    t0 = time.perf_counter()
    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        kp = detector.detect(frame, conf_threshold=0.30)
        H, info = fit_H_from_keypoints(kp, frame.shape)
        records.append({
            "frame_id":         int(fp.stem),
            "path":             str(fp),
            "H":                H,
            **info,
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(frame_paths):
            dt = time.perf_counter() - t0
            print(f"  [{i+1:>4}/{len(frame_paths)}]  "
                  f"{(i+1)/dt:4.1f}fps  "
                  f"last_frame_ok={info['sane']}")

    # ── Per-frame stats
    n      = len(records)
    n_ok   = sum(1 for r in records if r["sane"])
    reprojs = [r["mean_reproj_px"] for r in records if r["sane"]]
    kp_counts = [r["n_confident"] for r in records]

    print()
    print("=" * 68)
    print(f"  {seq}")
    print("=" * 68)
    print(f"  frames processed   : {n}")
    print(f"  frames with sane H : {n_ok}  ({100*n_ok/max(n,1):.1f}%)")
    if reprojs:
        print(f"  reproj error (px)  : "
              f"mean={np.mean(reprojs):.2f}  "
              f"median={np.median(reprojs):.2f}  "
              f"max={np.max(reprojs):.2f}")
    print(f"  confident keypoints: "
          f"mean={np.mean(kp_counts):.1f}  "
          f"median={np.median(kp_counts):.0f}  "
          f"max={np.max(kp_counts)}  "
          f"min={np.min(kp_counts)}")
    fail_reasons: dict = {}
    for r in records:
        if not r["sane"]:
            fail_reasons[r["fail_reason"]] = fail_reasons.get(r["fail_reason"], 0) + 1
    if fail_reasons:
        print(f"  failure reasons:")
        for reason, count in sorted(fail_reasons.items(),
                                    key=lambda x: -x[1]):
            print(f"    {count:>4}  {reason}")
    print("=" * 68)

    # ── Save CSV
    csv_path = OUT_DIR / f"{seq}_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "n_confident", "n_ransac_inliers",
                    "mean_reproj_px", "sane", "fail_reason"])
        for r in records:
            w.writerow([r["frame_id"], r["n_confident"],
                        r["n_ransac_inliers"],
                        f"{r['mean_reproj_px']:.3f}" if np.isfinite(r['mean_reproj_px']) else "",
                        int(r["sane"]), r["fail_reason"]])
    print(f"  wrote: {csv_path}")

    # ── Save sample panel (3x3 grid evenly spaced through clip)
    if save_samples:
        sample_path = OUT_DIR / f"{seq}_samples.jpg"
        render_sample_panel(records, sample_path)
        print(f"  wrote: {sample_path}")

    # ── Save video
    if save_video:
        video_path = OUT_DIR / f"{seq}.mp4"
        render_video(records, video_path)
        print(f"  wrote: {video_path}")


def render_sample_panel(records: list[dict], out_path: Path,
                         grid_rows: int = 3, grid_cols: int = 3) -> None:
    """3x3 grid of evenly-spaced frames with H overlay."""
    n_total = len(records)
    n = grid_rows * grid_cols
    if n_total == 0:
        return
    idxs = np.linspace(0, n_total - 1, n).astype(int)
    panels = []
    for i, idx in enumerate(idxs):
        r = records[idx]
        img = cv2.imread(r["path"])
        if img is None:
            panels.append(np.zeros((200, 320, 3), dtype=np.uint8))
            continue
        # shrink for the grid
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        if r["H"] is not None:
            H_scaled = np.diag([0.5, 0.5, 1.0]) @ r["H"]
            draw_overlay(img, H_scaled, colour=(0, 255, 0))
            status = f"OK reproj={r['mean_reproj_px']:.1f}px"
            colour = (0, 255, 0)
        else:
            status = f"FAIL {r['fail_reason']}"
            colour = (0, 0, 255)
        cv2.putText(img, f"f{r['frame_id']:>6}  kp={r['n_confident']}  {status}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        panels.append(img)

    # Resize to common shape (first panel)
    H, W = panels[0].shape[:2]
    for i in range(len(panels)):
        if panels[i].shape[:2] != (H, W):
            panels[i] = cv2.resize(panels[i], (W, H))
    rows = [np.hstack(panels[r*grid_cols:(r+1)*grid_cols])
            for r in range(grid_rows)]
    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)


def render_video(records: list[dict], out_path: Path, fps: float = 30.0) -> None:
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
        if r["H"] is not None:
            draw_overlay(frame, r["H"], colour=(0, 255, 0))
            tag = (f"frame {r['frame_id']:>6}  kp={r['n_confident']:>2}  "
                   f"inl={r['n_ransac_inliers']:>2}  "
                   f"reproj={r['mean_reproj_px']:.1f}px")
            colour = (0, 255, 0)
        else:
            tag = (f"frame {r['frame_id']:>6}  kp={r['n_confident']:>2}  "
                   f"FAIL: {r['fail_reason']}")
            colour = (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w_img, 32), (0, 0, 0), -1)
        cv2.putText(frame, tag, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-samples", action="store_true")
    args = ap.parse_args()

    process_clip(args.seq,
                 step         = args.step,
                 save_video   = not args.no_video,
                 save_samples = not args.no_samples)


if __name__ == "__main__":
    main()
