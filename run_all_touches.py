"""
Run the touch-detection pipeline on every SportsMOT clip in dataset/train.

Each clip is processed in a single pass:
    BallDetector (per-frame YOLO) -> MultiPlayerTouchAssigner -> CSVs

Outputs (per clip):
    touches/<seq>_events.csv
    touches/<seq>_summary.csv

Aggregate output:
    touches/_combined_summary.csv     (all clips, one row per (clip, pid))
    touches/_combined_summary.txt     (printable table for all clips)

Usage:
    python run_all_touches.py
    python run_all_touches.py --proximity-factor 1.2 --min-hold 2
    python run_all_touches.py --save-videos        # save annotated mp4 per clip

Wall time: ~10 minutes per clip (BallDetector is the bottleneck).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List

import cv2

from ball_detect import BallDetector
from homography import PitchHomography
from touch_assigner import MultiPlayerTouchAssigner
from touch_pipeline import (DATASET_TRAIN_DIR, HOMOGRAPHY_DIR, TOUCHES_OUT_DIR,
                            draw_hud, draw_overlay, load_gt_with_ids,
                            save_events_csv, save_summary_csv, read_seqinfo)


def list_clips(train_dir: Path) -> List[Path]:
    return sorted(p for p in train_dir.iterdir()
                  if p.is_dir() and (p / "img1").exists()
                  and (p / "gt" / "gt.txt").exists())


def run_clip(seq_dir: Path, args) -> dict:
    """Run the pipeline on one clip. Returns a small status dict."""
    seq = seq_dir.name
    t0 = time.perf_counter()

    info = read_seqinfo(seq_dir)
    fps_native = float(info.get("frameRate", 25))

    gt = load_gt_with_ids(seq_dir)
    if not gt:
        return {"seq": seq, "skipped": "no gt.txt"}

    homo = None
    homo_path = HOMOGRAPHY_DIR / f"{seq}.npz"
    if homo_path.exists():
        try:
            homo = PitchHomography.load(homo_path)
        except Exception:
            homo = None

    frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frame_paths:
        return {"seq": seq, "skipped": "no frames"}

    detector = BallDetector(conf_threshold=args.ball_conf, verbose=False)
    assigner = MultiPlayerTouchAssigner(
        proximity_factor = args.proximity_factor,
        min_hold_frames  = args.min_hold,
        cooldown_frames  = args.cooldown,
    )

    writer = None
    if args.save_videos:
        sample = cv2.imread(str(frame_paths[0]))
        h_img, w_img = sample.shape[:2]
        out_path = TOUCHES_OUT_DIR / f"{seq}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_native,
                                 (w_img, h_img))

    n_real = n_est = n_miss = 0
    last_log = t0
    n = len(frame_paths)

    for i, fp in enumerate(frame_paths):
        fid = int(fp.stem)
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        players = gt.get(fid, [])
        bboxes_for_fp = [(pb.x, pb.y, pb.w, pb.h) for pb in players] or None
        result = detector.track_ball(frame, player_bboxes=bboxes_for_fp)

        if result["ball_px"] is None:
            n_miss += 1
            ball_xy = None
        else:
            ball_xy = (float(result["ball_px"]), float(result["ball_py"]))
            if result["is_estimated"]:
                n_est += 1
            else:
                n_real += 1

        emitted = assigner.update(fid, ball_xy, players, homography=homo)

        if writer is not None:
            cur_fps = (i + 1) / max(time.perf_counter() - t0, 1e-3)
            draw_overlay(frame, fid, players, ball_xy, assigner,
                         just_touched=(emitted is not None))
            draw_hud(frame, seq, fid, n,
                     n_touches        = len(assigner.events),
                     n_players_active = len(players),
                     fps              = cur_fps,
                     proximity_factor = args.proximity_factor,
                     min_hold         = args.min_hold,
                     cooldown         = args.cooldown)
            writer.write(frame)

        # Progress log every 10s
        now = time.perf_counter()
        if now - last_log > 10.0:
            pct = 100.0 * (i + 1) / n
            cur_fps = (i + 1) / max(now - t0, 1e-3)
            print(f"    [{seq}]  {i + 1:>4}/{n}  ({pct:5.1f}%)  "
                  f"{cur_fps:4.1f}fps  touches={len(assigner.events)}")
            last_log = now

    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - t0
    rows = assigner.summary_table()
    save_events_csv(assigner.events,
                    TOUCHES_OUT_DIR / f"{seq}_events.csv")
    save_summary_csv(rows,
                     TOUCHES_OUT_DIR / f"{seq}_summary.csv")

    print(f"  [{seq}] DONE in {elapsed/60:.1f}min  "
          f"frames={n}  touches={len(assigner.events)}  "
          f"players_with_touches={len(rows)}  "
          f"ball_recall={100*(n_real+n_est)/max(n,1):.1f}%")

    return {
        "seq":          seq,
        "frames":       n,
        "touches":      len(assigner.events),
        "players":      len(rows),
        "real":         n_real,
        "est":          n_est,
        "miss":         n_miss,
        "elapsed_s":    elapsed,
        "rows":         rows,
        "events":       assigner.events,
    }


def write_combined(results: List[dict]) -> None:
    out_csv = TOUCHES_OUT_DIR / "_combined_summary.csv"
    out_txt = TOUCHES_OUT_DIR / "_combined_summary.txt"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "player_id", "touch_count",
                    "first_touch_frame", "last_touch_frame",
                    "zones", "touches_outside_H_support"])
        for r in results:
            if "rows" not in r:
                continue
            for row in r["rows"]:
                w.writerow([r["seq"], row["player_id"], row["touch_count"],
                            row["first_touch_frame"], row["last_touch_frame"],
                            row["zones"], row["touches_outside_H_support"]])

    lines = []
    lines.append("=" * 78)
    lines.append("  COMBINED TOUCH SUMMARY")
    lines.append("=" * 78)
    lines.append(f"  {'seq':<28}  {'frames':>6}  {'touches':>7}  "
                 f"{'players':>7}  {'recall':>6}  {'time':>7}")
    lines.append("  " + "-" * 76)
    for r in results:
        if "rows" not in r:
            lines.append(f"  {r['seq']:<28}  SKIPPED ({r.get('skipped','?')})")
            continue
        recall = 100.0 * (r["real"] + r["est"]) / max(r["frames"], 1)
        lines.append(f"  {r['seq']:<28}  {r['frames']:>6}  "
                     f"{r['touches']:>7}  {r['players']:>7}  "
                     f"{recall:>5.1f}%  {r['elapsed_s']/60:>5.1f}m")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Per-clip top players (by touch count):")
    for r in results:
        if "rows" not in r:
            continue
        lines.append("")
        lines.append(f"  {r['seq']}")
        lines.append(f"    {'pid':>4}  {'touches':>7}  {'first':>6}  "
                     f"{'last':>6}  zones")
        for row in r["rows"][:10]:
            lines.append(f"    {row['player_id']:>4}  "
                         f"{row['touch_count']:>7}  "
                         f"{row['first_touch_frame']:>6}  "
                         f"{row['last_touch_frame']:>6}  "
                         f"{row['zones']}")
    text = "\n".join(lines)
    out_txt.write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"\n  wrote: {out_csv}")
    print(f"  wrote: {out_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proximity-factor", type=float, default=1.0)
    ap.add_argument("--min-hold",         type=int,   default=3)
    ap.add_argument("--cooldown",         type=int,   default=8)
    ap.add_argument("--ball-conf",        type=float, default=0.15)
    ap.add_argument("--save-videos",      action="store_true",
                    help="Save touches/<seq>.mp4 per clip (slower, larger).")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of seq names to run.")
    args = ap.parse_args()

    clips = list_clips(DATASET_TRAIN_DIR)
    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        clips = [c for c in clips if c.name in wanted]

    print(f"[batch] processing {len(clips)} clip(s):")
    for c in clips:
        print(f"  - {c.name}")
    print(f"[batch] params: prox={args.proximity_factor}  "
          f"min_hold={args.min_hold}  cooldown={args.cooldown}  "
          f"save_videos={args.save_videos}")

    t_all = time.perf_counter()
    results = []
    for c in clips:
        print(f"\n[batch] === {c.name} ===")
        try:
            results.append(run_clip(c, args))
        except Exception as e:
            print(f"  [{c.name}] ERROR: {e}")
            results.append({"seq": c.name, "skipped": str(e)})

    print(f"\n[batch] all done in {(time.perf_counter() - t_all)/60:.1f} min")
    write_combined(results)


if __name__ == "__main__":
    main()
