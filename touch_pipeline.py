"""
End-to-end touch detection for a SportsMOT clip.

Pipeline
--------
For each frame:
    1. Read the image.
    2. Look up the player tracks for this frame from gt.txt
       (track_id + bbox).
    3. Run the BallDetector to get a ball pixel position
       (real detection or Kalman estimate).
    4. Pass to MultiPlayerTouchAssigner, which decides whether
       a touch event should be emitted on this frame.
After the loop:
    - Print a per-player summary table sorted by touch count.
    - Save per-touch CSV and per-player summary CSV.
    - Optionally save an annotated video (--save-video).

Usage
-----
    python touch_pipeline.py --seq v_gQNyhv8y0QY_c013
    python touch_pipeline.py --seq v_gQNyhv8y0QY_c013 --save-video out.mp4
    python touch_pipeline.py --seq v_gQNyhv8y0QY_c013 \
        --proximity-factor 1.2 --min-hold 3 --cooldown 8
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ball_detect import BallDetector
from homography import PitchHomography
from touch_assigner import MultiPlayerTouchAssigner, PlayerBox, TouchEvent


DATASET_TRAIN_DIR = Path("dataset/train")
HOMOGRAPHY_DIR    = Path("homographies")
TOUCHES_OUT_DIR   = Path("touches")


# ──────────────────────────────────────────────────────────────────────────────
#  Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_with_ids(seq_dir: Path) -> dict[int, list[PlayerBox]]:
    """
    Parse SportsMOT gt.txt. Returns {frame_id: [PlayerBox, ...]}.
    MOT format: frame, track_id, x, y, w, h, conf, class, vis
    """
    gt_file = seq_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        return {}
    out: dict[int, list[PlayerBox]] = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 6:
                continue
            try:
                fid = int(parts[0])
                pid = int(parts[1])
                x   = int(float(parts[2]))
                y   = int(float(parts[3]))
                w   = int(float(parts[4]))
                h   = int(float(parts[5]))
            except ValueError:
                continue
            out[fid].append(PlayerBox(pid=pid, x=x, y=y, w=w, h=h))
    return dict(out)


def read_seqinfo(seq_dir: Path) -> dict:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        return {}
    cp = ConfigParser()
    cp.read(ini)
    return dict(cp["Sequence"]) if "Sequence" in cp else {}


# ──────────────────────────────────────────────────────────────────────────────
#  Drawing
# ──────────────────────────────────────────────────────────────────────────────

POSSESSOR_COLOUR = (255, 80, 200)   # magenta
TOUCH_COLOUR     = (0, 255, 255)    # yellow
GHOST_COLOUR     = (90, 90, 90)


def draw_overlay(
    frame: np.ndarray,
    fid: int,
    players: list[PlayerBox],
    ball_xy: Optional[tuple],
    assigner: MultiPlayerTouchAssigner,
    just_touched: bool,
) -> None:
    # Faint boxes for everyone
    for pb in players:
        cv2.rectangle(frame, (pb.x, pb.y), (pb.x + pb.w, pb.y + pb.h),
                      GHOST_COLOUR, 1)
        cv2.putText(frame, f"#{pb.pid}", (pb.x, max(pb.y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GHOST_COLOUR, 1)

    # Highlight the confirmed possessor (if any) for this frame
    poss_pid = assigner.possessor_per_frame.get(fid)
    if poss_pid is not None:
        for pb in players:
            if pb.pid == poss_pid:
                cv2.rectangle(frame, (pb.x, pb.y), (pb.x + pb.w, pb.y + pb.h),
                              POSSESSOR_COLOUR, 2)
                tag = f"poss #{pb.pid}  touches={assigner.touches_per_player.get(pb.pid, 0)}"
                if just_touched:
                    tag += "  TOUCH!"
                cv2.putText(frame, tag, (pb.x, max(pb.y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            POSSESSOR_COLOUR, 1)
                break

    # Ball
    if ball_xy is not None:
        cv2.circle(frame, (int(ball_xy[0]), int(ball_xy[1])),
                   8, (0, 255, 255), 2)

    # Just-touched flash
    if just_touched and ball_xy is not None:
        cv2.circle(frame, (int(ball_xy[0]), int(ball_xy[1])),
                   24, TOUCH_COLOUR, 3)


def draw_hud(
    frame: np.ndarray,
    seq_name: str,
    fid: int,
    total: int,
    n_touches: int,
    n_players_active: int,
    fps: float,
    proximity_factor: float,
    min_hold: int,
    cooldown: int,
) -> None:
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"Seq: {seq_name}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame {fid}/{total}   {fps:5.1f} FPS   "
                f"touches={n_touches}   tracked-players={n_players_active}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(frame, f"prox-factor={proximity_factor}  "
                f"min-hold={min_hold}f  cooldown={cooldown}f",
                (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1)


# ──────────────────────────────────────────────────────────────────────────────
#  CSV writers
# ──────────────────────────────────────────────────────────────────────────────

def save_events_csv(events: list[TouchEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_id", "player_id",
            "ball_px", "ball_py",
            "world_x_m", "world_y_m",
            "zone", "h_supported",
            "foot_dist_px", "bbox_h_px",
        ])
        for ev in events:
            wx, wy = ev.world_xy if ev.world_xy is not None else ("", "")
            w.writerow([
                ev.frame_id, ev.player_id,
                f"{ev.ball_px:.1f}", f"{ev.ball_py:.1f}",
                f"{wx:.2f}" if isinstance(wx, float) else "",
                f"{wy:.2f}" if isinstance(wy, float) else "",
                ev.zone or "",
                int(ev.h_supported),
                f"{ev.foot_dist_px:.1f}",
                ev.bbox_h_px,
            ])


def save_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "player_id", "touch_count",
            "first_touch_frame", "last_touch_frame",
            "zones", "touches_outside_H_support",
        ])
        for r in rows:
            w.writerow([
                r["player_id"], r["touch_count"],
                r["first_touch_frame"], r["last_touch_frame"],
                r["zones"], r["touches_outside_H_support"],
            ])


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True,
                    help="Sequence folder name under dataset/train/")
    ap.add_argument("--proximity-factor", type=float, default=1.0,
                    help="Ball must be within (proximity_factor * bbox_height) "
                         "of a player's foot point to count as possession.")
    ap.add_argument("--min-hold", type=int, default=3,
                    help="Frames a candidate possessor must persist before "
                         "they're confirmed (debouncing).")
    ap.add_argument("--cooldown", type=int, default=8,
                    help="Per-player cooldown in frames (prevents double-count).")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=None)
    ap.add_argument("--save-video", default=None,
                    help="Optional path to save annotated mp4.")
    ap.add_argument("--no-display", action="store_true",
                    help="Don't open a live window (useful for batch runs).")
    ap.add_argument("--ball-conf", type=float, default=0.15)
    args = ap.parse_args()

    seq_dir = DATASET_TRAIN_DIR / args.seq
    if not seq_dir.exists():
        raise SystemExit(f"sequence not found: {seq_dir}")

    info = read_seqinfo(seq_dir)
    fps_native = float(info.get("frameRate", 25))

    # ── Load tracks ────────────────────────────────────────────────────
    gt = load_gt_with_ids(seq_dir)
    if not gt:
        raise SystemExit(f"no gt.txt found under {seq_dir}/gt/")
    n_frames_with_tracks = len(gt)
    n_unique_pids = len({pb.pid for plist in gt.values() for pb in plist})
    print(f"[touch] gt.txt: {n_frames_with_tracks} frames, "
          f"{n_unique_pids} unique track IDs")

    # ── Homography (optional but useful for zones) ─────────────────────
    homo: Optional[PitchHomography] = None
    homo_path = HOMOGRAPHY_DIR / f"{args.seq}.npz"
    if homo_path.exists():
        try:
            homo = PitchHomography.load(homo_path)
            print(f"[touch] homography: {homo_path}")
        except Exception as e:
            print(f"[touch] homography load failed: {e}")
    else:
        print(f"[touch] no homography at {homo_path} - "
              f"world coords / zones will be empty")

    # ── Frames ─────────────────────────────────────────────────────────
    frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"no frames in {seq_dir}/img1")

    start = max(1, args.start)
    end = args.end if args.end else len(frame_paths)
    end = min(end, len(frame_paths))
    frame_paths = [p for p in frame_paths if start <= int(p.stem) <= end]
    print(f"[touch] processing frames {start}..{end} ({len(frame_paths)} frames)")

    # ── Detector + assigner ────────────────────────────────────────────
    detector = BallDetector(conf_threshold=args.ball_conf, verbose=False)
    assigner = MultiPlayerTouchAssigner(
        proximity_factor = args.proximity_factor,
        min_hold_frames  = args.min_hold,
        cooldown_frames  = args.cooldown,
    )

    # ── Optional video writer ──────────────────────────────────────────
    writer = None
    if args.save_video:
        sample = cv2.imread(str(frame_paths[0]))
        h_img, w_img = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_native,
                                 (w_img, h_img))
        print(f"[touch] saving annotated video -> {args.save_video}")

    if not args.no_display:
        cv2.namedWindow("touch_pipeline", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("touch_pipeline", 1280, 720)

    # ── Loop ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    n_real = n_est = n_miss = 0

    for i, fp in enumerate(frame_paths):
        fid = int(fp.stem)
        frame = cv2.imread(str(fp))
        if frame is None:
            continue

        players = gt.get(fid, [])
        # Ball detection (use GT bboxes as FP filter, same as play_ball_detection.py)
        gt_bboxes_for_detector = [(pb.x, pb.y, pb.w, pb.h) for pb in players]
        result = detector.track_ball(frame,
                                     player_bboxes=gt_bboxes_for_detector or None)

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

        if not args.no_display or writer is not None:
            elapsed = time.perf_counter() - t0
            cur_fps = (i + 1) / max(elapsed, 1e-3)
            draw_overlay(frame, fid, players, ball_xy, assigner,
                         just_touched=(emitted is not None))
            draw_hud(frame, args.seq, fid, end,
                     n_touches        = len(assigner.events),
                     n_players_active = len(players),
                     fps              = cur_fps,
                     proximity_factor = args.proximity_factor,
                     min_hold         = args.min_hold,
                     cooldown         = args.cooldown)

            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("touch_pipeline", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    elapsed = time.perf_counter() - t0
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Print summary ──────────────────────────────────────────────────
    rows = assigner.summary_table()
    print("\n" + "=" * 72)
    print(f"  TOUCH SUMMARY  -  {args.seq}")
    print("=" * 72)
    print(f"  frames processed   : {len(frame_paths)}")
    print(f"  ball  real / est / miss : "
          f"{n_real} / {n_est} / {n_miss}  "
          f"({100*(n_real+n_est)/max(len(frame_paths),1):.1f}% recall)")
    print(f"  total touches      : {len(assigner.events)}")
    print(f"  unique players w/ touches : {len(rows)}")
    print(f"  wall time          : {elapsed:.1f}s "
          f"({len(frame_paths)/max(elapsed,1e-3):.1f}fps)")
    print()
    print(f"  {'pid':>4}  {'touches':>7}  {'first':>6}  {'last':>6}  "
          f"{'~outside_H':>10}  zones")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['player_id']:>4}  "
              f"{r['touch_count']:>7}  "
              f"{r['first_touch_frame']:>6}  "
              f"{r['last_touch_frame']:>6}  "
              f"{r['touches_outside_H_support']:>10}  "
              f"{r['zones']}")
    print("=" * 72)

    # ── Save CSVs ──────────────────────────────────────────────────────
    events_csv  = TOUCHES_OUT_DIR / f"{args.seq}_events.csv"
    summary_csv = TOUCHES_OUT_DIR / f"{args.seq}_summary.csv"
    save_events_csv(assigner.events,        events_csv)
    save_summary_csv(rows,                  summary_csv)
    print(f"\n  wrote: {events_csv}")
    print(f"  wrote: {summary_csv}")


if __name__ == "__main__":
    main()
