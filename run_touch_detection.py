"""
Live touch-detection viewer (mirrors play_ball_detection.py).

Prompts you to pick a clip, then plays it with ball detection + touch
detection running simultaneously.  Shows an OpenCV window with HUD,
player boxes, ball circle, possessor highlight, and touch flashes.

Controls (focus the OpenCV window):
    Q / ESC    quit
    SPACE      pause / resume
    N / ->     step one frame (while paused)
    R          reset ball detector
    P          toggle player bboxes

At the end: prints a Player ID -> Touches table and saves CSVs + touch map.

Usage:
    python run_touch_detection.py
    python run_touch_detection.py --seq v_gQNyhv8y0QY_c013
"""

from __future__ import annotations
import argparse, csv, time
from collections import defaultdict, deque
from configparser import ConfigParser
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ball_detect import BallDetector
from homography import PitchHomography
from pitch_config import world_xy_to_zone
from touch_assigner import MultiPlayerTouchAssigner, PlayerBox, TouchEvent

DATASET_TRAIN_DIR = Path("dataset/train")
HOMOGRAPHY_DIR    = Path("homographies")
TOUCHES_OUT_DIR   = Path("touches")
WINDOW_NAME       = "Touch Detection - Live"
FPS_WINDOW        = 30

# ── Sequence discovery + prompt (same style as play_ball_detection.py) ────────

def list_sequences(train_dir: Path) -> list[Path]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir.resolve()}")
    return sorted(p for p in train_dir.iterdir()
                  if p.is_dir() and (p / "img1").is_dir()
                  and (p / "gt" / "gt.txt").exists())

def read_seqinfo(seq_dir: Path) -> dict:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists(): return {}
    cp = ConfigParser(); cp.read(ini)
    return dict(cp["Sequence"]) if "Sequence" in cp else {}

def prompt_for_sequence(seqs: list[Path]) -> Path:
    print("\nAvailable sequences in dataset/train/:")
    for i, s in enumerate(seqs):
        info = read_seqinfo(s)
        n = info.get("seqLength", "?")
        w = info.get("imWidth", "?")
        h = info.get("imHeight", "?")
        fps = info.get("frameRate", "?")
        print(f"  [{i+1}] {s.name}   ({n} frames, {w}x{h}, {fps} fps)")
    while True:
        raw = input(f"\nPick a sequence [1-{len(seqs)}] (or Enter for 1): ").strip()
        if raw == "": return seqs[0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(seqs): return seqs[idx - 1]
        except ValueError: pass
        print("  invalid choice, try again.")

# ── GT loader ─────────────────────────────────────────────────────────────────

def load_gt_with_ids(seq_dir: Path) -> dict[int, list[PlayerBox]]:
    gt_file = seq_dir / "gt" / "gt.txt"
    out: dict[int, list[PlayerBox]] = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 6: continue
            try:
                fid, pid = int(parts[0]), int(parts[1])
                x, y, w, h = int(float(parts[2])), int(float(parts[3])), \
                              int(float(parts[4])), int(float(parts[5]))
            except ValueError: continue
            out[fid].append(PlayerBox(pid=pid, x=x, y=y, w=w, h=h))
    return dict(out)

# ── Key handler (same as play_ball_detection.py) ──────────────────────────────

def _handle_key(key: int) -> str | None:
    if key in (ord('q'), 27):  return "quit"
    if key == ord(' '):        return "pause"
    if key in (ord('n'), 83):  return "step"
    if key == ord('r'):        return "reset"
    if key == ord('p'):        return "toggle_players"
    return None

# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_hud(frame, seq_name, fid, total, fps, real, est, miss,
             paused, show_players, n_touches, poss_pid,
             ball_world=None, ball_zone=None):
    H, W = frame.shape[:2]
    box_h = 132 if ball_world else 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    pad = 10; white = (255,255,255)
    cv2.putText(frame, f"Seq: {seq_name}", (pad, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
    cv2.putText(frame, f"Frame {fid}/{total}   {fps:5.1f} FPS   "
                f"touches={n_touches}", (pad, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
    cv2.putText(frame, f"real={real}  est={est}  miss={miss}   "
                f"possessor={'#'+str(poss_pid) if poss_pid is not None else '-'}",
                (pad, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,255), 1)
    if ball_world:
        wx, wy = ball_world
        cv2.putText(frame, f"ball world: ({wx:5.1f}m, {wy:5.1f}m)   "
                    f"zone: {ball_zone or '?'}",
                    (pad, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,255,180), 1)
        cv2.putText(frame, "Q quit  SPACE pause  N step  R reset  P players",
                    (pad, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    else:
        cv2.putText(frame, "Q quit  SPACE pause  N step  R reset  P players",
                    (pad, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    if paused:
        cv2.putText(frame, "PAUSED", (W - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

def draw_players(frame, players):
    for pb in players:
        cv2.rectangle(frame, (pb.x, pb.y), (pb.x+pb.w, pb.y+pb.h), (90,90,90), 1)
        cv2.putText(frame, f"#{pb.pid}", (pb.x, max(pb.y-4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90,90,90), 1)

def draw_possessor(frame, players, poss_pid, touch_count, just_touched):
    if poss_pid is None: return
    for pb in players:
        if pb.pid == poss_pid:
            colour = (255, 80, 200)
            cv2.rectangle(frame, (pb.x, pb.y),
                          (pb.x+pb.w, pb.y+pb.h), colour, 2)
            tag = f"poss #{pb.pid}  touches={touch_count}"
            if just_touched: tag += "  TOUCH!"
            cv2.putText(frame, tag, (pb.x, max(pb.y-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
            break

def draw_ball(frame, ball_result, ball_xy_for_touch):
    """Draw ball circle: yellow=real, orange=estimated, big ring on touch-eligible."""
    if ball_result["ball_px"] is None: return
    px, py = int(ball_result["ball_px"]), int(ball_result["ball_py"])
    is_est = ball_result["is_estimated"]
    colour = (0, 165, 255) if is_est else (0, 255, 255)
    label = "ball~" if is_est else "ball"
    conf_s = f"{ball_result['confidence']:.2f}" if not is_est else "est"
    cv2.circle(frame, (px, py), 12, colour, 2)
    cv2.circle(frame, (px, py), 3, colour, -1)
    cv2.putText(frame, f"{label} {conf_s}", (px+14, py-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

def draw_roi_box(frame, detector):
    if not detector.kf_initialised: return
    state = detector.kf.statePost.flatten()
    cx, cy = float(state[0]), float(state[1])
    half = detector.roi_size // 2
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(cx-half)), max(0, int(cy-half))
    x1, y1 = min(W, int(cx+half)), min(H, int(cy+half))
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 0), 1)

def draw_touch_flash(frame, ball_xy):
    if ball_xy:
        cv2.circle(frame, (int(ball_xy[0]), int(ball_xy[1])), 24, (0,255,255), 3)

# ── Final table ───────────────────────────────────────────────────────────────

def print_touch_table(assigner, seq_name, n_frames, elapsed,
                      n_real, n_est, n_miss):
    rows = assigner.summary_table()
    recall = 100.0 * (n_real + n_est) / max(n_frames, 1)
    w = 70
    print()
    print("=" * w)
    print(f"  TOUCH DETECTION RESULTS  -  {seq_name}")
    print("=" * w)
    print(f"  Frames processed   : {n_frames}")
    print(f"  Wall time          : {elapsed:.1f}s "
          f"({n_frames/max(elapsed,0.01):.1f} fps)")
    print(f"  Ball real/est/miss : {n_real} / {n_est} / {n_miss}  "
          f"({recall:.1f}% recall)")
    print(f"  Total touches      : {len(assigner.events)}")
    print(f"  Players w/ touches : {len(rows)}")
    print()
    print(f"  {'Player ID':>10}  {'Touches':>8}  {'First':>6}  "
          f"{'Last':>6}  Zones")
    print("  " + "-" * (w - 4))
    for r in rows:
        z = r["zones"] if r["zones"] else "-"
        print(f"  {r['player_id']:>10}  {r['touch_count']:>8}  "
              f"{r['first_touch_frame']:>6}  {r['last_touch_frame']:>6}  {z}")
    print("=" * w)

# ── CSV savers ────────────────────────────────────────────────────────────────

def save_events_csv(events, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id","player_id","ball_px","ball_py",
                     "world_x_m","world_y_m","zone","h_supported",
                     "foot_dist_px","bbox_h_px"])
        for ev in events:
            wx, wy = ev.world_xy if ev.world_xy else ("","")
            w.writerow([ev.frame_id, ev.player_id,
                        f"{ev.ball_px:.1f}", f"{ev.ball_py:.1f}",
                        f"{wx:.2f}" if isinstance(wx,float) else "",
                        f"{wy:.2f}" if isinstance(wy,float) else "",
                        ev.zone or "", int(ev.h_supported),
                        f"{ev.foot_dist_px:.1f}", ev.bbox_h_px])

def save_summary_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id","touch_count","first_touch_frame",
                     "last_touch_frame","zones","touches_outside_H_support"])
        for r in rows:
            w.writerow([r["player_id"], r["touch_count"],
                        r["first_touch_frame"], r["last_touch_frame"],
                        r["zones"], r["touches_outside_H_support"]])

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Live touch detection viewer")
    ap.add_argument("--seq", default=None, help="Sequence name (skip prompt)")
    ap.add_argument("--proximity-factor", type=float, default=1.0)
    ap.add_argument("--min-hold", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=8)
    ap.add_argument("--foot-zone", type=float, default=0.4)
    ap.add_argument("--foot-zone-margin", type=float, default=15.0)
    ap.add_argument("--ball-conf", type=float, default=0.15)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--save", default=None, help="Save annotated video .mp4")
    args = ap.parse_args()

    # ── Pick sequence ─────────────────────────────────────────────────
    seqs = list_sequences(DATASET_TRAIN_DIR)
    if not seqs:
        print(f"[error] No sequences under {DATASET_TRAIN_DIR}"); return

    if args.seq:
        match = [s for s in seqs if s.name == args.seq]
        if not match:
            print(f"[error] '{args.seq}' not found. "
                  f"Available: {[s.name for s in seqs]}"); return
        seq = match[0]
    else:
        seq = prompt_for_sequence(seqs)

    info = read_seqinfo(seq)
    total = int(info.get("seqLength", 0))
    fps_native = float(info.get("frameRate", 25))
    print(f"\n[viewer] Sequence : {seq.name}")
    print(f"[viewer] Frames   : {total}  @ {fps_native} fps")

    # ── Frame paths ───────────────────────────────────────────────────
    frame_paths = sorted((seq / "img1").glob("*.jpg"))
    if not frame_paths:
        print(f"[error] No frames in {seq / 'img1'}"); return
    start = max(1, args.start)
    end = args.end if args.end else len(frame_paths)
    end = min(end, len(frame_paths))
    frame_paths = [p for p in frame_paths if start <= int(p.stem) <= end]
    print(f"[viewer] Playing  : frames {start}..{end} ({len(frame_paths)} frames)")

    # ── GT tracks ─────────────────────────────────────────────────────
    gt = load_gt_with_ids(seq)
    n_pids = len({pb.pid for plist in gt.values() for pb in plist})
    print(f"[viewer] GT tracks: {len(gt)} frames, {n_pids} unique players")

    # ── Homography ────────────────────────────────────────────────────
    homo: Optional[PitchHomography] = None
    homo_path = HOMOGRAPHY_DIR / f"{seq.name}.npz"
    if homo_path.exists():
        try:
            homo = PitchHomography.load(homo_path)
            print(f"[viewer] Homography: {homo_path}")
        except Exception as e:
            print(f"[viewer] Homography failed: {e}")
    else:
        print(f"[viewer] No homography (zones will be empty)")

    # ── Init detector + assigner ──────────────────────────────────────
    detector = BallDetector(conf_threshold=args.ball_conf, verbose=True)
    assigner = MultiPlayerTouchAssigner(
        proximity_factor    = args.proximity_factor,
        min_hold_frames     = args.min_hold,
        cooldown_frames     = args.cooldown,
        foot_zone_fraction  = args.foot_zone,
        foot_zone_margin_px = args.foot_zone_margin,
    )

    # ── Optional video writer ─────────────────────────────────────────
    writer = None
    if args.save:
        sample = cv2.imread(str(frame_paths[0]))
        h_img, w_img = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_native, (w_img, h_img))
        print(f"[viewer] Saving video -> {args.save}")

    # ── Window ────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    # ── Main loop ─────────────────────────────────────────────────────
    real = est = miss = 0
    paused = False; show_players = True; step_once = False
    recent_dt = deque(maxlen=FPS_WINDOW)
    i = 0

    while i < len(frame_paths):
        # Pause handling
        if paused and not step_once:
            key = cv2.waitKey(20) & 0xFF
            handled = _handle_key(key)
            if handled == "quit":       break
            elif handled == "pause":    paused = False
            elif handled == "step":     step_once = True
            elif handled == "reset":
                detector.reset(); real = est = miss = 0
                print("[viewer] detector reset")
            elif handled == "toggle_players": show_players = not show_players
            continue

        step_once = False
        fp = frame_paths[i]
        fid = int(fp.stem)
        frame = cv2.imread(str(fp))
        if frame is None: i += 1; continue

        players = gt.get(fid, [])
        gt_bboxes = [(pb.x, pb.y, pb.w, pb.h) for pb in players] or None

        # ── STEP 1: Ball detection ────────────────────────────────
        t0 = time.perf_counter()
        result = detector.track_ball(frame, player_bboxes=gt_bboxes)
        dt = time.perf_counter() - t0
        recent_dt.append(dt)
        cur_fps = 1.0 / (sum(recent_dt)/len(recent_dt)) if recent_dt else 0

        if result["ball_px"] is None:     miss += 1; ball_xy = None
        elif result["is_estimated"]:      est += 1;  ball_xy = None
        else:
            real += 1
            ball_xy = (float(result["ball_px"]), float(result["ball_py"]))

        # ── STEP 2: Touch detection (simultaneous) ────────────────
        emitted = assigner.update(fid, ball_xy, players, homography=homo)
        just_touched = emitted is not None
        poss_pid = assigner.possessor_per_frame.get(fid)
        poss_touches = assigner.touches_per_player.get(poss_pid, 0) if poss_pid else 0

        # ── World coords for HUD ──────────────────────────────────
        ball_world = ball_zone = None
        if homo and result["ball_px"] is not None:
            wx, wy = homo.pixel_to_world(result["ball_px"], result["ball_py"])
            ball_world = (wx, wy)
            ball_zone = world_xy_to_zone(wx, wy)

        # ── Draw everything ───────────────────────────────────────
        if show_players:
            draw_players(frame, players)
        draw_roi_box(frame, detector)
        draw_possessor(frame, players, poss_pid, poss_touches, just_touched)
        draw_ball(frame, result, ball_xy)
        if just_touched: draw_touch_flash(frame, ball_xy)
        draw_hud(frame, seq.name, fid, end, cur_fps,
                 real, est, miss, paused, show_players,
                 len(assigner.events), poss_pid,
                 ball_world, ball_zone)

        cv2.imshow(WINDOW_NAME, frame)
        if writer: writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        handled = _handle_key(key)
        if handled == "quit":       break
        elif handled == "pause":    paused = True
        elif handled == "reset":
            detector.reset(); real = est = miss = 0
            print("[viewer] detector reset")
        elif handled == "toggle_players": show_players = not show_players

        i += 1

    cv2.destroyAllWindows()
    if writer: writer.release(); print(f"[viewer] saved {args.save}")

    elapsed = sum(recent_dt) if recent_dt else 0.01
    n_processed = real + est + miss

    # ── Final table ───────────────────────────────────────────────
    print_touch_table(assigner, seq.name, n_processed, elapsed,
                      real, est, miss)

    # ── Save outputs ──────────────────────────────────────────────
    TOUCHES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_events_csv(assigner.events,
                    TOUCHES_OUT_DIR / f"{seq.name}_events.csv")
    save_summary_csv(assigner.summary_table(),
                     TOUCHES_OUT_DIR / f"{seq.name}_summary.csv")
    map_path = TOUCHES_OUT_DIR / f"{seq.name}_touchmap.png"
    assigner.render_touch_map(str(map_path))
    print(f"\n  Saved: {TOUCHES_OUT_DIR / seq.name}_events.csv")
    print(f"  Saved: {TOUCHES_OUT_DIR / seq.name}_summary.csv")
    print(f"  Saved: {map_path}")

if __name__ == "__main__":
    main()
