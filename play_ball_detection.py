"""
Live ball-detection viewer.

Prompts you to pick one of the sequences in `dataset/train/`, then plays
the entire clip frame 1 -> last with the refactored BallDetector running
live so you can eyeball the detections.

Controls (focus on the OpenCV window):
    Q or ESC   - quit
    SPACE      - pause / resume
    -> or n    - while paused, advance one frame
    R          - reset the BallDetector (re-acquire from the next frame)

What you should see:
    YELLOW circle  -> real YOLO detection
    ORANGE circle  -> Kalman filter estimate (detection missed; up to 10
                      consecutive frames before the track is dropped)
    GREEN dashed   -> ROI box used by the prediction-guided search

HUD shows: sequence name, current frame / total, FPS (rolling avg over
30 frames), real / est / miss counters.

Usage:
    python play_ball_detection.py
    python play_ball_detection.py --seq v_gQNyhv8y0QY_c013    # skip prompt
    python play_ball_detection.py --no-fp-filter              # disable FP filter
"""

from __future__ import annotations

import argparse
import time
import config
from collections import defaultdict, deque
from configparser import ConfigParser
from pathlib import Path

import cv2
import numpy as np

from ball_detect import BallDetector
from homography import PitchHomography
from pitch_config import world_xy_to_zone


# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_TRAIN_DIR = config.datasetRoot / config.testSplit
HOMOGRAPHY_DIR    = Path("homographies")
WINDOW_NAME       = "Ball Detection - Live"
FPS_WINDOW        = 30          # frames over which to compute rolling FPS


# ─── Sequence picker ──────────────────────────────────────────────────────────

def list_sequences(train_dir: Path) -> list[Path]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir.resolve()}")
    seqs = sorted(p for p in train_dir.iterdir()
                  if p.is_dir() and (p / "img1").is_dir())
    return seqs


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
        if raw == "":
            return seqs[0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(seqs):
                return seqs[idx - 1]
        except ValueError:
            pass
        print("  invalid choice, try again.")


def read_seqinfo(seq_dir: Path) -> dict:
    """Parse seqinfo.ini if present; returns {} on failure."""
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        return {}
    cp = ConfigParser()
    cp.read(ini)
    if "Sequence" not in cp:
        return {}
    return dict(cp["Sequence"])


# ─── GT bbox loader (used as FP filter) ───────────────────────────────────────

def load_gt(seq_dir: Path) -> dict[int, list[tuple]]:
    """Returns {frame_id: [(x, y, w, h), ...]} or {} if no GT file."""
    gt_file = seq_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        return {}
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


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_roi_box(frame: np.ndarray, detector: BallDetector) -> None:
    """If the detector is locked, draw the ROI it would search next."""
    if not detector.kf_initialised:
        return
    state = detector.kf.statePost.flatten()
    cx, cy = float(state[0]), float(state[1])
    half = detector.roi_size // 2
    H, W = frame.shape[:2]
    x0 = max(0, int(cx - half));  y0 = max(0, int(cy - half))
    x1 = min(W, int(cx + half));  y1 = min(H, int(cy + half))
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 0), 1)


def draw_possessor(frame: np.ndarray, detector: BallDetector) -> None:
    """Highlight the player bbox the tracker thinks currently has the ball."""
    bb = detector.possessor_bbox
    if bb is None:
        return
    x, y, w, h = bb[0], bb[1], bb[2], bb[3]
    # Magenta box + label so it can't be confused with the green ROI box
    colour = (255, 80, 200)
    cv2.rectangle(frame, (int(x), int(y)),
                  (int(x + w), int(y + h)), colour, 2)
    label = f"possessor (age={detector.possessor_age})"
    cv2.putText(frame, label, (int(x), max(int(y) - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


def draw_player_bboxes(frame: np.ndarray, bboxes) -> None:
    for (x, y, w, h) in bboxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)


def draw_hud(
    frame: np.ndarray,
    seq_name: str,
    frame_idx: int,
    total: int,
    fps: float,
    real: int,
    est: int,
    miss: int,
    paused: bool,
    show_players: bool,
    ball_world: tuple | None = None,
    ball_zone:  str | None  = None,
) -> None:
    H, W = frame.shape[:2]
    pad = 10
    box_h = 132 if ball_world is not None else 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    base = (255, 255, 255)
    cv2.putText(frame, f"Seq: {seq_name}", (pad, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, base, 2)
    cv2.putText(frame, f"Frame {frame_idx}/{total}   {fps:5.1f} FPS",
                (pad, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, base, 1)
    cv2.putText(frame,
                f"real={real}  est={est}  miss={miss}",
                (pad, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    if ball_world is not None:
        wx, wy = ball_world
        zone_s = ball_zone or "?"
        cv2.putText(frame,
                    f"ball world: ({wx:5.1f}m, {wy:5.1f}m)   zone: {zone_s}",
                    (pad, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (180, 255, 180), 1)
        cv2.putText(frame,
                    "Q/ESC quit  SPACE pause  N step  R reset  P players",
                    (pad, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    else:
        cv2.putText(frame,
                    "Q/ESC quit   SPACE pause   N step   R reset   "
                    "P toggle players",
                    (pad, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)

    if paused:
        cv2.putText(frame, "PAUSED", (W - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# ─── Main viewer ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None,
                    help="Sequence folder name (skip the picker)")
    ap.add_argument("--no-fp-filter", action="store_true",
                    help="Disable the player-bbox false-positive filter")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="Detector confidence threshold (default 0.15)")
    ap.add_argument("--imgsz-full", type=int, default=640)
    ap.add_argument("--imgsz-roi",  type=int, default=320)
    ap.add_argument("--roi-size",   type=int, default=320)
    ap.add_argument("--start", type=int, default=1,
                    help="Start frame number (default 1)")
    ap.add_argument("--end", type=int, default=None,
                    help="End frame number (inclusive, default = last)")
    ap.add_argument("--save", default=None,
                    help="Optional path to save the annotated video as .mp4")
    args = ap.parse_args()

    # ── Pick sequence ──────────────────────────────────────────────────────
    seqs = list_sequences(DATASET_TRAIN_DIR)
    if not seqs:
        print(f"[error] No sequences found under {DATASET_TRAIN_DIR}")
        return

    if args.seq:
        match = [s for s in seqs if s.name == args.seq]
        if not match:
            print(f"[error] Sequence '{args.seq}' not found. "
                  f"Available: {[s.name for s in seqs]}")
            return
        seq = match[0]
    else:
        seq = prompt_for_sequence(seqs)

    info = read_seqinfo(seq)
    total = int(info.get("seqLength", 0))
    fps_native = float(info.get("frameRate", 25))
    print(f"\n[viewer] Sequence : {seq.name}")
    print(f"[viewer] Frames   : {total}  @ {fps_native} fps")

    # ── Frame paths ────────────────────────────────────────────────────────
    img_dir = seq / "img1"
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"[error] No frames in {img_dir}")
        return

    start = max(1, args.start)
    end   = args.end if args.end else len(frame_paths)
    end   = min(end, len(frame_paths))
    frame_paths = [p for p in frame_paths
                   if start <= int(p.stem) <= end]
    print(f"[viewer] Playing : frames {start}..{end} "
          f"({len(frame_paths)} frames)")

    # ── GT for FP filter ───────────────────────────────────────────────────
    gt = {} if args.no_fp_filter else load_gt(seq)
    if args.no_fp_filter:
        print("[viewer] FP filter : DISABLED (--no-fp-filter)")
    elif gt:
        print(f"[viewer] FP filter : ON  (GT bboxes for {len(gt)} frames)")
    else:
        print("[viewer] FP filter : OFF (no gt.txt found)")

    # ── Homography (optional) ──────────────────────────────────────────────
    homo: PitchHomography | None = None
    homo_path = HOMOGRAPHY_DIR / f"{seq.name}.npz"
    if homo_path.exists():
        try:
            homo = PitchHomography.load(homo_path)
            print(f"[viewer] Homography: {homo_path}  ({homo})")
        except Exception as e:
            print(f"[viewer] Homography: failed to load {homo_path}: {e}")
    else:
        print(f"[viewer] Homography: not found at {homo_path} "
              f"(run compute_homographies.py to enable world coords)")

    # ── Detector ───────────────────────────────────────────────────────────
    detector = BallDetector(
        conf_threshold = args.conf,
        imgsz_full     = args.imgsz_full,
        imgsz_roi      = args.imgsz_roi,
        roi_size       = args.roi_size,
        verbose        = True,
    )

    # ── Optional video writer ──────────────────────────────────────────────
    writer = None
    if args.save:
        sample = cv2.imread(str(frame_paths[0]))
        if sample is None:
            print(f"[error] Could not read {frame_paths[0]}")
            return
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_native, (w, h))
        print(f"[viewer] Saving annotated video -> {args.save}")

    # ── Loop ───────────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    real = est = miss = 0
    paused = False
    show_players = False
    step_once = False
    recent_dt = deque(maxlen=FPS_WINDOW)
    i = 0

    while i < len(frame_paths):
        if paused and not step_once:
            key = cv2.waitKey(20) & 0xFF
            handled = _handle_key(key)
            if handled == "quit":
                break
            elif handled == "pause":
                paused = not paused
            elif handled == "step":
                step_once = True
            elif handled == "reset":
                detector.reset()
                real = est = miss = 0
                print("[viewer] detector reset")
            elif handled == "toggle_players":
                show_players = not show_players
            continue

        step_once = False
        p = frame_paths[i]
        fid = int(p.stem)
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"  [warn] could not read frame {fid}")
            i += 1
            continue

        bboxes = gt.get(fid) if gt else None

        t0 = time.perf_counter()
        result = detector.track_ball(frame, player_bboxes=bboxes)
        dt = time.perf_counter() - t0
        recent_dt.append(dt)
        cur_fps = 1.0 / (sum(recent_dt) / len(recent_dt)) if recent_dt else 0.0

        if result["ball_px"] is None:
            miss += 1
        elif result["is_estimated"]:
            est += 1
        else:
            real += 1

        # ── Compute world coords if homography available ──
        ball_world: tuple | None = None
        ball_zone:  str | None  = None
        if homo is not None and result["ball_px"] is not None:
            wx, wy = homo.pixel_to_world(result["ball_px"], result["ball_py"])
            ball_world = (wx, wy)
            ball_zone  = world_xy_to_zone(wx, wy)

        # ── Draw ──
        if show_players and bboxes:
            draw_player_bboxes(frame, bboxes)
        draw_roi_box(frame, detector)
        draw_possessor(frame, detector)
        detector.draw_ball(frame, result)
        draw_hud(frame, seq.name, fid, end, cur_fps,
                 real, est, miss, paused, show_players,
                 ball_world=ball_world, ball_zone=ball_zone)

        cv2.imshow(WINDOW_NAME, frame)
        if writer is not None:
            writer.write(frame)

        # waitKey at 1ms - the inference itself is the throttle
        key = cv2.waitKey(1) & 0xFF
        handled = _handle_key(key)
        if handled == "quit":
            break
        elif handled == "pause":
            paused = True
        elif handled == "reset":
            detector.reset()
            real = est = miss = 0
            print("[viewer] detector reset")
        elif handled == "toggle_players":
            show_players = not show_players

        i += 1

    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[viewer] saved {args.save}")

    # ── Final report ──
    n = max(real + est + miss, 1)
    print("\n=== Run summary ===")
    print(f"  Frames seen      : {real + est + miss}")
    print(f"  Real detections  : {real:5d}  ({100*real/n:5.1f}%)")
    print(f"  Kalman estimates : {est:5d}  ({100*est/n:5.1f}%)")
    print(f"  Missed           : {miss:5d}  ({100*miss/n:5.1f}%)")
    print(f"  Effective recall : {100*(real+est)/n:5.1f}%")


def _handle_key(key: int) -> str | None:
    if key in (ord('q'), 27):
        return "quit"
    if key == ord(' '):
        return "pause"
    if key in (ord('n'), 83):       # 'n' or right-arrow
        return "step"
    if key == ord('r'):
        return "reset"
    if key == ord('p'):
        return "toggle_players"
    return None


if __name__ == "__main__":
    main()
