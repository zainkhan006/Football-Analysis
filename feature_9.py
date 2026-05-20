"""
Feature 9 — Jersey Metadata
============================
S3  Jersey colour naming  (K-means on HSV crops from Feature 8)
S4  GK/Referee split + clash detection

Reads from:
    output/feature_8/<seq>/jersey_crops/{home,away,gk}/   (S3)
    videos/<seq>/img1/                                      (frames)
    videos/<seq>/gt/gt.txt                                  (GT boxes)

Writes to:
    output/feature_9/<seq>/overlays/    ← annotated frames (main output)
    output/feature_9/<seq>/colours.txt  ← colour summary per sequence
    output/feature_9/<seq>/output_video.mp4  ← video output

Each output overlay shows:
  • Coloured bounding boxes per player  (green=home, blue=away, amber=gk, magenta=referee)
  • Label above each box: #trackId  home|away|gk
  • Header bar at the top:
        home:<colour>  away:<colour>  gk:<colour>
        referee: DETECTED(frame X) | NOT DETECTED
        GK: home→#tid  away→#tid

Usage
-----
    python feature_9.py                        # all sequences, first 50 frames
    python feature_9.py --seq v_HdiyOtliFiw_c003
    python feature_9.py --seq v_HdiyOtliFiw_c003 --frames 100
    python feature_9.py --seq v_HdiyOtliFiw_c003 --frames 50 --no-video
    python feature_9.py --seq v_HdiyOtliFiw_c003 --frames 100 --play
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ── import Feature 8 helpers ────────────────────────────────────────────
try:
    import config

    _DATASET_ROOT = config.datasetRoot / config.testSplit
except ImportError:
    config = None
    _DATASET_ROOT = Path(r"C:\Users\samee\Documents\Computer Vision\project\videos")

sys.path.append(str(Path(__file__).parent))
from feature_8 import (
    FeatureConfig,
    crop_jersey,
    _describe,
    _kmeans_robust,
    _load_all_gt,
    _collect_frame_paths,
    _frame_path_for_idx,
    _frame_index_of,
    apply_majority_vote,
    assign_teams,
)

# ══════════════════════════════════════════════════════════════════════
# Colours for bounding-box drawing
# ══════════════════════════════════════════════════════════════════════

_BOX_COLOUR = {
    "home": (60, 200, 60),  # green
    "away": (60, 120, 220),  # blue
    "gk": (220, 180, 40),  # amber
    "referee": (220, 60, 220),  # magenta
    "?": (120, 120, 120),
}

# Video settings
_VIDEO_FPS = 10


# ══════════════════════════════════════════════════════════════════════
# S3 — Jersey colour naming
# ══════════════════════════════════════════════════════════════════════

_MIN_VALUE_FOR_CLUSTERING = 20
_MIN_CLUSTER_SHARE = 0.25
_KMEANS_K = 4


def _hsv_to_name(h: float, s: float, v: float) -> str:
    if v < 60:
        return "black"
    if s < 50:
        return "white" if v > 170 else "grey"
    if h <= 10 or h >= 165:
        return "red"
    if h <= 25:
        return "orange"
    if h <= 34:
        return "yellow"
    if h <= 85:
        return "green"
    if h <= 99:
        return "cyan"
    if h <= 130:
        return "blue"
    if h <= 150:
        return "purple"
    return "pink"


def _colour_for_folder(folder: Path) -> str:
    """
    Pool all crops in folder, cluster HSV pixels, return colour name(s).
    Plain kit  → single name.
    Designed kit (stripes/hoops) → 'name1/name2'.
    """
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    if not paths:
        return "unknown"

    pooled: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 4 or w < 4:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        flat = hsv.reshape(-1, 3)
        flat = flat[flat[:, 2] >= _MIN_VALUE_FOR_CLUSTERING]
        if len(flat) == 0:
            continue
        # mask grass
        grass = ~((flat[:, 0] >= 36) & (flat[:, 0] <= 90) & (flat[:, 1] >= 35))
        flat = flat[grass]
        if len(flat) == 0:
            continue
        if len(flat) > 400:
            idx = np.random.choice(len(flat), 400, replace=False)
            flat = flat[idx]
        pooled.append(flat)

    if not pooled:
        return "unknown"

    all_px = np.vstack(pooled)
    if len(all_px) < _KMEANS_K * 4:
        return "unknown"

    samples = all_px.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels_km, centres = cv2.kmeans(
        samples, _KMEANS_K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels_km = labels_km.flatten()

    clusters = []
    for k in range(_KMEANS_K):
        cnt = int(np.sum(labels_km == k))
        if cnt:
            clusters.append(
                ((int(centres[k][0]), int(centres[k][1]), int(centres[k][2])), cnt)
            )
    clusters.sort(key=lambda c: c[1], reverse=True)

    total = sum(c for _, c in clusters)
    merged: Dict[str, int] = {}
    for hsv_val, cnt in clusters:
        name = _hsv_to_name(*hsv_val)
        merged[name] = merged.get(name, 0) + cnt

    kept: List[str] = [
        name
        for name, cnt in sorted(merged.items(), key=lambda x: x[1], reverse=True)
        if cnt / total >= _MIN_CLUSTER_SHARE
    ]
    if not kept:
        kept = [_hsv_to_name(*clusters[0][0])]
    return "/".join(kept)


def name_jersey_colours(jersey_crops_dir: Path) -> Dict[str, str]:
    """Return {home_colour, away_colour, gk_colour}."""
    result: Dict[str, str] = {}
    for team in ("home", "away", "gk"):
        folder = jersey_crops_dir / team
        if not folder.is_dir():
            result[f"{team}_colour"] = "unknown"
        else:
            result[f"{team}_colour"] = _colour_for_folder(folder)
    return result


# ══════════════════════════════════════════════════════════════════════
# S4-A — GK identification (majority-vote per team)
# ══════════════════════════════════════════════════════════════════════


def identify_goalkeepers(
    stable_labels: Dict[int, Dict[int, str]],
) -> Dict[str, Optional[int]]:
    """
    Among all tracks labelled "gk", find the one most likely belonging to
    each team.  Strategy:
        • For every frame, count home vs away players.
        • Each "gk" track gets a per-frame vote for the team that has MORE
          players in that frame (they are defending — more players visible).
        • The gk track with the most votes for "home" → home GK, and vice
          versa for "away".  If only one gk track exists, assign it to
          whichever team it votes for most.

    Returns {home_gk: track_id|None, away_gk: track_id|None}
    """
    # Collect which tracks are ever labelled gk
    gk_track_ids: set = set()
    for frame_labels in stable_labels.values():
        for tid, lbl in frame_labels.items():
            if lbl == "gk":
                gk_track_ids.add(tid)

    if not gk_track_ids:
        return {"home_gk": None, "away_gk": None}

    # For each gk track, vote which team it belongs to
    gk_votes: Dict[int, Dict[str, int]] = {
        tid: {"home": 0, "away": 0} for tid in gk_track_ids
    }

    for frame_labels in stable_labels.values():
        home_count = sum(1 for l in frame_labels.values() if l == "home")
        away_count = sum(1 for l in frame_labels.values() if l == "away")
        dominant_team = "home" if home_count >= away_count else "away"

        for tid, lbl in frame_labels.items():
            if lbl == "gk" and tid in gk_votes:
                gk_votes[tid][dominant_team] += 1

    # Sort gk tracks by their dominant team vote
    home_gk_candidates = sorted(
        gk_track_ids, key=lambda t: gk_votes[t]["home"], reverse=True
    )
    away_gk_candidates = sorted(
        gk_track_ids, key=lambda t: gk_votes[t]["away"], reverse=True
    )

    home_gk = home_gk_candidates[0] if home_gk_candidates else None
    away_gk = None

    # Second gk track (if it exists) gets the other team
    if len(gk_track_ids) >= 2:
        for tid in away_gk_candidates:
            if tid != home_gk:
                away_gk = tid
                break

    return {"home_gk": home_gk, "away_gk": away_gk}


# ══════════════════════════════════════════════════════════════════════
# S4-B — Colour clash detection
# ══════════════════════════════════════════════════════════════════════

# Pairs of colour names considered clashing (same or very similar)
_CLASH_PAIRS = {
    frozenset(["white", "white"]),
    frozenset(["black", "black"]),
    frozenset(["red", "red"]),
    frozenset(["blue", "blue"]),
    frozenset(["green", "green"]),
    frozenset(["yellow", "yellow"]),
    frozenset(["orange", "orange"]),
    frozenset(["grey", "grey"]),
    frozenset(["purple", "purple"]),
    frozenset(["cyan", "cyan"]),
    frozenset(["pink", "pink"]),
    # near-clashes
    frozenset(["red", "orange"]),
    frozenset(["blue", "cyan"]),
    frozenset(["green", "cyan"]),
    frozenset(["yellow", "orange"]),
    frozenset(["purple", "blue"]),
    frozenset(["pink", "red"]),
    frozenset(["grey", "white"]),
    frozenset(["black", "grey"]),
}


def check_colour_clash(home_colour: str, away_colour: str) -> bool:
    """Return True if home and away primary colours are likely to clash."""
    home_primary = home_colour.split("/")[0]
    away_primary = away_colour.split("/")[0]
    return frozenset([home_primary, away_primary]) in _CLASH_PAIRS


# ══════════════════════════════════════════════════════════════════════
# S4-C — Referee detection (YOLO vs GT mismatch)
# ══════════════════════════════════════════════════════════════════════


def _compute_iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _box_center(box: Tuple) -> Tuple[int, int]:
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def _find_unmatched_yolo(
    gt_boxes: List[Tuple], yolo_boxes: List[Tuple], iou_thresh: float = 0.5
) -> List[Tuple]:
    matched: set = set()
    for gt in gt_boxes:
        best_iou, best_idx = 0.0, -1
        for i, yb in enumerate(yolo_boxes):
            if i in matched:
                continue
            iou = _compute_iou(gt, yb)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thresh and best_idx >= 0:
            matched.add(best_idx)
    return [yolo_boxes[i] for i in range(len(yolo_boxes)) if i not in matched]


def _is_vertical_outlier(candidate: Tuple, gt_boxes: List[Tuple]) -> bool:
    if not gt_boxes:
        return False
    cy = _box_center(candidate)[1]
    gt_ys = [_box_center(b)[1] for b in gt_boxes]
    return cy < min(gt_ys) or cy > max(gt_ys)


def detect_referee_in_frame(
    frame: np.ndarray,
    gt_boxes: List[Tuple],
    model: "YOLO",
    iou_thresh: float = 0.5,
    conf: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Run YOLO, find unmatched boxes, pick closest to GT centroid, validate
    vertical position.  Returns (x,y,w,h) or None.
    """
    results = model(frame, classes=[0], conf=conf, verbose=False)
    yolo_boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            yolo_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    if not yolo_boxes or not gt_boxes:
        return None

    unmatched = _find_unmatched_yolo(gt_boxes, yolo_boxes, iou_thresh)
    if not unmatched:
        return None

    # Pick unmatched box with smallest average distance to GT centres
    def _avg_dist(box):
        cx, cy = _box_center(box)
        dists = [
            np.hypot(cx - _box_center(g)[0], cy - _box_center(g)[1]) for g in gt_boxes
        ]
        return sum(dists) / len(dists)

    best = min(unmatched, key=_avg_dist)
    if _is_vertical_outlier(best, gt_boxes):
        return None
    return best


# ══════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════

_HEADER_H = 80  # pixels for the info bar at top
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.50
_FONT_THICK = 1
_FONT_SCALE_BIG = 0.55
_FONT_THICK_BIG = 2


def _draw_header(
    frame: np.ndarray,
    frame_idx: int,
    colours: Dict[str, str],
    gk_ids: Dict[str, Optional[int]],
    referee_box: Optional[Tuple],
    clash: bool,
) -> np.ndarray:
    """
    Prepend a dark info bar above the frame.
    Lines:
      Line 1:  Frame XXXXXX   home:<col>  away:<col>  gk:<col>
      Line 2:  GK: home→#tid  away→#tid
      Line 3:  Referee: DETECTED at (cx,cy) | NOT DETECTED
    """
    H, W = frame.shape[:2]
    bar = np.zeros((_HEADER_H, W, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    home_col = colours.get("home_colour", "?")
    away_col = colours.get("away_colour", "?")
    gk_col = colours.get("gk_colour", "?")
    home_gk = gk_ids.get("home_gk")
    away_gk = gk_ids.get("away_gk")

    # Line 1 — frame index + team colours
    line1 = (
        f"Frame {frame_idx:06d}   " f"home:{home_col}   away:{away_col}   gk:{gk_col}"
    )
    cv2.putText(
        bar,
        line1,
        (8, 22),
        _FONT,
        _FONT_SCALE_BIG,
        (230, 230, 230),
        _FONT_THICK_BIG,
        cv2.LINE_AA,
    )

    # Line 2 — GK ids + clash warning
    home_gk_str = f"#{home_gk}" if home_gk is not None else "none"
    away_gk_str = f"#{away_gk}" if away_gk is not None else "none"
    line2 = f"GK: home->{home_gk_str}  away->{away_gk_str}"

    clash_col = (0, 100, 255) if clash else (180, 180, 180)
    cv2.putText(
        bar, line2, (8, 47), _FONT, _FONT_SCALE, clash_col, _FONT_THICK, cv2.LINE_AA
    )

    # Line 3 — referee
    if referee_box is not None:
        cx, cy = _box_center(referee_box)
        ref_text = f"Referee: DETECTED at ({cx},{cy})"
        ref_colour = (0, 255, 180)
    else:
        ref_text = "Referee: NOT DETECTED"
        ref_colour = (100, 100, 100)
    cv2.putText(
        bar, ref_text, (8, 70), _FONT, _FONT_SCALE, ref_colour, _FONT_THICK, cv2.LINE_AA
    )

    return np.vstack([bar, frame])


def _draw_boxes(
    frame: np.ndarray,
    entries: List[Tuple[int, Tuple]],
    labels: Dict[int, str],
    gk_ids: Dict[str, Optional[int]],
    referee_box: Optional[Tuple],
) -> np.ndarray:
    """Draw all player boxes and the referee box."""
    out = frame.copy()

    for tid, (x, y, w, h) in entries:
        lbl = labels.get(tid, "?")
        color = _BOX_COLOUR.get(lbl, _BOX_COLOUR["?"])

        # Thicker box for goalkeepers
        thickness = (
            3 if (tid == gk_ids.get("home_gk") or tid == gk_ids.get("away_gk")) else 2
        )

        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        tag = f"#{tid} {lbl}"
        if tid == gk_ids.get("home_gk"):
            tag += " (GK-H)"
        elif tid == gk_ids.get("away_gk"):
            tag += " (GK-A)"
        cv2.putText(out, tag, (x, max(y - 4, 12)), _FONT, 0.42, color, 1, cv2.LINE_AA)

    # Referee
    if referee_box is not None:
        x, y, w, h = referee_box
        cv2.rectangle(out, (x, y), (x + w, y + h), _BOX_COLOUR["referee"], 3)
        cv2.putText(
            out,
            "REFEREE",
            (x, max(y - 4, 12)),
            _FONT,
            0.50,
            _BOX_COLOUR["referee"],
            2,
            cv2.LINE_AA,
        )

    return out


def play_video(video_path: Path, fps: int = 25) -> None:
    """Play the generated video file."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    print(f"Playing video: {video_path}")
    print("Press 'q' or ESC to quit, SPACE to pause/resume")
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        if not paused and ret:
            cv2.imshow("Feature 9 - Jersey Metadata", frame)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("Paused... Press SPACE to resume")
            else:
                print("Resuming...")
    
    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════
# Main per-sequence runner
# ══════════════════════════════════════════════════════════════════════


def run_feature9_sequence(
    seq_dir: Path,
    out_root: Path,
    first_n: int = 50,
    yolo_conf: float = 0.25,
    iou_thresh: float = 0.5,
    model_name: str = "yolov8m.pt",
    yolo_model: Optional["YOLO"] = None,
    save_video: bool = True,
    play_video_flag: bool = False,
) -> None:
    seq_name = seq_dir.name
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    f8_out = out_root.parent / "feature_8" / seq_name
    jersey_crops = f8_out / "jersey_crops"

    # ── output folders ──────────────────────────────────────────────
    seq_out = out_root / seq_name
    overlays_out = seq_out / "overlays"
    overlays_out.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        print(f"[{seq_name}] no img1/ — skipped")
        return
    if not gt_path.is_file():
        print(f"[{seq_name}] no gt/gt.txt — skipped")
        return

    print(f"\n[{seq_name}] ── Feature 9 ──────────────────────────")

    # ── S3: jersey colours from Feature 8 crops ─────────────────────
    if jersey_crops.is_dir():
        print(f"[{seq_name}] S3: naming jersey colours from {jersey_crops}")
        colours = name_jersey_colours(jersey_crops)
    else:
        print(
            f"[{seq_name}] S3: no jersey_crops/ — run feature_8 first; "
            f"colours set to unknown"
        )
        colours = {
            "home_colour": "unknown",
            "away_colour": "unknown",
            "gk_colour": "unknown",
        }

    print(
        f"[{seq_name}] colours → "
        f"home:{colours['home_colour']}  "
        f"away:{colours['away_colour']}  "
        f"gk:{colours['gk_colour']}"
    )

    # ── S4-B: clash detection ────────────────────────────────────────
    clash = check_colour_clash(colours["home_colour"], colours["away_colour"])

    # ── Collect frame paths and GT ───────────────────────────────────
    all_paths = _collect_frame_paths(str(img_dir))
    all_indices = sorted(
        _frame_index_of(p) for p in all_paths if _frame_index_of(p) >= 0
    )
    frame_indices = all_indices[:first_n]

    if not frame_indices:
        print(f"[{seq_name}] no frames — skipped")
        return

    frame_gt = _load_all_gt(str(gt_path))

    # ── Re-run Feature 8 assign_teams to get stable labels ──────────
    cfg = FeatureConfig()
    anchor_feats: List[np.ndarray] = []
    anchored_centroids = None
    original_anchor = None
    raw_labels: Dict[int, Dict[int, str]] = {}
    frame_entries: Dict[int, List] = {}

    for fi in frame_indices:
        fp = _frame_path_for_idx(all_paths, fi)
        if fp is None:
            continue
        fr = cv2.imread(fp)
        if fr is None:
            continue
        entries = frame_gt.get(fi, [])
        if not entries:
            continue

        if len(anchor_feats) < cfg.anchor_frames * 6:
            for _, bbox in entries:
                feat = _describe(crop_jersey(fr, bbox, cfg), cfg)
                if feat is not None:
                    anchor_feats.append(feat)
            if (
                anchored_centroids is None
                and len(anchor_feats) >= cfg.min_tracks_for_kmeans
            ):
                _, anchored_centroids, _ = _kmeans_robust(np.stack(anchor_feats), cfg)
                original_anchor = anchored_centroids.copy()

        lbl_dict, anchored_centroids = assign_teams(
            fr,
            entries,
            cfg,
            anchored_centroids,
            anchor_centroids=original_anchor,
        )
        raw_labels[fi] = lbl_dict
        frame_entries[fi] = entries

    stable_labels, voted, _ = apply_majority_vote(raw_labels)

    # ── S4-A: GK identification ──────────────────────────────────────
    gk_ids = identify_goalkeepers(stable_labels)
    print(f"[{seq_name}] GK → home:#{gk_ids['home_gk']}  away:#{gk_ids['away_gk']}")

    # ── Load YOLO for referee detection ─────────────────────────────
    if yolo_model is None:
        print(f"[{seq_name}] loading YOLO ({model_name}) …")
        yolo_model = YOLO(model_name)

    # ── Setup video writer ──────────────────────────────────────────
    video_writer = None
    video_path = seq_out / "output_video.mp4"
    
    if save_video:
        first_fi = frame_indices[0]
        first_fp = _frame_path_for_idx(all_paths, first_fi)
        if first_fp:
            first_frame = cv2.imread(first_fp)
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    _VIDEO_FPS,
                    (w, h + _HEADER_H)
                )

    # ── Per-frame rendering ──────────────────────────────────────────
    referee_count = 0
    frames_saved = 0
    
    for fi, entries in sorted(frame_entries.items()):
        fp = _frame_path_for_idx(all_paths, fi)
        if fp is None:
            continue
        frame = cv2.imread(fp)
        if frame is None:
            continue

        labels = stable_labels[fi]
        gt_boxes = [bbox for _, bbox in entries]

        # S4-C: referee
        referee_box = detect_referee_in_frame(
            frame, gt_boxes, yolo_model, iou_thresh, yolo_conf
        )
        if referee_box is not None:
            referee_count += 1

        # Draw player boxes
        annotated = _draw_boxes(frame, entries, labels, gk_ids, referee_box)

        # Prepend header bar
        annotated = _draw_header(annotated, fi, colours, gk_ids, referee_box, clash)

        # Save frame image
        out_path = overlays_out / f"frame_{fi:06d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        frames_saved += 1

        # Write to video
        if video_writer is not None:
            video_writer.write(annotated)

        if frames_saved % 50 == 0:
            print(f"[{seq_name}] processed {frames_saved} frames...")

    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"[{seq_name}] video saved → {video_path}")
        
        # Play video if requested
        if play_video_flag:
            print(f"[{seq_name}] playing video...")
            play_video(video_path, _VIDEO_FPS)

    print(
        f"[{seq_name}] referee detected in {referee_count}/{len(frame_entries)} frames"
    )

    # ── Write colours summary file ───────────────────────────────────
    summary_path = seq_out / "colours.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Sequence : {seq_name}\n")
        f.write(f"home     : {colours['home_colour']}\n")
        f.write(f"away     : {colours['away_colour']}\n")
        f.write(f"gk       : {colours['gk_colour']}\n")
        f.write(f"clash    : {'YES' if clash else 'no'}\n")
        f.write(f"home_gk  : #{gk_ids['home_gk']}\n")
        f.write(f"away_gk  : #{gk_ids['away_gk']}\n")
        f.write(f"referee_frames : {referee_count}/{len(frame_entries)}\n")
    print(f"[{seq_name}] colours.txt → {summary_path}")
    print(f"[{seq_name}] overlays   → {overlays_out}")


# ══════════════════════════════════════════════════════════════════════
# Top-level runner
# ══════════════════════════════════════════════════════════════════════


def run_feature9(
    seq_filter: Optional[str] = None,
    first_n: int = 50,
    yolo_conf: float = 0.25,
    iou_thresh: float = 0.5,
    model_name: str = "yolov8m.pt",
    save_video: bool = True,
    play_video_flag: bool = False,
) -> None:
    here = Path(__file__).parent
    videos_root = _DATASET_ROOT
    out_root = here / "output" / "feature_9"

    if not videos_root.is_dir():
        raise FileNotFoundError(
            f"videos root not found at {videos_root}\n"
            "Check config.py or update _DATASET_ROOT in this file."
        )

    seq_dirs = sorted(d for d in videos_root.iterdir() if d.is_dir())
    if seq_filter:
        seq_dirs = [d for d in seq_dirs if d.name == seq_filter]
        if not seq_dirs:
            raise FileNotFoundError(
                f"Sequence '{seq_filter}' not found under {videos_root}"
            )

    print(
        f"Feature 9 — processing {len(seq_dirs)} sequence(s), first {first_n} frames each"
    )
    if save_video:
        print("  Video generation: ENABLED")
    if play_video_flag:
        print("  Video playback: ENABLED (after processing)")

    # Load YOLO once and reuse across sequences
    print(f"Loading YOLO model: {model_name}")
    yolo_model = YOLO(model_name)

    for seq_dir in seq_dirs:
        run_feature9_sequence(
            seq_dir=seq_dir,
            out_root=out_root,
            first_n=first_n,
            yolo_conf=yolo_conf,
            iou_thresh=iou_thresh,
            model_name=model_name,
            yolo_model=yolo_model,
            save_video=save_video,
            play_video_flag=play_video_flag,
        )

    print("\nFeature 9 complete.")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature 9: jersey colour naming + GK/referee detection overlays"
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        help="Run only this sequence (folder name inside videos/)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of leading frames to process (default: 50)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for GT-YOLO matching (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="YOLO model weights (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the generated video after processing",
    )
    
    args = parser.parse_args()

    run_feature9(
        seq_filter=args.seq,
        first_n=args.frames,
        yolo_conf=args.conf,
        iou_thresh=args.iou,
        model_name=args.model,
        save_video=not args.no_video,
        play_video_flag=args.play,
    )