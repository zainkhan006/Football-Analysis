"""
Visual sanity-check viewer for a saved homography.

Loads `homographies/<seq>.npz`, projects the canonical pitch model onto
the source frame, and shows you the result.  If the lines line up with
the actual pitch, the homography is good.  If they're skewed, it's bad.

Drawing legend:
    SOLID lines  pitch features inside the inlier-supported region
                 (where H was fit -- trustworthy)
    DASHED lines pitch features in the EXTRAPOLATED region
                 (camera never saw landmarks here, so H is just guessing)
    GREEN  pitch outline (touchlines + goal lines)
    CYAN   penalty boxes + goal boxes + halfway line
    YELLOW centre circle (sampled at 36 points and projected)
    BLUE dots projected canonical keypoints (where pitch_config says they
                are, mapped through H — should overlap actual landmarks)

Hotkeys:
    Q / ESC   quit
    N / SPACE next frame in the clip (re-projects the same H -- shows
              how badly H drifts as the camera pans)
    P         previous frame
    J         jump to the calibration frame H was fitted on
    +/-       step through clips (cycles through homographies/*.npz)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from homography import PitchHomography
from pitch_config import DEFAULT_PITCH


HOMOGRAPHY_DIR     = Path("homographies")
DATASET_TRAIN_DIR  = Path("dataset/train")
WINDOW_NAME        = "Homography sanity check"


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def _safe_pt(p: Tuple[float, float], shape) -> Tuple[int, int] | None:
    """Clamp + reject points that are wildly outside the frame."""
    H, W = shape[:2]
    x, y = p
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if x < -W or x > 2 * W or y < -H or y > 2 * H:
        return None
    return int(round(x)), int(round(y))


def _segment_in_support(H: PitchHomography, w0, w1) -> bool:
    """Both endpoints inside the inlier world bbox (with small margin)?"""
    if H.inlier_world_bbox is None:
        return True
    return H.is_within_support(*w0, margin=2.0) \
       and H.is_within_support(*w1, margin=2.0)


def _line_through_frame(frame: np.ndarray, H: PitchHomography,
                        w0, w1, colour, thickness=2) -> None:
    """
    Draw a world-space line through H. Solid line if both endpoints are
    inside the inlier-supported region; dashed if either is extrapolated.
    """
    p0 = _safe_pt(H.world_to_pixel(*w0), frame.shape)
    p1 = _safe_pt(H.world_to_pixel(*w1), frame.shape)
    if p0 is None or p1 is None:
        return

    if _segment_in_support(H, w0, w1):
        cv2.line(frame, p0, p1, colour, thickness, cv2.LINE_AA)
    else:
        # Dashed: chunks of 8px on, 6px off.
        x0, y0 = p0
        x1, y1 = p1
        L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if L < 1:
            return
        n = max(2, int(L / 14))
        for i in range(n):
            if i % 2 == 1:
                continue
            t0 = i / n
            t1 = (i + 1) / n
            a = (int(round(x0 + t0 * (x1 - x0))),
                 int(round(y0 + t0 * (y1 - y0))))
            b = (int(round(x0 + t1 * (x1 - x0))),
                 int(round(y0 + t1 * (y1 - y0))))
            cv2.line(frame, a, b, colour, max(1, thickness - 1), cv2.LINE_AA)


def draw_pitch(frame: np.ndarray, H: PitchHomography) -> None:
    """Project the canonical pitch model and draw it onto `frame`."""
    cfg = DEFAULT_PITCH
    L, W = cfg.length, cfg.width
    pl, pw = cfg.penalty_box_length, cfg.penalty_box_width
    gl, gw = cfg.goal_box_length, cfg.goal_box_width
    r      = cfg.centre_circle_radius

    GREEN  = (0, 220,  60)
    CYAN   = (255, 220,  0)
    YELLOW = (0, 240, 240)

    # Outline (touchlines + goal lines)
    outline = [
        (0, 0), (L, 0), (L, W), (0, W), (0, 0),
    ]
    for a, b in zip(outline[:-1], outline[1:]):
        _line_through_frame(frame, H, a, b, GREEN, 2)

    # Halfway line
    _line_through_frame(frame, H, (L / 2, 0), (L / 2, W), CYAN, 1)

    # Left penalty box
    box_l = [
        (0,  (W - pw) / 2),
        (pl, (W - pw) / 2),
        (pl, (W + pw) / 2),
        (0,  (W + pw) / 2),
    ]
    for a, b in zip(box_l, box_l[1:]):
        _line_through_frame(frame, H, a, b, CYAN, 1)

    # Right penalty box
    box_r = [
        (L,      (W - pw) / 2),
        (L - pl, (W - pw) / 2),
        (L - pl, (W + pw) / 2),
        (L,      (W + pw) / 2),
    ]
    for a, b in zip(box_r, box_r[1:]):
        _line_through_frame(frame, H, a, b, CYAN, 1)

    # Goal boxes
    gbl = [(0, (W - gw)/2), (gl, (W - gw)/2),
           (gl, (W + gw)/2), (0, (W + gw)/2)]
    gbr = [(L, (W - gw)/2), (L - gl, (W - gw)/2),
           (L - gl, (W + gw)/2), (L, (W + gw)/2)]
    for boxx in (gbl, gbr):
        for a, b in zip(boxx, boxx[1:]):
            _line_through_frame(frame, H, a, b, CYAN, 1)

    # Centre circle (36 samples projected and stroked, dashed if extrapolated)
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=True)
    pts_world = np.stack(
        [L / 2 + r * np.cos(angles), W / 2 + r * np.sin(angles)], axis=1
    )
    pts_px = H.world_to_pixel_batch(pts_world)
    for i in range(len(pts_px) - 1):
        w0 = (float(pts_world[i, 0]),     float(pts_world[i, 1]))
        w1 = (float(pts_world[i + 1, 0]), float(pts_world[i + 1, 1]))
        if not _segment_in_support(H, w0, w1):
            # Extrapolated -- draw only every other sample to mimic dashed
            if i % 2 == 1:
                continue
        a = _safe_pt(tuple(pts_px[i]), frame.shape)
        b = _safe_pt(tuple(pts_px[i + 1]), frame.shape)
        if a and b:
            cv2.line(frame, a, b, YELLOW, 1, cv2.LINE_AA)

    # Centre spot
    cs = _safe_pt(H.world_to_pixel(L / 2, W / 2), frame.shape)
    if cs:
        cv2.circle(frame, cs, 3, YELLOW, -1)

    # Inlier-supported world-bbox outline (orange dashes) so the user can
    # SEE which world region H is actually backed by evidence.
    if H.inlier_world_bbox is not None:
        ORANGE = (0, 165, 255)
        x0, y0, x1, y1 = H.inlier_world_bbox
        # Pad slightly so the rectangle isn't right on top of the keypoints.
        pad = 1.0
        corners = [(x0 - pad, y0 - pad), (x1 + pad, y0 - pad),
                   (x1 + pad, y1 + pad), (x0 - pad, y1 + pad),
                   (x0 - pad, y0 - pad)]
        for a, b in zip(corners[:-1], corners[1:]):
            p0 = _safe_pt(H.world_to_pixel(*a), frame.shape)
            p1 = _safe_pt(H.world_to_pixel(*b), frame.shape)
            if p0 and p1:
                # Dashed orange so it doesn't blend with the green outline
                xa, ya = p0; xb, yb = p1
                Lpx = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
                if Lpx < 1:
                    continue
                n = max(2, int(Lpx / 12))
                for i in range(n):
                    if i % 2 == 1:
                        continue
                    t0 = i / n; t1 = (i + 1) / n
                    pa = (int(round(xa + t0*(xb-xa))), int(round(ya + t0*(yb-ya))))
                    pb = (int(round(xa + t1*(xb-xa))), int(round(ya + t1*(yb-ya))))
                    cv2.line(frame, pa, pb, ORANGE, 1, cv2.LINE_AA)


def draw_canonical_keypoints(frame: np.ndarray, H: PitchHomography) -> None:
    """Draw all 32 canonical keypoints projected through H."""
    BLUE = (255, 100, 0)
    pts_world = np.array(DEFAULT_PITCH.vertices_m, dtype=np.float64)
    pts_px    = H.world_to_pixel_batch(pts_world)
    for i, (px, py) in enumerate(pts_px):
        p = _safe_pt((px, py), frame.shape)
        if p:
            cv2.circle(frame, p, 4, BLUE, -1)
            cv2.putText(frame, str(i + 1), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    seq_name: str,
    frame_id: int,
    calib_frame_id: int | None,
    H: PitchHomography,
) -> None:
    Hh, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    err_s = (f"{H.fit_error_px:.2f}px"
             if H.fit_error_px is not None else "?")
    n_s   = H.n_correspondences if H.n_correspondences is not None else "?"

    if H.inlier_world_bbox is not None:
        x0, y0, x1, y1 = H.inlier_world_bbox
        bbox_s = (f"inliers x={x0:.0f}-{x1:.0f}m  "
                  f"y={y0:.0f}-{y1:.0f}m  "
                  f"({(x1-x0)/120*100:.0f}% of length)")
    else:
        bbox_s = "inliers: ?"

    cv2.putText(frame, f"Seq: {seq_name}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame {frame_id}    "
                       f"calib={calib_frame_id}    "
                       f"reproj_err={err_s}   N={n_s}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(frame, bbox_s, (10, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    cv2.putText(frame, "Q quit  N/SPACE next  P prev  "
                       "J jump-to-calib  +/- next/prev clip   "
                       "(orange box = where H is supported)",
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1)


# ─── Main loop ───────────────────────────────────────────────────────────────

def list_homographies(homo_dir: Path) -> List[Path]:
    """Calibration-H files only — exclude the per-frame _track.npz files."""
    return sorted(p for p in homo_dir.glob("*.npz")
                  if not p.stem.endswith("_track"))


def _load_track(homo_dir: Path, seq_name: str):
    """Return (H_per_frame, frame_id_to_index, costs, reseeded) or None."""
    track_path = homo_dir / f"{seq_name}_track.npz"
    if not track_path.exists():
        return None
    data = np.load(track_path)
    fid_to_idx = {int(fid): i for i, fid in enumerate(data["frame_ids"])}
    return data["H_per_frame"], fid_to_idx, data["costs"], data["reseeded"]


def show_clip(homo_path: Path) -> str:
    """
    Display the homography projection on the calibration frame, plus
    let the user step through other frames to see how it drifts.

    Returns one of:
        "next_clip", "prev_clip", "quit"
    """
    seq_name  = homo_path.stem
    seq_dir   = DATASET_TRAIN_DIR / seq_name
    if not seq_dir.exists():
        print(f"  [warn] no dataset folder for {seq_name}, skipping")
        return "next_clip"

    H_calib = PitchHomography.load(homo_path)
    track   = _load_track(HOMOGRAPHY_DIR, seq_name)

    frames = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frames:
        print(f"  [warn] no images in {seq_dir}, skipping")
        return "next_clip"

    calib_id = H_calib.source_frame_id or int(frames[0].stem)
    fids = [int(f.stem) for f in frames]
    try:
        idx = fids.index(calib_id)
    except ValueError:
        idx = 0

    print(f"\n[viewer] {seq_name}  "
          f"calib_frame={calib_id}  "
          f"reproj_err={H_calib.fit_error_px:.2f}px  "
          f"N={H_calib.n_correspondences}  "
          f"per-frame-track={'YES' if track is not None else 'no'}")
    print("[viewer]  Q quit · SPACE/N next frame · P prev · J jump-to-calib"
          " · +/- next/prev clip")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    while True:
        idx = max(0, min(idx, len(frames) - 1))
        fp  = frames[idx]
        img = cv2.imread(str(fp))
        if img is None:
            break

        # Use the per-frame H if a tracking file is available, else the
        # single calibration H. We rebuild a PitchHomography for this
        # frame so the existing draw_* routines work unchanged.
        H = H_calib
        track_cost = None
        track_reseed = False
        if track is not None:
            H_per_frame, fid_to_idx, costs, reseeded = track
            j = fid_to_idx.get(int(fp.stem))
            if j is not None:
                H_mat = H_per_frame[j]
                try:
                    H_inv = np.linalg.inv(H_mat)
                except np.linalg.LinAlgError:
                    H_inv = H_calib.H_pixel_to_world
                H = PitchHomography(
                    H_world_to_pixel  = H_mat,
                    H_pixel_to_world  = H_inv,
                    fit_inliers       = H_calib.fit_inliers,
                    fit_error_px      = H_calib.fit_error_px,
                    n_correspondences = H_calib.n_correspondences,
                    source_frame_id   = H_calib.source_frame_id,
                    inlier_world_bbox = H_calib.inlier_world_bbox,
                )
                track_cost   = float(costs[j])
                track_reseed = bool(reseeded[j])

        draw_pitch(img, H)
        draw_canonical_keypoints(img, H)
        draw_hud(img, seq_name, int(fp.stem), calib_id, H)
        if track_cost is not None:
            tag = f"track_cost={track_cost:.1f}px"
            if track_reseed:
                tag += "  RESEED"
            cv2.putText(img, tag, (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            return "quit"
        elif key in (ord('n'), ord(' '), 83):
            idx += 1
        elif key in (ord('p'), 81):
            idx -= 1
        elif key == ord('j'):
            try:
                idx = fids.index(calib_id)
            except ValueError:
                idx = 0
        elif key in (ord('='), ord('+')):
            return "next_clip"
        elif key in (ord('-'), ord('_')):
            return "prev_clip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None,
                    help="Show only this clip; otherwise iterate through all "
                         "homographies/*.npz")
    args = ap.parse_args()

    homos = list_homographies(HOMOGRAPHY_DIR)
    if not homos:
        raise SystemExit(f"No .npz files in {HOMOGRAPHY_DIR}. Run "
                         f"compute_homographies.py first.")

    if args.seq:
        wanted = HOMOGRAPHY_DIR / f"{args.seq}.npz"
        if not wanted.exists():
            raise SystemExit(f"{wanted} not found")
        homos = [wanted]

    i = 0
    while 0 <= i < len(homos):
        action = show_clip(homos[i])
        if action == "quit":
            break
        elif action == "next_clip":
            i += 1
        elif action == "prev_clip":
            i = max(0, i - 1)
        else:
            i += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
