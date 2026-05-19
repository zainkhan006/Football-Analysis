"""
view_homography_video.py

Visual sanity check for a homography computed from a video file.
Projects the canonical pitch model onto frames from the video.

If the coloured lines align with the real pitch lines on screen, H is good.
If they are skewed or offset, H is bad.

Usage:
    python view_homography_video.py --video staticCam/tactical_clipB.mp4 --homo homographies/napoli_roma.npz

Hotkeys:
    Q / ESC   quit
    N / SPACE next frame
    P         previous frame
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from homography import PitchHomography
from pitch_config import DEFAULT_PITCH

WINDOW_NAME = "Homography check on video"


# ─── Drawing (copied from view_homography.py) ─────────────────────────────────

def _safePt(p: Tuple[float, float], shape) -> Optional[Tuple[int, int]]:
    H, W = shape[:2]
    x, y = p
    if(not (np.isfinite(x) and np.isfinite(y))):
        return None
    if(x < -W or x > 2 * W or y < -H or y > 2 * H):
        return None
    return int(round(x)), int(round(y))


def _inSupport(H: PitchHomography, w0, w1) -> bool:
    if(H.inlier_world_bbox is None):
        return True
    return H.is_within_support(*w0, margin=2.0) and H.is_within_support(*w1, margin=2.0)


def _drawLine(frame, H: PitchHomography, w0, w1, colour, thickness=2):
    p0 = _safePt(H.world_to_pixel(*w0), frame.shape)
    p1 = _safePt(H.world_to_pixel(*w1), frame.shape)
    if(p0 is None or p1 is None):
        return
    if(_inSupport(H, w0, w1)):
        cv2.line(frame, p0, p1, colour, thickness, cv2.LINE_AA)
    else:
        x0, y0 = p0; x1, y1 = p1
        L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if(L < 1):
            return
        n = max(2, int(L / 14))
        for i in range(n):
            if(i % 2 == 1):
                continue
            t0 = i / n; t1 = (i + 1) / n
            a = (int(round(x0 + t0 * (x1 - x0))), int(round(y0 + t0 * (y1 - y0))))
            b = (int(round(x0 + t1 * (x1 - x0))), int(round(y0 + t1 * (y1 - y0))))
            cv2.line(frame, a, b, colour, max(1, thickness - 1), cv2.LINE_AA)


def drawPitch(frame, H: PitchHomography):
    cfg = DEFAULT_PITCH
    L, W = cfg.length, cfg.width
    pl, pw = cfg.penalty_box_length, cfg.penalty_box_width
    gl, gw = cfg.goal_box_length, cfg.goal_box_width
    r = cfg.centre_circle_radius

    GREEN  = (0, 220, 60)
    CYAN   = (255, 220, 0)
    YELLOW = (0, 240, 240)

    outline = [(0, 0), (L, 0), (L, W), (0, W), (0, 0)]
    for a, b in zip(outline[:-1], outline[1:]):
        _drawLine(frame, H, a, b, GREEN, 2)

    _drawLine(frame, H, (L / 2, 0), (L / 2, W), CYAN, 1)

    boxL = [(0, (W - pw) / 2), (pl, (W - pw) / 2), (pl, (W + pw) / 2), (0, (W + pw) / 2)]
    boxR = [(L, (W - pw) / 2), (L - pl, (W - pw) / 2), (L - pl, (W + pw) / 2), (L, (W + pw) / 2)]
    for box in (boxL, boxR):
        for a, b in zip(box[:-1], box[1:]):
            _drawLine(frame, H, a, b, CYAN, 1)

    gbL = [(0, (W - gw) / 2), (gl, (W - gw) / 2), (gl, (W + gw) / 2), (0, (W + gw) / 2)]
    gbR = [(L, (W - gw) / 2), (L - gl, (W - gw) / 2), (L - gl, (W + gw) / 2), (L, (W + gw) / 2)]
    for box in (gbL, gbR):
        for a, b in zip(box[:-1], box[1:]):
            _drawLine(frame, H, a, b, CYAN, 1)

    angles  = np.linspace(0, 2 * np.pi, 36, endpoint=True)
    ptsW    = np.stack([L / 2 + r * np.cos(angles), W / 2 + r * np.sin(angles)], axis=1)
    ptsPx   = H.world_to_pixel_batch(ptsW)
    for i in range(len(ptsPx) - 1):
        w0 = (float(ptsW[i, 0]), float(ptsW[i, 1]))
        w1 = (float(ptsW[i + 1, 0]), float(ptsW[i + 1, 1]))
        if(not _inSupport(H, w0, w1) and i % 2 == 1):
            continue
        a = _safePt(tuple(ptsPx[i]), frame.shape)
        b = _safePt(tuple(ptsPx[i + 1]), frame.shape)
        if(a and b):
            cv2.line(frame, a, b, YELLOW, 1, cv2.LINE_AA)

    cs = _safePt(H.world_to_pixel(L / 2, W / 2), frame.shape)
    if(cs):
        cv2.circle(frame, cs, 3, YELLOW, -1)

    ptsWorld = np.array(DEFAULT_PITCH.vertices_m, dtype=np.float64)
    ptsPx2   = H.world_to_pixel_batch(ptsWorld)
    for i, (px, py) in enumerate(ptsPx2):
        p = _safePt((px, py), frame.shape)
        if(p):
            cv2.circle(frame, p, 4, (255, 100, 0), -1)
            cv2.putText(frame, str(i + 1), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1, cv2.LINE_AA)


def drawHud(frame, frameIdx: int, totalFrames: int, H: PitchHomography, videoName: str):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    errS = f"{H.fit_error_px:.2f}px" if H.fit_error_px is not None else "?"
    cv2.putText(frame, f"{videoName}  frame {frameIdx}/{totalFrames}  reproj_err={errS}  N={H.n_correspondences}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Q/ESC quit   N/SPACE next   P prev",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to the video file")
    ap.add_argument("--homo",  required=True, help="Path to the .npz homography file")
    ap.add_argument("--step",  type=int, default=75,
                    help="Frame step size for N/P navigation (default 75 = 3s at 25fps)")
    args = ap.parse_args()

    videoPath = Path(args.video)
    homoPath  = Path(args.homo)

    H = PitchHomography.load(homoPath)
    print(f"loaded {H}")

    cap = cv2.VideoCapture(str(videoPath))
    if(not cap.isOpened()):
        raise SystemExit(f"could not open video at {videoPath}")

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    calibFrame  = H.source_frame_id if H.source_frame_id is not None else 0

    # source_frame_id came from whatever video the homography was computed on.
    # if it falls outside this video, warn the user that the homography and the
    # video may not match, then start from frame 0 instead.
    if(calibFrame < 0 or calibFrame >= totalFrames):
        print(f"warning: calibration frame {calibFrame} is outside this video "
              f"(0 to {totalFrames - 1}). the homography may have been computed "
              f"on a different clip. starting from frame 0")
        calibFrame = 0
    idx = calibFrame

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    while True:
        idx = max(0, min(idx, totalFrames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if(not ok or frame is None):
            print(f"could not read frame {idx}")
            # step back toward a readable frame, but if we are already at 0
            # there is nowhere left to go, so stop instead of looping forever
            if(idx <= 0):
                print("could not read frame 0, stopping")
                break
            idx = max(0, idx - args.step)
            continue

        drawPitch(frame, H)
        drawHud(frame, idx, totalFrames, H, videoPath.name)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(0) & 0xFF
        if(key in (ord('q'), 27)):
            break
        elif(key in (ord('n'), ord(' '), 83)):
            idx += args.step
        elif(key in (ord('p'), 81)):
            idx -= args.step

    cap.release()
    cv2.destroyAllWindows()


if(__name__ == "__main__"):
    main()