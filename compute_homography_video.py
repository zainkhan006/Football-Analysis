"""
compute_homography_video.py

Computes a pitch homography from a .mp4 (or any OpenCV-readable video)
instead of a SportsMOT folder structure.

Steps:
  1. Sample N frames evenly from the video (or from a time window).
  2. Run the Roboflow pitch keypoint detector on each frame.
  3. Pick the frame with the best world-space keypoint spread.
  4. Fit H with RANSAC.
  5. Optionally refine with line-DT.
  6. Save to homographies/<output_name>.npz

Usage examples:
    # Basic — sample 16 frames from the whole clip
    python compute_homography_video.py --video staticCam/napoli_roma.mp4

    # Restrict to a specific time window (seconds)
    python compute_homography_video.py --video staticCam/napoli_roma.mp4 --start 2280 --end 2640

    # Override output name
    python compute_homography_video.py --video staticCam/napoli_roma.mp4 --name napoli_roma_38_44

    # Skip line refinement
    python compute_homography_video.py --video staticCam/napoli_roma.mp4 --no-refine
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from homography import PitchHomography
from pitch_config import DEFAULT_PITCH
from pitch_keypoint_detector import PitchKeypointDetector, PitchKeypointResult
from refine_homography import build_line_mask, refine_homography, sample_canonical_pitch

HOMOGRAPHY_OUT_DIR = Path("homographies")
THIRD_BOUNDARIES = (40.0, 80.0)


# ─── Scoring (same logic as compute_homographies.py) ─────────────────────────

def _spreadScore(worldXys: np.ndarray) -> float:
    if(len(worldXys) == 0):
        return -1.0
    xs = worldXys[:, 0]
    nLeft  = (xs <  THIRD_BOUNDARIES[0]).any()
    nMid   = ((xs >= THIRD_BOUNDARIES[0]) & (xs <= THIRD_BOUNDARIES[1])).any()
    nRight = (xs >  THIRD_BOUNDARIES[1]).any()
    thirds = int(nLeft) + int(nMid) + int(nRight)
    xSpan  = float(xs.max() - xs.min())
    return thirds * 1000.0 + xSpan + 0.5 * len(worldXys)


# ─── Frame sampling from video ────────────────────────────────────────────────

def sampleFrames(
    videoPath: Path,
    numCandidates: int,
    startSec: Optional[float] = None,
    endSec: Optional[float] = None,
) -> List[Tuple[int, np.ndarray]]:
    """
    Sample up to numCandidates frames evenly from a video file.
    Optionally restrict to [startSec, endSec] window.
    Returns list of (frameIndex, bgrFrame) tuples.
    """
    cap = cv2.VideoCapture(str(videoPath))
    if(not cap.isOpened()):
        raise IOError(f"could not open video at {videoPath}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = totalFrames / fps if fps > 0 else 0

    startFrame = int(startSec * fps) if startSec is not None else 0
    endFrame   = int(endSec   * fps) if endSec   is not None else totalFrames - 1

    startFrame = max(0, min(startFrame, totalFrames - 1))
    endFrame   = max(startFrame, min(endFrame, totalFrames - 1))

    print(f"video is {duration:.1f}s at {fps:.1f}fps, {totalFrames} frames total")
    print(f"sampling window: frames {startFrame} to {endFrame}")

    indices = np.linspace(startFrame, endFrame, numCandidates).astype(int)
    indices = np.unique(indices)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if(ok and frame is not None):
            frames.append((int(idx), frame))
    cap.release()

    print(f"successfully read {len(frames)} candidate frames")
    return frames


# ─── Best frame picker ────────────────────────────────────────────────────────

def findBestFrame(
    frames: List[Tuple[int, np.ndarray]],
    detector: PitchKeypointDetector,
    confThreshold: float,
) -> Tuple[Optional[int], Optional[np.ndarray], Optional[PitchKeypointResult]]:
    """
    Run detector on each sampled frame, return the one with best keypoint spread.
    Returns (frameIndex, bgrFrame, keypointResult) or (None, None, None).
    """
    pitchXy = np.array(DEFAULT_PITCH.vertices_m)

    bestIdx   = None
    bestFrame = None
    bestRes   = None
    bestScore = -1.0

    print(f"\n  {'frame':>6} {'n_kp':>4} {'thirds':>6} {'x_min':>5} {'x_max':>5} {'score':>8}")

    for frameIdx, bgr in frames:
        res       = detector.detect(bgr, conf_threshold=confThreshold)
        confident = res.confident(confThreshold)

        if(not confident):
            print(f"  {frameIdx:>6} {0:>4}")
            continue

        world = pitchXy[[kp.index for kp in confident]]
        score = _spreadScore(world)
        xs    = world[:, 0]
        thirds = (
            int((xs <  THIRD_BOUNDARIES[0]).any())
          + int(((xs >= THIRD_BOUNDARIES[0]) & (xs <= THIRD_BOUNDARIES[1])).any())
          + int((xs >  THIRD_BOUNDARIES[1]).any())
        )
        print(f"  {frameIdx:>6} {len(confident):>4} {thirds:>6} "
              f"{xs.min():>5.1f} {xs.max():>5.1f} {score:>8.1f}")

        if(score > bestScore):
            bestScore = score
            bestIdx   = frameIdx
            bestFrame = bgr
            bestRes   = res

    return bestIdx, bestFrame, bestRes


# ─── Main driver ─────────────────────────────────────────────────────────────

def computeHomographyVideo(
    videoPath: Path,
    outputName: str,
    numCandidates: int          = 16,
    confThreshold: float        = 0.5,
    minKeypoints:  int          = 4,
    ransacThresh:  float        = 8.0,
    refine: bool                = True,
    refineMaxPx:   float        = 30.0,
    startSec: Optional[float]  = None,
    endSec:   Optional[float]  = None,
    outDir: Path                = HOMOGRAPHY_OUT_DIR,
) -> Optional[PitchHomography]:

    print(f"\n[homo] video: {videoPath.name}")

    frames = sampleFrames(videoPath, numCandidates, startSec, endSec)
    if(not frames):
        print("[fail] no frames could be read from video")
        return None

    detector = PitchKeypointDetector(verbose=True)

    print(f"\nevaluating {len(frames)} candidate frames...")
    bestIdx, bestBgr, bestRes = findBestFrame(frames, detector, confThreshold)

    if(bestRes is None):
        print("[fail] no confident keypoints found in any sampled frame")
        return None

    confident = bestRes.confident(confThreshold)
    print(f"\nbest frame: index {bestIdx}  ({len(confident)} confident keypoints)")

    if(len(confident) < minKeypoints):
        print(f"[fail] only {len(confident)} keypoints, need >= {minKeypoints}")
        return None

    pitchXy  = DEFAULT_PITCH.vertices_m
    worldPts: List[Tuple[float, float]] = []
    pixelPts: List[Tuple[float, float]] = []
    for kp in confident:
        wx, wy = pitchXy[kp.index]
        worldPts.append((wx, wy))
        pixelPts.append(kp.pixel_xy)

    try:
        H = PitchHomography.from_correspondences(
            world_xy_m          = worldPts,
            pixel_xy            = pixelPts,
            ransac_threshold_px = ransacThresh,
            source_frame_id     = bestIdx,
        )
    except ValueError as e:
        print(f"[fail] homography fit failed: {e}")
        return None

    nIn = int(H.fit_inliers.sum()) if H.fit_inliers is not None else "?"
    print(f"[seed] RANSAC inliers={nIn}/{len(confident)}  reproj_err={H.fit_error_px:.2f}px")

    if(refine):
        mask    = build_line_mask(bestBgr)
        samples = sample_canonical_pitch(step_m=1.0)
        try:
            hRefMat, info = refine_homography(
                H.H_world_to_pixel, mask, samples,
                max_px       = refineMaxPx,
                max_iter     = 30,
                prior_weight = 5.0,
            )
        except Exception as e:
            print(f"[warn] line refinement failed: {e}, keeping seed H")
        else:
            hRefInv = np.linalg.inv(hRefMat)
            H = PitchHomography(
                H_world_to_pixel  = hRefMat.astype(np.float64),
                H_pixel_to_world  = hRefInv.astype(np.float64),
                fit_inliers       = H.fit_inliers,
                fit_error_px      = H.fit_error_px,
                n_correspondences = H.n_correspondences,
                source_frame_id   = H.source_frame_id,
                inlier_world_bbox = H.inlier_world_bbox,
            )
            print(f"[refine] line-DT cost {info.cost_before:.2f}px -> {info.cost_after:.2f}px  "
                  f"(n_active={info.n_active}/{len(samples)}  iters={info.iterations})")

    outDir.mkdir(parents=True, exist_ok=True)
    outPath = outDir / f"{outputName}.npz"
    H.save(outPath)
    print(f"[ok] saved -> {outPath}")
    return H


# ─── CLI ─────────────────────────────────────────────────────────────────────

if(__name__ == "__main__"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",          required=True,
                    help="Path to the video file (.mp4 or any OpenCV-readable format)")
    ap.add_argument("--name",           default=None,
                    help="Output name for the .npz file (default: video filename stem)")
    ap.add_argument("--start",          type=float, default=None,
                    help="Start of sampling window in seconds")
    ap.add_argument("--end",            type=float, default=None,
                    help="End of sampling window in seconds")
    ap.add_argument("--num-candidates", type=int,   default=16,
                    help="Number of frames to sample from the window (default 16)")
    ap.add_argument("--conf",           type=float, default=0.5,
                    help="Keypoint confidence threshold (default 0.5)")
    ap.add_argument("--min-kp",         type=int,   default=4,
                    help="Minimum keypoints needed to fit H (default 4)")
    ap.add_argument("--ransac-thresh",  type=float, default=8.0,
                    help="RANSAC reprojection threshold in pixels (default 8.0)")
    ap.add_argument("--no-refine",      action="store_true",
                    help="Skip line-DT refinement step")
    ap.add_argument("--refine-max-px",  type=float, default=30.0,
                    help="Huber cap for refinement cost in pixels (default 30.0)")
    ap.add_argument("--out-dir",        default=str(HOMOGRAPHY_OUT_DIR),
                    help="Directory to write the .npz file (default: homographies/)")
    args = ap.parse_args()

    videoPath  = Path(args.video)
    outputName = args.name if args.name else videoPath.stem

    t0 = time.perf_counter()
    H = computeHomographyVideo(
        videoPath     = videoPath,
        outputName    = outputName,
        numCandidates = args.num_candidates,
        confThreshold = args.conf,
        minKeypoints  = args.min_kp,
        ransacThresh  = args.ransac_thresh,
        refine        = not args.no_refine,
        refineMaxPx   = args.refine_max_px,
        startSec      = args.start,
        endSec        = args.end,
        outDir        = Path(args.out_dir),
    )
    dt = time.perf_counter() - t0
    if(H is not None):
        print(f"\ndone in {dt:.1f}s  {H}")
    else:
        print(f"\nfailed after {dt:.1f}s")