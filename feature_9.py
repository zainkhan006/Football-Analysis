"""
Feature 9 S3 — Jersey Colour Naming
=====================================
Reads jersey crops saved by Feature 8.
Takes the center pixel of each crop, removes outliers, averages, names colour.

Run after feature_8.py:
    python feature_9.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Colour name lookup  (OpenCV HSV: H∈[0,179], S∈[0,255], V∈[0,255])
# ══════════════════════════════════════════════════════════════════════

def _hsv_to_name(h: float, s: float, v: float) -> str:
    # Achromatic first
    if v < 60:
        return "black"
    if s < 50:
        if v > 170:
            return "white"
        return "grey"
    # Chromatic by hue
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


# ══════════════════════════════════════════════════════════════════════
# S3 — Main logic
# ══════════════════════════════════════════════════════════════════════

def _get_colour_for_team(folder: Path) -> str:
    """
    For every crop image in folder:
        1. Read center pixel
        2. Remove outliers (> 2 std from mean in HSV)
        3. Average remaining pixels
        4. Convert to colour name
    """
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]

    if not paths:
        return "unknown"

    # Collect center pixels in BGR
    center_pixels = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 4 or w < 4:
            continue
        cx, cy = w // 2, h // 2
        center_pixels.append(img[cy, cx].astype(np.float32))  # BGR

    if len(center_pixels) < 5:
        return "unknown"

    pixels = np.array(center_pixels)  # (N, 3) BGR float32

    # Remove outliers — pixels whose distance from mean > 2 std
    mean = pixels.mean(axis=0)
    std  = pixels.std(axis=0).mean() + 1e-6   # scalar std across all channels
    dists = np.linalg.norm(pixels - mean, axis=1)
    inliers = pixels[dists < 2 * std]

    if len(inliers) < 3:
        inliers = pixels   # fallback if too aggressive

    # Average inlier pixels and convert to HSV colour name
    avg_bgr = inliers.mean(axis=0).astype(np.uint8).reshape(1, 1, 3)
    avg_hsv = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(avg_hsv[0]), int(avg_hsv[1]), int(avg_hsv[2])

    colour = _hsv_to_name(h, s, v)
    print(f"    {len(paths)} crops | {len(inliers)} inliers | "
          f"HSV=({h},{s},{v}) → {colour}")
    return colour


def name_jersey_colours(jersey_crops_dir: Path) -> Dict[str, str]:
    result = {}
    for team in ("home", "away", "gk"):
        folder = jersey_crops_dir / team
        if not folder.is_dir():
            result[f"{team}_colour"] = "unknown"
            continue
        print(f"  [{team}]")
        result[f"{team}_colour"] = _get_colour_for_team(folder)
    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    here    = Path(__file__).parent
    f8_root = here / "output" / "feature_8"

    for seq_dir in sorted(f8_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        jersey_crops_dir = seq_dir / "jersey_crops"
        if not jersey_crops_dir.is_dir():
            print(f"[{seq_dir.name}] no jersey_crops/ — run feature_8 first")
            continue

        print(f"\n[{seq_dir.name}]")
        colours = name_jersey_colours(jersey_crops_dir)
        print(f"  home : {colours.get('home_colour', '?')}")
        print(f"  away : {colours.get('away_colour', '?')}")
        print(f"  gk   : {colours.get('gk_colour',   '?')}")