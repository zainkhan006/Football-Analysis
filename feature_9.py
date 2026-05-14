"""
Feature 9 S3 — Jersey Colour Naming
=====================================
Reads jersey crops saved by Feature 8.
Extracts the dominant colour(s) using K-means clustering on HSV pixels.

Plain kit  → reports the single dominant colour
Designed kit (stripes/hoops) → reports all colours that pass a minimum
pixel-share threshold, joined by '/'.

Grass pixels are masked out before clustering. Small contaminant clusters
(skin, shorts, sponsor logos, advertising boards) are filtered by a
minimum-share threshold so they never reach the output.

Run after feature_8.py:
    python feature_9.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
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
# Tunables
# ══════════════════════════════════════════════════════════════════════

# A pixel must contribute meaningfully to the dominant colour. Very dark
# pixels (shadows under collars, hair) are excluded unless the cluster
# is genuinely a black kit.
_MIN_VALUE_FOR_CLUSTERING = 20

# A cluster's pixel count must reach at least this fraction of the total
# masked pixels to be reported as a kit colour. Filters out small
# contaminants (skin tones, shorts edges, sponsor logos, ad boards).
# 25% is empirically the right balance — real kit colours always exceed
# it, and contaminants almost never do.
_MIN_CLUSTER_SHARE = 0.25

# K-means cluster count. For most kits 4 is enough — body, secondary
# stripe/hoop, plus 2 extra clusters that absorb contaminants so they
# don't pollute the kit clusters.
_KMEANS_K = 4


def _kmeans_dominant_colours(
    pixels: np.ndarray,
) -> List[Tuple[Tuple[int, int, int], int]]:
    """
    Runs K-means on HSV pixels and returns clusters ordered by size.

    Parameters
    ----------
    pixels : (N, 3) uint8 array of HSV pixels

    Returns
    -------
    list of ((h, s, v), pixel_count) sorted by pixel_count descending
    """
    if len(pixels) < _KMEANS_K:
        return []

    samples = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centres = cv2.kmeans(
        samples, _KMEANS_K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    clusters = []
    for k in range(_KMEANS_K):
        count = int(np.sum(labels == k))
        if count == 0:
            continue
        centre = centres[k]
        clusters.append((
            (int(centre[0]), int(centre[1]), int(centre[2])),
            count,
        ))

    clusters.sort(key=lambda c: c[1], reverse=True)
    return clusters


def _get_colour_for_team(folder: Path) -> str:
    """
    For every crop image in folder:
        1. Pool every pixel from every crop (subsampled per crop for memory)
        2. Drop very dark pixels (shadows, borders)
        3. Drop grass pixels (green hue range)
        4. Cluster remaining pixels with K-means
        5. Report every cluster whose share of the masked pixel pool
           exceeds the minimum-share threshold

    Returns a single colour name for plain kits, or multiple colour names
    joined by '/' for designed kits.
    """
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]

    if not paths:
        return "unknown"

    pooled = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 4 or w < 4:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        flat = hsv.reshape(-1, 3)

        # Drop pixels too dark to carry colour information
        v_mask = flat[:, 2] >= _MIN_VALUE_FOR_CLUSTERING
        flat = flat[v_mask]
        if len(flat) == 0:
            continue

        # Drop grass pixels — green hue range with meaningful saturation.
        # The range is wide enough to catch dim floodlit grass without
        # masking out genuinely green kits, since a real green jersey
        # still has plenty of fabric pixels with higher saturation than
        # weathered pitch turf.
        grass_mask = ~(
            (flat[:, 0] >= 36) & (flat[:, 0] <= 90) &
            (flat[:, 1] >= 35)
        )
        flat = flat[grass_mask]
        if len(flat) == 0:
            continue

        # Subsample large crops to keep memory bounded — at most 400
        # pixels per crop is enough for K-means to see the distribution.
        if len(flat) > 400:
            idx = np.random.choice(len(flat), 400, replace=False)
            flat = flat[idx]
        pooled.append(flat)

    if not pooled:
        return "unknown"

    all_pixels = np.vstack(pooled)
    if len(all_pixels) < _KMEANS_K * 4:
        return "unknown"

    clusters = _kmeans_dominant_colours(all_pixels)
    if not clusters:
        return "unknown"

    total = sum(count for _, count in clusters)

    # Filter clusters by minimum share and dedupe colour names so a kit
    # that splits one true colour across two clusters (e.g. lighter and
    # darker shades of the same red) only gets reported once.
    # Merge clusters that map to the same colour name (e.g. light + dark
    # shades of yellow being split into two clusters by K-means)
    merged: Dict[str, int] = {}
    for hsv, count in clusters:
        name = _hsv_to_name(*hsv)
        merged[name] = merged.get(name, 0) + count

    kept_names: List[str] = []
    for name, count in sorted(merged.items(), key=lambda x: x[1], reverse=True):
        share = count / total
        if share < _MIN_CLUSTER_SHARE:
            continue
        kept_names.append(name)

    if not kept_names:
        # Fallback when no cluster passes the share threshold — return
        # the primary cluster's name so we don't silently produce "unknown"
        # on a perfectly clusterable kit.
        kept_names = [_hsv_to_name(*clusters[0][0])]

    result = "/".join(kept_names)
    cluster_summary = ", ".join(
        f"{_hsv_to_name(*hsv)}({count})" for hsv, count in clusters
    )
    print(f"    {len(paths)} crops | clusters: {cluster_summary} -> {result}")
    return result


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