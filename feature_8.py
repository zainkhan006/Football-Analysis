"""
Feature 8 — Team Assignment  (robust rewrite)
==============================================
S1  crop_jersey(frame, bbox)   → BGR patch
S2  assign_teams(frame, entries, cfg) → {track_id: "home"|"away"|"gk"}

Design goals
------------
* Works when BOTH teams wear green (no hard-coded colour exclusion).
* Handles tiny bboxes (as small as 22x57 px in the Leipzig dataset).
* Handles partial occlusion, edge-of-frame players, and blurry crops.
* Separates referee / GK outliers from the two main clusters.
* Produces stable labels across a sequence via centroid anchoring.
* Zero mandatory tuning — sensible defaults cover the common cases;
  every knob is in FeatureConfig so callers can override per-sequence.

Pipeline per frame
------------------
  raw bbox
    └─► crop_jersey()       — extract torso band, multi-sample fallback
          └─► _describe()   — HSV histogram feature vector (no hard mask)
                └─► _kmeans_robust() — k=2 with outlier bucket
                      └─► _resolve_labels() — map cluster→home/away/gk
                            └─► {track_id: label}

Folder layout expected
----------------------
    <root>/
      videos/
        <seq_name>/
          img1/        ← JPG frames named 000001.jpg, 000002.jpg, …
          gt/
            gt.txt
      feature_8.py     ← this file
    output/
      feature_8/
        <seq_name>/
          crops_frame_001.jpg
          overlay_frame_001.jpg
          …

Usage (script mode)
-------------------
    python feature_8.py                        # all sequences, first 50 frames
    python feature_8.py --seq v_HdiyOtliFiw_c003
    python feature_8.py --seq v_HdiyOtliFiw_c003 --frames 155 200
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import config
    _DATASET_ROOT = config.datasetRoot / config.testSplit
except ImportError:
    config = None
    _DATASET_ROOT = Path(r"C:\Users\Zain Ul Ibad\Desktop\projects\cv_project\sportsmot_publish\dataset\train")

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FeatureConfig:
    # ── Jersey band (fraction of bbox height) ─────────────────────────
    # We try multiple vertical windows and pick the one with the lowest
    # brightness variance (= most uniform fabric, least sky/grass).
    band_candidates: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.10, 0.45),   # primary: head cleared, stops before shorts
        (0.15, 0.55),   # fallback A: slightly lower
        (0.05, 0.40),   # fallback B: slightly higher (tall/close players)
    ])

    # ── Horizontal strip (fraction of bbox width) ─────────────────────
    strip_left:  float = 0.20   # wider than original — captures more fabric
    strip_right: float = 0.80

    # ── Minimum crop area to be considered usable ─────────────────────
    min_crop_px: int = 60       # pixels² after clamping

    # ── HSV histogram feature vector ──────────────────────────────────
    hue_bins: int = 16          # 0-179 → 16 bins of ~11° each
    sat_bins: int = 4           # coarse saturation (washed/mid/vivid/very vivid)
    val_bins: int = 4           # coarse brightness (dark/mid/bright/very bright)
    # Final feature dim = hue_bins + sat_bins + val_bins = 24 by default

    # Saturation threshold for hue histogram masking — pixels below this
    # are treated as achromatic (white/grey/black) and excluded from the
    # hue distribution since their hue values are noise.
    chromatic_sat_min: int = 30

    # ── Outlier detection (referee / GK) ──────────────────────────────
    # A track is tagged "gk" if its distance to BOTH team centroids
    # exceeds this fraction of the centroid-to-centroid distance.
    outlier_threshold: float = 0.50
    # Suppress outlier flagging when either cluster has fewer than this many
    # members — prevents a lone attacker from being wrongly tagged gk.
    min_cluster_size_for_outlier: int = 2

    # ── K-means ───────────────────────────────────────────────────────
    kmeans_attempts:  int = 15
    kmeans_k_initial: int = 6
    kmeans_max_iter:  int = 60
    kmeans_epsilon:   float = 0.5

    # ── Sequence-level centroid anchoring ─────────────────────────────
    # Once centroids are computed for the first "anchor" frame batch,
    # subsequent frames use them as warm-start seeds (prevents cluster
    # flip between frames).
    anchor_frames: int = 10     # number of frames pooled for initial centroids

    # ── Minimum tracks needed to run K-means ──────────────────────────
    min_tracks_for_kmeans: int = 4

    # ── Debug strip output ────────────────────────────────────────────
    tile_w: int = 64
    tile_h: int = 112


# ══════════════════════════════════════════════════════════════════════
# S1 — Jersey crop pipeline
# ══════════════════════════════════════════════════════════════════════

def _band_score(patch: np.ndarray) -> float:
    """
    Score a candidate jersey band — lower = better.

    Combines BGR uniformity (low variance is good — uniform fabric) with
    saturation (high is good — chromatic jersey beats white shorts).
    A plain white shorts patch has low BGR variance but also low
    saturation, so its score is penalised to keep numbered jerseys ahead
    of solid-colour shorts.
    """
    if patch.size == 0:
        return 1e9
    bgr_var = float(np.mean(np.var(patch.reshape(-1, 3).astype(np.float32), axis=0)))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    # divide by (1 + sat/30): boosts saturated bands, demotes washed-out ones
    return bgr_var / (1.0 + mean_sat / 30.0)


def crop_jersey(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    cfg: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """
    Extract the jersey torso region from one bounding box.

    Strategy
    --------
    1. Try each band_candidate window (different vertical fractions).
    2. Score each candidate by chromatic uniformity (BGR variance
       penalised by low saturation, so white shorts don't beat a
       numbered jersey).
    3. Return the lowest-score crop.
    4. If all candidates are degenerate (< min_crop_px), return a 1×1
       grey fallback so downstream code never crashes.

    Parameters
    ----------
    frame : np.ndarray  BGR full frame
    bbox  : (x, y, w, h) pixel coordinates
    cfg   : FeatureConfig

    Returns
    -------
    np.ndarray  BGR crop of the jersey region (may be very small).
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    fH, fW = frame.shape[:2]

    # Clamp to frame boundaries
    x  = max(0, x);       y  = max(0, y)
    x2 = min(fW, x + w);  y2 = min(fH, y + h)
    bh = y2 - y;          bw = x2 - x

    if bh <= 0 or bw <= 0:
        return np.full((1, 1, 3), 128, dtype=np.uint8)

    sx = x + int(bw * cfg.strip_left)
    ex = x + int(bw * cfg.strip_right)
    sx = max(x, sx);  ex = min(x2, ex)
    if ex <= sx:
        ex = x2;  sx = x

    best_crop: Optional[np.ndarray] = None
    best_score = 1e9

    for (top_f, bot_f) in cfg.band_candidates:
        sy = y + int(bh * top_f)
        ey = y + int(bh * bot_f)
        sy = max(y, sy);  ey = min(y2, ey)
        if ey <= sy:
            continue

        patch = frame[sy:ey, sx:ex]
        area  = patch.shape[0] * patch.shape[1]
        if area < cfg.min_crop_px:
            continue

        score = _band_score(patch)
        if score < best_score:
            best_score = score
            best_crop  = patch

    if best_crop is None or best_crop.size == 0:
        return np.full((1, 1, 3), 128, dtype=np.uint8)

    return best_crop


# ══════════════════════════════════════════════════════════════════════
# S2-A — Feature extraction (HSV histogram, no colour hard-mask)
# ══════════════════════════════════════════════════════════════════════

def _describe(crop: np.ndarray, cfg: FeatureConfig) -> Optional[np.ndarray]:
    """
    Compute a normalised HSV histogram feature vector for one crop.

    Why HSV histogram instead of mean BGR?
    ───────────────────────────────────────
    • Mean BGR collapses multi-colour kits (stripes/hoops) to a muddy mid-tone.
    • A histogram captures the *distribution* of hues — a red-and-white kit
      has peaks at red AND near-zero-saturation (white), which is distinctive.
    • HSV separates chromatic information (H) from lighting (V), making it
      robust to the shadowy/brightly-lit regions of a broadcast pitch.
    • We do NOT hard-mask grass pixels because green teams must also work.
      Instead we use all pixels equally — the clustering separates teams by
      *relative* difference, which still works even if both are greenish.

    The hue histogram is computed only over chromatic pixels (saturation
    above cfg.chromatic_sat_min). Achromatic pixels (white/grey/black) have
    noisy hue values that pollute the histogram. The saturation and value
    histograms still see all pixels so achromatic information is preserved.

    Returns None if the crop is too small to be reliable.
    """
    area = crop.shape[0] * crop.shape[1]
    if area < cfg.min_crop_px:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    sat_mask = (hsv[:, :, 1] >= cfg.chromatic_sat_min).astype(np.uint8) * 255

    h_hist = cv2.calcHist([hsv], [0], sat_mask, [cfg.hue_bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None,     [cfg.sat_bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None,     [cfg.val_bins], [0, 256]).flatten()

    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    total = feat.sum()
    if total == 0:
        return None
    return feat / total      # L1-normalise → comparable across crop sizes


# ══════════════════════════════════════════════════════════════════════
# S2-B — Robust K-means (k=2, with outlier bucket)
# ══════════════════════════════════════════════════════════════════════

def _kmeans_robust(
    features: np.ndarray,          # (N, D) float32
    cfg: FeatureConfig,
    init_centroids: Optional[np.ndarray] = None,   # (2, D) warm-start
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run k=2 K-means and mark outliers.

    Returns
    -------
    labels    : (N,) int  — 0, 1, or 2 (outlier)
    centroids : (2, D)    — the two team centroids
    distances : (N, 2)    — L2 distance of each sample to each centroid
    """
    N = len(features)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        cfg.kmeans_max_iter,
        cfg.kmeans_epsilon,
    )

    if init_centroids is not None and init_centroids.shape == (2, features.shape[1]):
        # Warm-start: assign each point to nearest seed, then iterate
        flags = cv2.KMEANS_USE_INITIAL_LABELS
        # Build initial label assignment from seeds
        d0 = np.linalg.norm(features - init_centroids[0], axis=1)
        d1 = np.linalg.norm(features - init_centroids[1], axis=1)
        init_labels = (d1 < d0).astype(np.int32).reshape(-1, 1)
        _, labels, centroids = cv2.kmeans(
            features, 2, init_labels, criteria, 1, flags
        )
    else:
        # Over-cluster into k_initial micro-clusters first to give K-means
        # room to separate visually-similar kits (e.g. white vs yellow-with-
        # white-stripes). We then pick the two micro-clusters that are
        # most colour-different from each other (not the two largest), so
        # the team seeds maximise inter-team distance.
        k_init = min(cfg.kmeans_k_initial, len(features))
        _, micro_labels, micro_centroids = cv2.kmeans(
            features, k_init, None, criteria,
            cfg.kmeans_attempts, cv2.KMEANS_PP_CENTERS
        )
        micro_labels = micro_labels.flatten()

        # Find the pair of micro-centroids with maximum L2 distance — these
        # are the two most colour-distinct clusters and become our seeds.
        # Each must also have at least a few members so we don't pick a
        # tiny outlier cluster like a single referee crop.
        micro_sizes = np.array([int(np.sum(micro_labels == k)) for k in range(k_init)])
        min_size = max(2, int(0.05 * len(features)))
        valid_idx = [k for k in range(k_init) if micro_sizes[k] >= min_size]
        if len(valid_idx) < 2:
            valid_idx = list(range(k_init))

        best_pair = (valid_idx[0], valid_idx[1])
        best_dist = -1.0
        for i in valid_idx:
            for j in valid_idx:
                if i >= j:
                    continue
                d = float(np.linalg.norm(micro_centroids[i] - micro_centroids[j]))
                if d > best_dist:
                    best_dist = d
                    best_pair = (i, j)

        seed_centroids = np.stack([micro_centroids[best_pair[0]],
                                    micro_centroids[best_pair[1]]])

        # Re-cluster everything into exactly 2 using the seeds as warm-start
        d0 = np.linalg.norm(features - seed_centroids[0], axis=1)
        d1 = np.linalg.norm(features - seed_centroids[1], axis=1)
        init_labels = (d1 < d0).astype(np.int32).reshape(-1, 1)
        _, labels, centroids = cv2.kmeans(
            features, 2, init_labels, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS
        )

    labels = labels.flatten().astype(int)       # (N,)

    # Compute L2 distances to both centroids
    d0 = np.linalg.norm(features - centroids[0], axis=1)
    d1 = np.linalg.norm(features - centroids[1], axis=1)
    distances = np.stack([d0, d1], axis=1)      # (N, 2)

    # Outlier test: far from BOTH centroids → referee / GK
    # Cluster-size guard: suppress outlier flagging when either cluster is tiny
    # (e.g. lone attacker in frame) — they are a real team, just underrepresented.
    centroid_dist = float(np.linalg.norm(centroids[0] - centroids[1]))
    threshold     = cfg.outlier_threshold * max(centroid_dist, 1e-6)
    count0 = int(np.sum(labels == 0))
    count1 = int(np.sum(labels == 1))
    if count0 >= cfg.min_cluster_size_for_outlier and count1 >= cfg.min_cluster_size_for_outlier:
        is_outlier = (d0 > threshold) & (d1 > threshold)
        labels[is_outlier] = 2
    # else: one cluster tiny → trust all assignments, flag nothing as outlier

    return labels, centroids, distances


# ══════════════════════════════════════════════════════════════════════
# S2-C — Label resolution (cluster index → semantic string)
# ══════════════════════════════════════════════════════════════════════

def _resolve_labels(
    track_ids: List[int],
    labels:    np.ndarray,           # (N,) with 0/1/2
    centroids: np.ndarray,           # (2, D)
    anchor_centroids: Optional[np.ndarray],  # frozen original anchor — prevents flip
) -> Dict[int, str]:
    """
    Map cluster indices 0/1/2 to "home"/"away"/"gk".

    Flip prevention
    ---------------
    If anchored centroids exist from the original sequence anchor, we
    compare the current centroids to them.  If cluster-0 is closer to the
    anchor cluster-1 centroid, we flip.  Using the FROZEN anchor (not the
    previous frame) prevents gradual drift over long sequences — even if
    lighting slowly shifts both centroids, the home/away identity stays
    pinned to the original anchor reference.
    """
    result: Dict[int, str] = {}

    # Default mapping: 0→home, 1→away
    swap = False
    if anchor_centroids is not None:
        d_same = (
            np.linalg.norm(centroids[0] - anchor_centroids[0]) +
            np.linalg.norm(centroids[1] - anchor_centroids[1])
        )
        d_swap = (
            np.linalg.norm(centroids[0] - anchor_centroids[1]) +
            np.linalg.norm(centroids[1] - anchor_centroids[0])
        )
        swap = d_swap < d_same

    label_map = {0: "away" if swap else "home",
                 1: "home" if swap else "away",
                 2: "gk"}

    for tid, lbl in zip(track_ids, labels):
        result[tid] = label_map[lbl]

    return result


# ══════════════════════════════════════════════════════════════════════
# S2 — Public assign_teams  (one frame)
# ══════════════════════════════════════════════════════════════════════

def assign_teams(
    frame:   np.ndarray,
    entries: List[Tuple[int, Tuple[int, int, int, int]]],
    cfg:     FeatureConfig = FeatureConfig(),
    init_centroids: Optional[np.ndarray] = None,
    anchor_centroids: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, str], Optional[np.ndarray]]:
    """
    Assign every tracked player to home / away / gk for one frame.

    Parameters
    ----------
    frame            : BGR full frame
    entries          : [(track_id, (x, y, w, h)), ...]
    cfg              : FeatureConfig
    init_centroids   : (2, D) float32 — rolling warm-start seed for K-means.
                       Pass None for the first batch; pass returned centroids
                       for subsequent frames.
    anchor_centroids : (2, D) float32 — FROZEN anchor reference used only for
                       flip prevention. Should be set once from the initial
                       anchor and never updated. If None, falls back to
                       init_centroids (legacy behaviour).

    Returns
    -------
    labels     : {track_id: "home"|"away"|"gk"}
    centroids  : (2, D) float32 — updated centroids (feed back as init_centroids)
                 Returns None if there were not enough tracks to cluster.
    """
    if not entries:
        return {}, init_centroids

    crops    = [crop_jersey(frame, bbox, cfg) for _, bbox in entries]
    feats    = [_describe(c, cfg) for c in crops]

    valid_ids:   List[int]         = []
    valid_feats: List[np.ndarray]  = []
    invalid_ids: List[int]         = []

    for (tid, _), feat in zip(entries, feats):
        if feat is not None:
            valid_ids.append(tid)
            valid_feats.append(feat)
        else:
            invalid_ids.append(tid)

    # Not enough usable tracks — assign everything to gk
    if len(valid_ids) < cfg.min_tracks_for_kmeans:
        fallback = {tid: "gk" for tid in valid_ids + invalid_ids}
        return fallback, init_centroids

    feat_matrix = np.stack(valid_feats)   # (N, D)
    labels, centroids, _ = _kmeans_robust(feat_matrix, cfg, init_centroids)

    # Use the frozen anchor for flip detection if available, otherwise
    # fall back to init_centroids (preserves backward compatibility).
    flip_reference = anchor_centroids if anchor_centroids is not None else init_centroids
    label_dict = _resolve_labels(valid_ids, labels, centroids, flip_reference)

    # Invalid (too-small bbox) tracks marked gk
    for tid in invalid_ids:
        label_dict[tid] = "gk"

    return label_dict, centroids


# ══════════════════════════════════════════════════════════════════════
# Majority-vote stabiliser  (sequence-level, post-processing pass)
# ══════════════════════════════════════════════════════════════════════

def majority_vote_labels(
    all_labels: Dict[int, Dict[int, str]],
) -> Tuple[Dict[int, str], List[str]]:
    """
    Compute a stable, per-track final label by majority vote across all frames.

    gk is treated as uncertainty, not a real identity.  A track needs
    more than 30% of its visible frames labelled home/away before it is
    rescued from gk.  gk only wins if the track was never
    confidently assigned a team label across the whole sequence.
    """
    votes: Dict[int, Dict[str, int]] = {}

    for frame_labels in all_labels.values():
        for tid, lbl in frame_labels.items():
            if tid not in votes:
                votes[tid] = {}
            votes[tid][lbl] = votes[tid].get(lbl, 0) + 1

    final: Dict[int, str] = {}
    rescue_log = []

    for tid, counts in votes.items():
        total_frames = sum(counts.values())
        team_votes   = {k: v for k, v in counts.items() if k != "gk"}
        total_team   = sum(team_votes.values())
        gk_frames    = counts.get("gk", 0)

        if total_team > int((30 / 100) * total_frames):
            final[tid] = max(team_votes, key=team_votes.__getitem__)
            if gk_frames > 0:
                rescue_log.append(
                    f"track #{tid} -> rescued to '{final[tid]}' "
                    f"(was gk in {gk_frames} frames, "
                    f"team label in {total_team}/{total_frames} frames)"
                )
        else:
            final[tid] = "gk"

    return final, rescue_log


def apply_majority_vote(
    all_labels: Dict[int, Dict[int, str]],
) -> Tuple[Dict[int, Dict[int, str]], Dict[int, str], List[str]]:
    """
    Replace every per-frame label with the sequence-level majority-vote label.

    Returns
    -------
    stabilised       : per-frame labels with all frames sharing one stable label per track
    final_per_track  : {track_id: final_label} for the whole sequence
    rescue_log       : list of rescue messages
    """
    final_per_track, rescue_log = majority_vote_labels(all_labels)
    stabilised: Dict[int, Dict[int, str]] = {}
    for fi, frame_labels in all_labels.items():
        stabilised[fi] = {
            tid: final_per_track.get(tid, lbl)
            for tid, lbl in frame_labels.items()
        }
    return stabilised, final_per_track, rescue_log


# ══════════════════════════════════════════════════════════════════════
# Sequence-level runner  (pools first N frames for stable centroids)
# ══════════════════════════════════════════════════════════════════════

def run_sequence(
    frame_paths: List[str],
    gt_path:     str,
    frame_indices: List[int],
    cfg:         FeatureConfig = FeatureConfig(),
) -> Dict[int, Dict[int, str]]:
    """
    Process multiple frames of one sequence with centroid anchoring.

    Returns {frame_idx: {track_id: label}}
    """
    all_labels: Dict[int, Dict[int, str]] = {}
    anchored_centroids: Optional[np.ndarray] = None
    original_anchor:    Optional[np.ndarray] = None
    anchor_feats: List[np.ndarray] = []
    anchor_collected = 0

    frame_gt = _load_all_gt(gt_path)

    for frame_idx in sorted(frame_indices):
        frame_path = _frame_path_for_idx(frame_paths, frame_idx)
        if frame_path is None:
            continue
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        entries = frame_gt.get(frame_idx, [])
        if not entries:
            continue

        # Collect features for anchor pool
        if anchor_collected < cfg.anchor_frames:
            crops = [crop_jersey(frame, bbox, cfg) for _, bbox in entries]
            for c in crops:
                f = _describe(c, cfg)
                if f is not None:
                    anchor_feats.append(f)
            anchor_collected += 1

            # Once enough anchor frames collected, compute initial centroids
            if anchor_collected == cfg.anchor_frames and len(anchor_feats) >= cfg.min_tracks_for_kmeans:
                feat_matrix = np.stack(anchor_feats)
                _, anchored_centroids, _ = _kmeans_robust(feat_matrix, cfg)
                original_anchor = anchored_centroids.copy()   # freeze for flip detection

        labels, anchored_centroids = assign_teams(
            frame, entries, cfg, anchored_centroids,
            anchor_centroids=original_anchor,
        )
        all_labels[frame_idx] = labels

    all_labels, _, _ = apply_majority_vote(all_labels)
    return all_labels


# ══════════════════════════════════════════════════════════════════════
# Pipeline adapter — TeamAssigner class
# ══════════════════════════════════════════════════════════════════════

class TeamAssigner:
    """
    Stateful adapter so pipeline.py can call team assignment frame-by-frame
    without managing centroid state manually.

    Accepts tracks in Zain's pipeline format (list of dicts with 'trackId'
    and 'bbox' keys) and returns {trackId: 'home'|'away'|'gk'}.

    Usage
    -----
        assigner = TeamAssigner()
        for frame, tracks in stream:
            labels = assigner.assign(frame, tracks)
    """
    def __init__(self, cfg: Optional[FeatureConfig] = None):
        self.cfg = cfg if cfg is not None else FeatureConfig()
        self.anchored_centroids: Optional[np.ndarray] = None
        self.original_anchor:    Optional[np.ndarray] = None

    def reset(self):
        """Clear centroid state — call when starting a new sequence."""
        self.anchored_centroids = None
        self.original_anchor    = None

    def assign(self, frame: np.ndarray, tracks: List[Dict]) -> Dict[int, str]:
        """
        Assign teams for one frame.

        Parameters
        ----------
        frame  : BGR full frame
        tracks : list of dicts with 'trackId' and 'bbox' keys
                 (bbox is (x, y, w, h))
        """
        entries = [(int(t["trackId"]), t["bbox"]) for t in tracks]
        labels, self.anchored_centroids = assign_teams(
            frame, entries, self.cfg, self.anchored_centroids,
            anchor_centroids=self.original_anchor,
        )
        # Freeze the very first valid centroids as the anchor reference.
        if self.original_anchor is None and self.anchored_centroids is not None:
            self.original_anchor = self.anchored_centroids.copy()
        return labels


# ══════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════

_LABEL_COLOUR = {
    "home":   (60,  200, 60),    # green tint
    "away":   (60,  120, 220),   # blue tint
    "gk": (220, 180, 40),    # amber
    "?":      (120, 120, 120),
}


def _crop_full_bbox(frame: np.ndarray, bbox) -> np.ndarray:
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    H, W = frame.shape[:2]
    x = max(0, x);  y = max(0, y)
    x2 = min(W, x+w);  y2 = min(H, y+h)
    if x2 <= x or y2 <= y:
        return np.full((1, 1, 3), 128, dtype=np.uint8)
    return frame[y:y2, x:x2]


def build_labelled_strip(
    frame:    np.ndarray,
    entries:  List[Tuple[int, Tuple[int, int, int, int]]],
    labels:   Dict[int, str],
    cfg:      FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """
    Three-row strip per track:
      Row 1: full bbox crop
      Row 2: jersey crop (the actual band used)
      Row 3: header with track id + label

    Useful for visual inspection of S1 quality.
    """
    HEADER_H = 26
    tw, th = cfg.tile_w, cfg.tile_h
    tiles = []

    for tid, bbox in entries:
        lbl   = labels.get(tid, "?")
        color = _LABEL_COLOUR.get(lbl, _LABEL_COLOUR["?"])

        full    = _crop_full_bbox(frame, bbox)
        jersey  = crop_jersey(frame, bbox, cfg)

        full_t   = cv2.resize(full,   (tw, th))
        jersey_t = cv2.resize(jersey, (tw, th))

        header = np.zeros((HEADER_H, tw, 3), dtype=np.uint8)
        cv2.putText(header, f"#{tid}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(header, lbl, (2, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        sep = np.full((3, tw, 3), 50, dtype=np.uint8)
        tiles.append(np.vstack([header, full_t, sep, jersey_t]))

    if not tiles:
        return np.zeros((HEADER_H + th*2 + 3, tw, 3), dtype=np.uint8)
    return np.hstack(tiles)


def draw_overlay(
    frame:   np.ndarray,
    entries: List[Tuple[int, Tuple[int, int, int, int]]],
    labels:  Dict[int, str],
) -> np.ndarray:
    """Draw coloured bounding boxes + labels directly on the frame."""
    out = frame.copy()
    for tid, (x, y, w, h) in entries:
        lbl   = labels.get(tid, "?")
        color = _LABEL_COLOUR.get(lbl, _LABEL_COLOUR["?"])
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        cv2.putText(out, f"#{tid} {lbl}", (x, max(y-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════
# GT / filesystem helpers
# ══════════════════════════════════════════════════════════════════════

def _load_all_gt(gt_path: str) -> Dict[int, List[Tuple[int, Tuple[int,int,int,int]]]]:
    """Load entire gt.txt into {frame_idx: [(track_id, (x,y,w,h)), ...]}."""
    data: Dict[int, list] = {}
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            fi  = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            data.setdefault(fi, []).append((tid, (x, y, w, h)))
    return data


def _collect_frame_paths(img_dir: str) -> List[str]:
    """Return all image paths in img_dir sorted by filename."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        str(Path(img_dir) / f)
        for f in os.listdir(img_dir)
        if Path(f).suffix.lower() in exts
    )
    return paths


def _frame_index_of(path: str) -> int:
    """Parse the numeric stem of a frame filename: '000042.jpg' -> 42."""
    try:
        return int(Path(path).stem)
    except ValueError:
        return -1


def _frame_path_for_idx(frame_paths: List[str], frame_idx: int) -> Optional[str]:
    """Find the frame path whose numeric stem equals frame_idx."""
    for p in frame_paths:
        if _frame_index_of(p) == frame_idx:
            return p
    return None


# ══════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════

DEFAULT_FIRST_N = 50   # process first N frames of every sequence by default


def run_feature8(
    seq_filter:    Optional[str]      = None,
    explicit_frames: Optional[List[int]] = None,
    first_n:       int                = DEFAULT_FIRST_N,
) -> None:
    """
    Walk videos/<seq>/img1/, process frames, write to output/feature_8/<seq>/.

    Output layout
    -------------
    output/
      feature_8/
        <seq_name>/
          crops/
            crops_frame_001.jpg    ← labelled crop strip (S1 quality check)
            …
          overlays/
            overlay_frame_001.jpg  ← full frame with coloured bboxes (S2 labels)
            …

    Parameters
    ----------
    seq_filter      : if set, only process this one sequence folder name.
    explicit_frames : if set, process exactly these frame indices (1-based).
                      Otherwise takes the first `first_n` frames found in img1/.
    first_n         : how many leading frames to process when explicit_frames
                      is not given (default 50).
    """
    here        = Path(__file__).parent
    # Dataset root comes from config.py so the path stays consistent with
    # the rest of the pipeline. Sequences are searched under config.testSplit
    # by default — update config.testSplit (e.g. to "train") if running on
    # other splits.
    videos_root = _DATASET_ROOT
    out_root    = here / "output" / "feature_8"

    if not videos_root.is_dir():
        raise FileNotFoundError(
            f"dataset folder not found at {videos_root}\n"
            f"check config.datasetRoot and config.testSplit"
        )

    seq_dirs = sorted(d for d in videos_root.iterdir() if d.is_dir())
    if seq_filter:
        seq_dirs = [d for d in seq_dirs if d.name == seq_filter]
        if not seq_dirs:
            raise FileNotFoundError(
                f"Sequence '{seq_filter}' not found under {videos_root}"
            )

    print(f"Found {len(seq_dirs)} sequence(s)")
    cfg = FeatureConfig()

    for seq_dir in seq_dirs:
        img_dir = seq_dir / "img1"
        gt_path = seq_dir / "gt" / "gt.txt"
        seq_out      = out_root / seq_dir.name
        crops_out    = seq_out / "crops"
        overlays_out = seq_out / "overlays"

        if not img_dir.is_dir():
            print(f"[{seq_dir.name}] no img1/ folder — skipped")
            continue
        if not gt_path.is_file():
            print(f"[{seq_dir.name}] no gt/gt.txt — skipped")
            continue

        crops_out.mkdir(parents=True, exist_ok=True)
        overlays_out.mkdir(parents=True, exist_ok=True)

        # ── Discover frames ────────────────────────────────────────────
        all_frame_paths = _collect_frame_paths(str(img_dir))
        if not all_frame_paths:
            print(f"[{seq_dir.name}] img1/ is empty — skipped")
            continue

        if explicit_frames is not None:
            # Caller supplied specific indices — honour them exactly
            frame_indices = sorted(explicit_frames)
        else:
            # Take the first `first_n` frames by sorted filename order
            chosen_paths  = all_frame_paths[:first_n]
            frame_indices = sorted(
                _frame_index_of(p) for p in chosen_paths
                if _frame_index_of(p) >= 0
            )

        print(f"[{seq_dir.name}] processing {len(frame_indices)} frame(s): "
              f"{frame_indices[0]}…{frame_indices[-1]}")

        frame_gt = _load_all_gt(str(gt_path))

        # ── Anchor phase: pool first cfg.anchor_frames for stable centroids
        anchor_feats: List[np.ndarray] = []
        anchored_centroids: Optional[np.ndarray] = None
        original_anchor:    Optional[np.ndarray] = None
        frames_for_anchor = frame_indices[: cfg.anchor_frames]

        for fi in frames_for_anchor:
            fp = _frame_path_for_idx(all_frame_paths, fi)
            if fp is None:
                continue
            fr = cv2.imread(fp)
            if fr is None:
                continue
            for _, bbox in frame_gt.get(fi, []):
                feat = _describe(crop_jersey(fr, bbox, cfg), cfg)
                if feat is not None:
                    anchor_feats.append(feat)

        if len(anchor_feats) >= cfg.min_tracks_for_kmeans:
            _, anchored_centroids, _ = _kmeans_robust(
                np.stack(anchor_feats), cfg
            )
            original_anchor = anchored_centroids.copy()   # freeze for flip detection
            print(f"[{seq_dir.name}] anchor built from "
                  f"{len(anchor_feats)} crops "
                  f"({len(frames_for_anchor)} anchor frame(s))")
        else:
            print(f"[{seq_dir.name}] not enough anchor crops "
                  f"({len(anchor_feats)}) — will cluster per-frame")

        # ── Pass 1: collect raw per-frame labels ──────────────────────
        raw_labels:    Dict[int, Dict[int, str]] = {}
        frame_entries: Dict[int, List]            = {}

        for fi in frame_indices:
            fp = _frame_path_for_idx(all_frame_paths, fi)
            if fp is None:
                print(f"[{seq_dir.name}] frame {fi:06d} not found — skipped")
                continue

            frame = cv2.imread(fp)
            if frame is None:
                print(f"[{seq_dir.name}] frame {fi:06d} unreadable — skipped")
                continue

            entries = frame_gt.get(fi, [])
            if not entries:
                print(f"[{seq_dir.name}] frame {fi:06d} has no GT rows — skipped")
                continue

            labels, anchored_centroids = assign_teams(
                frame, entries, cfg, anchored_centroids,
                anchor_centroids=original_anchor,
            )
            raw_labels[fi]    = labels
            frame_entries[fi] = entries

        # ── Majority-vote stabilisation ───────────────────────────────
        stable_labels, voted, rescue_log = apply_majority_vote(raw_labels)

        log_path = seq_out / "gk_rescue_log.txt"
        with open(log_path, "w") as f:
            if rescue_log:
                f.write(f"GK Rescue Log — {seq_dir.name}\n")
                f.write("=" * 50 + "\n")
                for line in rescue_log:
                    f.write(line + "\n")
            else:
                f.write("No tracks were rescued from gk.\n")
        print(f"[{seq_dir.name}] rescue log → {log_path}")


        print(f"[{seq_dir.name}] majority-vote final assignments: "
              + ", ".join(f"#{tid}->{lbl}" for tid, lbl in sorted(voted.items())))

        # ── Pass 2: write visualisation images using stable labels ────
        for fi, entries in sorted(frame_entries.items()):
            fp = _frame_path_for_idx(all_frame_paths, fi)
            if fp is None:
                continue
            frame = cv2.imread(fp)
            if frame is None:
                continue

            labels = stable_labels[fi]

            strip   = build_labelled_strip(frame, entries, labels, cfg)
            cv2.imwrite(str(crops_out / f"crops_frame_{fi:03d}.jpg"), strip)

            overlay = draw_overlay(frame, entries, labels)
            cv2.imwrite(str(overlays_out / f"overlay_frame_{fi:03d}.jpg"), overlay)

            # Save individual jersey crops per team
            for tid, bbox in entries:
                lbl  = labels.get(tid, "?")
                if lbl not in ("home", "away", "gk"):
                    continue
                jersey = crop_jersey(frame, bbox, cfg)
                if jersey.shape[0] * jersey.shape[1] < cfg.min_crop_px:
                    continue
                team_dir = seq_out / "jersey_crops" / lbl
                team_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(team_dir / f"frame_{fi:03d}_track_{tid}.jpg"), jersey)


            counts: Dict[str, int] = {}
            for v in labels.values():
                counts[v] = counts.get(v, 0) + 1

            print(f"[{seq_dir.name}] frame {fi:4d} | "
                  f"tracks={len(entries):2d} | "
                  f"home={counts.get('home',0):2d}  "
                  f"away={counts.get('away',0):2d}  "
                  f"gk={counts.get('gk',0):2d}")

        print(f"[{seq_dir.name}] done → crops: {crops_out} | overlays: {overlays_out}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── DEFINE WHICH SEQUENCE TO RUN ──────────────────────────────────
    SEQ_NAME = "v_HdiyOtliFiw_c003"  # ← CHANGE THIS to your sequence name
    
    # Dataset root comes from config.py — update config.testSplit if you
    # need to run on a different split (e.g. "train").
    videos_root = config.datasetRoot / "train" if config is not None else Path(r"C:\Users\Zain Ul Ibad\Desktop\projects\cv_project\sportsmot_publish\dataset\train")
    
    # Process ONLY the specified sequence
    seq_dir = videos_root / SEQ_NAME
    
    if not seq_dir.is_dir():
        print(f"Error: Sequence '{SEQ_NAME}' not found in {videos_root}")
        exit(1)
    
    all_frame_paths = _collect_frame_paths(str(seq_dir / "img1"))
    all_frame_indices = sorted(
        _frame_index_of(p)
        for p in all_frame_paths
        if _frame_index_of(p) >= 0
    )
    
    print(f"Processing {len(all_frame_indices)} frames from {seq_dir.name}")
    
    run_feature8(
        seq_filter=seq_dir.name,
        explicit_frames=all_frame_indices,
    )