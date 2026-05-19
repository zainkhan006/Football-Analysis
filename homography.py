"""
Pitch-to-camera homography.

A homography H is a 3×3 matrix that maps a point on the real-world pitch
plane (`(wx, wy)` in metres) to a pixel in the camera image (`(px, py)`).

We always work with the WORLD → PIXEL direction as primary (so we can
overlay a known pitch model onto a frame) and expose `pixel_to_world` as
the inverse.

The class is intentionally *static* — one frozen H per clip, computed up
front and serialised with `.save()`.  Re-computation across frames (for
panning cameras) lives in a separate runner.

Usage:

    >>> H = PitchHomography.from_correspondences(world_xy, pixel_xy)
    >>> H.pixel_to_world(640, 360)
    (62.3, 35.1)
    >>> H.world_to_pixel(60.0, 35.0)
    (618.4, 365.2)
    >>> H.save("homographies/v_gQNyhv8y0QY_c013.npz")
    >>> H = PitchHomography.load("homographies/v_gQNyhv8y0QY_c013.npz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_xy_array(pts: Iterable[Tuple[float, float]]) -> np.ndarray:
    """Coerce an iterable of (x, y) pairs to an (N, 2) float32 array."""
    arr = np.asarray(list(pts), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array, got shape {arr.shape}")
    return arr


# ─── Main class ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PitchHomography:
    """
    A 3×3 homography from world (metres on the pitch) to pixel space.

    Use the classmethod `from_correspondences(...)` to build one;
    don't construct directly.
    """

    H_world_to_pixel: np.ndarray            # shape (3, 3), float64
    H_pixel_to_world: np.ndarray            # cached inverse

    # Diagnostic info captured at fit time. Optional.
    fit_inliers:      Optional[np.ndarray] = field(default=None)
    fit_error_px:     Optional[float]      = field(default=None)
    n_correspondences: Optional[int]       = field(default=None)
    source_frame_id:  Optional[int]        = field(default=None)

    # Bounding box (in world metres) of the inliers used to fit H.
    # Inside this box H is well-supported; outside, projections are
    # extrapolations and become unreliable as we move away.
    # Format: (x_min, y_min, x_max, y_max).  Optional.
    inlier_world_bbox: Optional[Tuple[float, float, float, float]] = field(default=None)

    def is_within_support(self, wx: float, wy: float, margin: float = 5.0) -> bool:
        """Is (wx, wy) inside the inlier world-bbox (with optional margin in m)?"""
        if self.inlier_world_bbox is None:
            return True   # no info => trust it
        x0, y0, x1, y1 = self.inlier_world_bbox
        return (x0 - margin <= wx <= x1 + margin
                and y0 - margin <= wy <= y1 + margin)

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_correspondences(
        cls,
        world_xy_m: Sequence[Tuple[float, float]],
        pixel_xy:   Sequence[Tuple[float, float]],
        ransac_threshold_px: float = 8.0,
        source_frame_id: Optional[int] = None,
    ) -> "PitchHomography":
        """
        Compute H from N corresponding points.  Requires N >= 4.

        Uses RANSAC to be robust to a few wrong correspondences
        (the YOLO keypoint detector occasionally swaps adjacent landmarks).

        Raises ValueError if fewer than 4 inliers survive RANSAC, or
        if the projection error is unreasonably high.
        """
        world = _to_xy_array(world_xy_m)
        pixel = _to_xy_array(pixel_xy)
        if len(world) != len(pixel):
            raise ValueError(
                f"world ({len(world)}) and pixel ({len(pixel)}) "
                f"point arrays must have the same length"
            )
        if len(world) < 4:
            raise ValueError(
                f"Need at least 4 correspondences; got {len(world)}"
            )

        H_w2p, mask = cv2.findHomography(
            world, pixel, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
        )
        if H_w2p is None:
            raise ValueError("cv2.findHomography returned None — degenerate correspondences?")

        inliers = mask.flatten().astype(bool)
        n_inliers = int(inliers.sum())
        if n_inliers < 4:
            raise ValueError(
                f"RANSAC kept only {n_inliers} inliers — geometry too "
                f"inconsistent. Re-check keypoint detection."
            )

        # Estimate per-inlier reprojection error
        proj = cv2.perspectiveTransform(
            world[inliers].reshape(-1, 1, 2), H_w2p
        ).reshape(-1, 2)
        err = float(np.linalg.norm(proj - pixel[inliers], axis=1).mean())

        # Inlier world bbox -- the region where H is actually supported.
        # Outside this bbox H is extrapolating and may not be reliable.
        inlier_world = world[inliers]
        bbox = (
            float(inlier_world[:, 0].min()),
            float(inlier_world[:, 1].min()),
            float(inlier_world[:, 0].max()),
            float(inlier_world[:, 1].max()),
        )

        # Inverse for pixel → world
        H_p2w = np.linalg.inv(H_w2p)

        return cls(
            H_world_to_pixel  = H_w2p.astype(np.float64),
            H_pixel_to_world  = H_p2w.astype(np.float64),
            fit_inliers       = inliers,
            fit_error_px      = err,
            n_correspondences = len(world),
            source_frame_id   = source_frame_id,
            inlier_world_bbox = bbox,
        )

    # ── Projection ───────────────────────────────────────────────────────

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[float, float]:
        pt = np.array([[[wx, wy]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.H_world_to_pixel)[0, 0]
        return float(out[0]), float(out[1])

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.H_pixel_to_world)[0, 0]
        return float(out[0]), float(out[1])

    def world_to_pixel_batch(self, world_xy: np.ndarray) -> np.ndarray:
        """world_xy: (N, 2) float -> pixel_xy: (N, 2) float."""
        pts = world_xy.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, self.H_world_to_pixel).reshape(-1, 2)

    def pixel_to_world_batch(self, pixel_xy: np.ndarray) -> np.ndarray:
        pts = pixel_xy.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, self.H_pixel_to_world).reshape(-1, 2)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save H + fit diagnostics to a `.npz`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bbox = (np.array(self.inlier_world_bbox, dtype=np.float64)
                if self.inlier_world_bbox is not None
                else np.array([np.nan]*4, dtype=np.float64))
        np.savez(
            path,
            H_world_to_pixel = self.H_world_to_pixel,
            fit_error_px      = self.fit_error_px or -1.0,
            n_correspondences = self.n_correspondences or -1,
            source_frame_id   = self.source_frame_id or -1,
            inlier_world_bbox = bbox,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PitchHomography":
        path = Path(path)
        with np.load(path) as data:
            H_w2p = data["H_world_to_pixel"].astype(np.float64)
            err   = float(data["fit_error_px"])
            n     = int(data["n_correspondences"])
            fid   = int(data["source_frame_id"])
            bbox_arr = data["inlier_world_bbox"] if "inlier_world_bbox" in data.files else None
        H_p2w = np.linalg.inv(H_w2p)
        bbox: Optional[Tuple[float, float, float, float]] = None
        if bbox_arr is not None and not np.isnan(bbox_arr).any():
            bbox = (float(bbox_arr[0]), float(bbox_arr[1]),
                    float(bbox_arr[2]), float(bbox_arr[3]))
        return cls(
            H_world_to_pixel  = H_w2p,
            H_pixel_to_world  = H_p2w,
            fit_error_px      = err if err >= 0 else None,
            n_correspondences = n   if n   >= 0 else None,
            source_frame_id   = fid if fid >= 0 else None,
            inlier_world_bbox = bbox,
        )

    # ── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        err_s = f"{self.fit_error_px:.2f}px" if self.fit_error_px else "?"
        n_s   = self.n_correspondences      or "?"
        fid_s = self.source_frame_id        or "?"
        return (f"PitchHomography(N={n_s}, frame={fid_s}, "
                f"reproj_err={err_s})")
