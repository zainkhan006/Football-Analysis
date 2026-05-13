"""
Per-frame homography via classical line extraction.

Instead of using one static H for the entire clip, this module refines H
every frame using the white pitch lines visible in that frame:

    1. HSV segmentation → binary mask of white-on-green line pixels
    2. Distance transform of the mask
    3. Levenberg-Marquardt: nudge previous frame's H so canonical pitch
       samples land on detected lines (minimise distance-transform cost)

The YOLO keypoint H (from homographies/*.npz) is used ONLY as the seed
for frame 1. Every subsequent frame refines from the previous frame's H
with a prior weight that prevents wild drift.

Usage:
    from frame_homography import FrameByFrameHomography

    fbh = FrameByFrameHomography.from_saved("homographies/v_xxx.npz")
    for frame in frames:
        H = fbh.update(frame)       # returns PitchHomography for this frame
        wx, wy = H.pixel_to_world(px, py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from homography import PitchHomography
from refine_homography import (
    build_line_mask,
    refine_homography,
    sample_canonical_pitch,
    RefineInfo,
)


class FrameByFrameHomography:
    """
    Tracks the pitch homography frame-by-frame using line extraction.

    Parameters
    ----------
    seed_H_w2p : np.ndarray
        Initial 3x3 world-to-pixel homography (from YOLO keypoint fit).
    prior_weight : float
        How strongly to anchor each frame's H to the previous frame's H.
        Higher = more stable, slower to adapt.  Lower = more responsive.
        0.5 is a good default for broadcast football pans.
    max_px : float
        Huber cap for the distance-transform cost. Samples landing further
        than this from any line pixel are capped (don't dominate the fit).
    max_iter : int
        LM iteration budget per frame.
    step_m : float
        Spacing (metres) between canonical pitch samples. 1.0 = ~1000 pts
        covering every pitch line, circle, and box edge.
    skip_refine_every : int
        Only refine every Nth frame to save CPU. Intermediate frames reuse
        the last refined H. Set to 1 to refine every frame.
    """

    def __init__(
        self,
        seed_H_w2p: np.ndarray,
        prior_weight: float = 0.5,
        max_px: float = 30.0,
        max_iter: int = 150,
        step_m: float = 1.0,
        skip_refine_every: int = 1,
        inlier_world_bbox: Optional[Tuple[float, float, float, float]] = None,
        source_frame_id: Optional[int] = None,
    ):
        self._H_w2p = seed_H_w2p.astype(np.float64).copy()
        self._H_w2p /= self._H_w2p[2, 2]  # normalise

        self.prior_weight = float(prior_weight)
        self.max_px = float(max_px)
        self.max_iter = int(max_iter)
        self.skip_every = max(1, int(skip_refine_every))

        self._samples = sample_canonical_pitch(step_m=step_m)
        self._inlier_world_bbox = inlier_world_bbox
        self._source_frame_id = source_frame_id

        # Stats from the last refinement
        self.last_info: Optional[RefineInfo] = None
        self.last_cost: float = -1.0
        self._frame_count: int = 0
        self._refined_count: int = 0

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        npz_path: str | Path,
        **kwargs,
    ) -> "FrameByFrameHomography":
        """Load the seed H from a saved homography .npz file."""
        H = PitchHomography.load(npz_path)
        return cls(
            seed_H_w2p=H.H_world_to_pixel,
            inlier_world_bbox=H.inlier_world_bbox,
            source_frame_id=H.source_frame_id,
            **kwargs,
        )

    # ── Per-frame update ──────────────────────────────────────────────

    def update(self, frame_bgr: np.ndarray) -> PitchHomography:
        """
        Refine H for this frame using line extraction.

        Returns a PitchHomography with the updated H. If refinement fails
        or this frame is skipped (skip_refine_every > 1), returns H from
        the previous successful refinement.
        """
        self._frame_count += 1

        # Skip frames if configured (reuse last H)
        if self._frame_count % self.skip_every != 0:
            return self._build_homography()

        # 1. Extract white pitch lines from this frame
        mask = build_line_mask(frame_bgr)

        # Check if we have enough line pixels to be useful
        line_ratio = np.count_nonzero(mask) / max(mask.size, 1)
        if line_ratio < 0.001:
            # Almost no lines detected — keep previous H
            return self._build_homography()

        # 2. Refine H using distance-transform alignment
        try:
            H_new, info = refine_homography(
                H_init=self._H_w2p,
                line_mask=mask,
                canonical_samples=self._samples,
                max_px=self.max_px,
                max_iter=self.max_iter,
                prior_weight=self.prior_weight,
            )
            self._H_w2p = H_new
            self.last_info = info
            self.last_cost = info.cost_after
            self._refined_count += 1
        except Exception:
            # Refinement failed — keep previous H
            pass

        return self._build_homography()

    # ── Accessors ─────────────────────────────────────────────────────

    def current_homography(self) -> PitchHomography:
        """Get the current H without updating."""
        return self._build_homography()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def refined_count(self) -> int:
        return self._refined_count

    def _build_homography(self) -> PitchHomography:
        H_p2w = np.linalg.inv(self._H_w2p)
        return PitchHomography(
            H_world_to_pixel=self._H_w2p.copy(),
            H_pixel_to_world=H_p2w,
            inlier_world_bbox=self._inlier_world_bbox,
            source_frame_id=self._source_frame_id,
        )
