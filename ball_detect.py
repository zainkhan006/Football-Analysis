"""
Feature 11 — Ball Detection & Tracking  (CPU-optimised)

Phase L1: YOLO ball detection (default = Roboflow football-ball-detection.pt)
Phase L2: Kalman filter tracking with prediction-guided ROI search

Speed strategy on CPU (no GPU, no fine-tuning of heavy model):
  1. Auto-loads the Roboflow football-ball-detection.pt if present
     (massively better ball recall than COCO YOLOv8n).
  2. Once the ball is *locked*, search a small ROI around the Kalman
     prediction (default 320 px) instead of the full 1280x720 frame —
     this is the dominant speedup (8-16x fewer pixels per frame).
  3. Configurable inference image size (`imgsz`).  Default 960 for
     full-frame search, 320 for ROI search.
  4. Periodic full-frame re-detect (every `full_frame_every` frames)
     to recover from drift if the ROI ever loses the ball.
  5. Optional player-bbox false-positive filter: detections whose
     centre lies inside the torso of any tracked player are rejected
     (jerseys/heads are the most common ball false positives).
  6. Ball-size sanity gate: reject boxes whose largest side is outside
     a plausible pixel range — the ball is small (~8-25 px in 720p
     broadcast).
  7. Detections are scored by  `conf + alpha * proximity_bonus(pred)`,
     so when locked we prefer the candidate closest to where physics
     says the ball *should* be.

Backwards-compatible API:
    detector = BallDetector()
    out = detector.detect_ball(frame)                 # raw L1
    out = detector.track_ball(frame)                  # L1 + Kalman
    out = detector.track_ball(frame, player_bboxes)   # + FP filter
    detector.draw_ball(frame, out)
    detector.reset()

Output dict per frame:
    {
        "frame_id":     int,
        "ball_px":      float | None,
        "ball_py":      float | None,
        "confidence":   float,
        "is_estimated": bool,    # True = Kalman fill-in, False = real det
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_ROBOFLOW_PATH = Path("models/football-ball-detection.pt")
DEFAULT_FALLBACK_PATH = Path("yolov8n.pt")     # COCO, sports ball = class 32


# ─── Kalman filter ────────────────────────────────────────────────────────────

def _build_kalman_filter() -> cv2.KalmanFilter:
    """4-state constant-velocity Kalman filter [x, y, dx, dy]."""
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # Higher process noise on velocity than position — the ball changes
    # direction sharply (kicks, deflections), so we trust velocity less.
    kf.processNoiseCov = np.diag(
        np.array([1e-2, 1e-2, 5e-2, 5e-2], dtype=np.float32)
    )

    # Detector pixel noise is small (a few px of jitter).
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

    kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
    return kf


# ─── Ball Detector ────────────────────────────────────────────────────────────

class BallDetector:
    """Phase L1 + L2 ball detector/tracker, optimised for CPU inference."""

    COCO_BALL_CLASS_ID = 32        # 'sports ball' in COCO

    # Plausible ball pixel size in 720p broadcast. Adjust if your input
    # resolution differs significantly.
    DEFAULT_MIN_BALL_PX = 4
    DEFAULT_MAX_BALL_PX = 60

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.15,
        imgsz_full: int = 640,
        imgsz_roi:  int = 320,
        roi_size:   int = 320,
        full_frame_every: int = 0,         # 0 = disabled (was 30 — caused jumps)
        max_miss_frames:  int = 10,
        min_ball_px:      int = DEFAULT_MIN_BALL_PX,
        max_ball_px:      int = DEFAULT_MAX_BALL_PX,
        proximity_bonus_alpha: float = 0.3,
        proximity_bonus_radius_px: float = 200.0,
        # ── Sticky possession (Q2 fix) ──────────────────────────────────
        possession_max_dist_px: float = 30.0,    # how close to bbox to count
        possession_search_margin_px: int = 20,   # expand bbox before searching
        # ── Hard gating on full-frame detections (Q1 fix) ───────────────
        gate_radius_initial_px:    float = 200.0,   # gate at miss=0..1
        gate_radius_extended_px:   float = 400.0,   # gate at miss=2..4
        gate_radius_release_after: int   = 5,       # after this many misses,
                                                    #   gate is removed
        # ── Throttle full-frame fallback (Q3 fix) ───────────────────────
        full_frame_on_miss_every: int = 3,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        model_path
            Path to a YOLO weights file.  If None, auto-loads
            `models/football-ball-detection.pt` if present, else falls
            back to `yolov8n.pt` (COCO).
        conf_threshold
            Minimum YOLO confidence accepted.  Lower = more recall, more
            false positives (which the FP filter helps remove).
        imgsz_full / imgsz_roi
            Inference image size for full-frame and ROI searches.
            Smaller = faster.  Must be a multiple of 32.
        roi_size
            Side length (px) of the ROI cropped around the Kalman
            prediction when the ball is locked.
        full_frame_every
            Even when locked, perform a full-frame search every N frames
            as a drift safety net.  Set to 0 to disable.
        max_miss_frames
            After this many consecutive missed detections, drop the
            track (Kalman re-initialises on next real detection).
        min_ball_px / max_ball_px
            Ball-size sanity gate (largest side of the bounding box,
            measured in original-frame pixels).
        proximity_bonus_alpha
            Weight of the prediction-proximity bonus when ranking
            candidates against confidence.  Set to 0 to disable.
        proximity_bonus_radius_px
            Distance (px) at which the proximity bonus becomes 0.
        """
        from ultralytics import YOLO

        self.conf_threshold = float(conf_threshold)
        self.imgsz_full = int(imgsz_full)
        self.imgsz_roi  = int(imgsz_roi)
        self.roi_size   = int(roi_size)
        self.full_frame_every = int(full_frame_every)
        self.max_miss_frames  = int(max_miss_frames)
        self.min_ball_px = int(min_ball_px)
        self.max_ball_px = int(max_ball_px)
        self.alpha       = float(proximity_bonus_alpha)
        self.gate_radius = float(proximity_bonus_radius_px)
        self.verbose     = bool(verbose)

        # Sticky-possession config
        self.possession_max_dist_px      = float(possession_max_dist_px)
        self.possession_search_margin_px = int(possession_search_margin_px)

        # Hard-gate config (Q1)
        self.gate_radius_initial_px    = float(gate_radius_initial_px)
        self.gate_radius_extended_px   = float(gate_radius_extended_px)
        self.gate_radius_release_after = int(gate_radius_release_after)

        # Full-frame throttle config (Q3)
        self.full_frame_on_miss_every = max(1, int(full_frame_on_miss_every))

        # ── Resolve model ──────────────────────────────────────────────
        resolved = self._resolve_model_path(model_path)
        self.model_path = resolved
        self.model = YOLO(str(resolved))
        self.ball_class_id = self._infer_ball_class_id(self.model.names)
        if self.verbose:
            print(f"[BallDetector] Model       : {resolved}")
            print(f"[BallDetector] Classes     : {self.model.names}")
            print(f"[BallDetector] Ball class  : "
                  f"{self.ball_class_id} = "
                  f"'{self.model.names[self.ball_class_id]}'")
            print(f"[BallDetector] imgsz_full  : {self.imgsz_full}")
            print(f"[BallDetector] imgsz_roi   : {self.imgsz_roi}  "
                  f"(roi={self.roi_size}px)")

        # ── Kalman state ───────────────────────────────────────────────
        self.kf = _build_kalman_filter()
        self.kf_initialised = False
        self.miss_count     = 0
        self.frame_counter  = 0
        self._last_full_frame = -10**9

        # ── Possessor state (Q2) ───────────────────────────────────────
        # Last player bbox the ball was seen near (in original-frame px).
        # Used as a priority search region while the ball is missing.
        self.possessor_bbox: Optional[tuple] = None
        self.possessor_age:  int = 0    # frames since possessor was set

    # ── Setup helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_model_path(model_path: Optional[str | Path]) -> Path:
        if model_path is not None:
            p = Path(model_path)
            if p.exists():
                return p
            # Treat as a model name to be downloaded by ultralytics
            return p
        if DEFAULT_ROBOFLOW_PATH.exists():
            return DEFAULT_ROBOFLOW_PATH
        return DEFAULT_FALLBACK_PATH

    @staticmethod
    def _infer_ball_class_id(names: dict) -> int:
        for k, v in names.items():
            if "ball" in str(v).lower():
                return int(k)
        # fall back to COCO id
        return BallDetector.COCO_BALL_CLASS_ID

    # ── Public: L1 (single frame, no Kalman) ───────────────────────────────

    def detect_ball(
        self,
        frame: np.ndarray,
        player_bboxes: Optional[Iterable[tuple]] = None,
    ) -> dict:
        """Run a full-frame detection and return the best ball candidate."""
        H, W = frame.shape[:2]
        det = self._detect_in_region(
            frame, 0, 0, W, H,
            pred_xy=None,
            player_bboxes=player_bboxes,
            imgsz=self.imgsz_full,
        )
        return {
            "ball_px":      det["ball_px"],
            "ball_py":      det["ball_py"],
            "confidence":   det["confidence"],
            "is_estimated": False,
        }

    # ── Public: L2 (detection + Kalman with ROI search) ────────────────────

    def track_ball(
        self,
        frame: np.ndarray,
        player_bboxes: Optional[Iterable[tuple]] = None,
    ) -> dict:
        """
        Detect + track the ball in `frame`.

        `player_bboxes` (optional): iterable of (x, y, w, h) tuples for
        all currently-tracked players.  Used to filter false positives
        that fall inside a player's torso AND, when the ball is missing,
        to identify the most likely possessor's bbox to search inside.
        """
        self.frame_counter += 1
        H, W = frame.shape[:2]
        player_bboxes = list(player_bboxes) if player_bboxes else []

        # 1. Kalman predict step
        pred_xy: Optional[tuple[float, float]] = None
        if self.kf_initialised:
            pred = self.kf.predict()
            pred_xy = (float(pred[0]), float(pred[1]))

        # 2. Pick a search strategy
        force_full = (
            self.full_frame_every > 0
            and (self.frame_counter % self.full_frame_every == 0)
        )
        do_roi = (
            self.kf_initialised
            and pred_xy is not None
            and self.miss_count == 0
            and not force_full
        )

        det = {"ball_px": None, "ball_py": None, "confidence": 0.0}

        if do_roi:
            # 2a. ROI search around Kalman prediction (cheap, normal case)
            det = self._search_roi(frame, pred_xy, player_bboxes)

            # 2b. If ROI missed, try the possessor's bbox (Q2 — sticky
            #     possession). FP filter is suspended for that bbox so a
            #     ball at the player's feet can finally be detected.
            if det["ball_px"] is None:
                det = self._search_possessor(frame, pred_xy, player_bboxes)

            # 2c. If still missing AND we should rescan this frame
            #     (Q3 — full-frame fallback throttled to every Nth miss),
            #     do a full-image search with hard gating.
            if det["ball_px"] is None and self._should_rescan_now():
                det = self._search_full_gated(
                    frame, pred_xy, player_bboxes,
                    miss_count_at_frame=self.miss_count + 1,
                )
                self._last_full_frame = self.frame_counter
        else:
            # First-time acquisition or `force_full` periodic re-anchor.
            # Still gate against Kalman prediction (if any) to prevent
            # the well-known "snap-to-distant-blob" failure (Q1).
            det = self._search_full_gated(
                frame, pred_xy, player_bboxes,
                miss_count_at_frame=self.miss_count,
            )
            self._last_full_frame = self.frame_counter

        # 3. Detection succeeded → update Kalman + possessor
        if det["ball_px"] is not None:
            self._update_kalman_with(det["ball_px"], det["ball_py"])
            self._update_possessor(det["ball_px"], det["ball_py"], player_bboxes)
            self.miss_count = 0
            return {
                "frame_id":     self.frame_counter,
                "ball_px":      det["ball_px"],
                "ball_py":      det["ball_py"],
                "confidence":   det["confidence"],
                "is_estimated": False,
            }

        # 4. Detection failed
        if not self.kf_initialised:
            return self._empty_result()

        self.miss_count += 1
        self.possessor_age += 1

        if self.miss_count > self.max_miss_frames:
            self.reset(full=False)        # drop track, keep counters at 0
            return self._empty_result()

        # Use Kalman prediction as the reported (estimated) position
        return {
            "frame_id":     self.frame_counter,
            "ball_px":      pred_xy[0],
            "ball_py":      pred_xy[1],
            "confidence":   0.0,
            "is_estimated": True,
        }

    # ── Internal: search strategies ────────────────────────────────────────

    def _search_roi(self, frame, pred_xy, player_bboxes) -> dict:
        """Search a small box around the Kalman prediction."""
        H, W = frame.shape[:2]
        cx, cy = pred_xy
        half = self.roi_size // 2
        rx0 = max(0, int(cx - half));  ry0 = max(0, int(cy - half))
        rx1 = min(W, int(cx + half));  ry1 = min(H, int(cy + half))
        return self._detect_in_region(
            frame, rx0, ry0, rx1, ry1,
            pred_xy=pred_xy,
            player_bboxes=player_bboxes,
            imgsz=self.imgsz_roi,
        )

    def _search_possessor(self, frame, pred_xy, player_bboxes) -> dict:
        """
        If the prediction is inside-or-very-near a player bbox, search
        inside that bbox with the FP filter suspended for that one
        player.  This is the fix for "ball disappears once a player
        gets it" — the FP filter was rejecting in-bbox detections.
        """
        if not player_bboxes or pred_xy is None:
            return {"ball_px": None, "ball_py": None, "confidence": 0.0}

        possessor = self._find_possessor_bbox(pred_xy, player_bboxes)
        if possessor is None and self.possessor_bbox is not None \
                and self.possessor_age <= 2:
            # Carry-over: use the most recent possessor bbox briefly.
            possessor = self.possessor_bbox

        if possessor is None:
            return {"ball_px": None, "ball_py": None, "confidence": 0.0}

        H, W = frame.shape[:2]
        x, y, w, h = possessor[0], possessor[1], possessor[2], possessor[3]
        m = self.possession_search_margin_px
        rx0 = max(0, int(x - m));  ry0 = max(0, int(y - m))
        rx1 = min(W, int(x + w + m));  ry1 = min(H, int(y + h + m))

        # Suspend FP filter for the possessor; keep filtering against
        # OTHER players (so we don't accidentally jump to a head).
        other_players = [bb for bb in player_bboxes
                         if not _bbox_equal(bb, possessor)]

        return self._detect_in_region(
            frame, rx0, ry0, rx1, ry1,
            pred_xy=pred_xy,
            player_bboxes=other_players,
            imgsz=self.imgsz_roi,
        )

    def _search_full_gated(
        self,
        frame, pred_xy, player_bboxes,
        miss_count_at_frame: int,
    ) -> dict:
        """Full-frame YOLO + hard distance gate against Kalman prediction."""
        H, W = frame.shape[:2]
        det = self._detect_in_region(
            frame, 0, 0, W, H,
            pred_xy=pred_xy,
            player_bboxes=player_bboxes,
            imgsz=self.imgsz_full,
        )
        # Hard gate (Q1) — only if we have a prediction to gate against
        if det["ball_px"] is None or pred_xy is None:
            return det
        if miss_count_at_frame >= self.gate_radius_release_after:
            return det                     # gate released → accept anything
        gate = (self.gate_radius_initial_px
                if miss_count_at_frame <= 1
                else self.gate_radius_extended_px)
        d = float(np.hypot(det["ball_px"] - pred_xy[0],
                           det["ball_py"] - pred_xy[1]))
        if d > gate:
            # Reject — don't let YOLO teleport the tracker.
            return {"ball_px": None, "ball_py": None, "confidence": 0.0}
        return det

    def _should_rescan_now(self) -> bool:
        """
        Q3 — throttle full-frame fallback.  We rescan on the FIRST miss
        of a streak (so brief 1-frame occlusions still recover quickly)
        and then every `full_frame_on_miss_every` misses after that.
        """
        next_miss = self.miss_count + 1
        if next_miss == 1:
            return True
        return (next_miss % self.full_frame_on_miss_every) == 0

    # ── Internal: state updates ────────────────────────────────────────────

    def _update_kalman_with(self, x: float, y: float) -> None:
        meas = np.array([[x], [y]], dtype=np.float32)
        if not self.kf_initialised:
            self.kf.statePost = np.array(
                [[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.kf.statePre = self.kf.statePost.copy()
            self.kf_initialised = True
        else:
            self.kf.correct(meas)

    def _update_possessor(
        self,
        ball_x: float, ball_y: float,
        player_bboxes: list,
    ) -> None:
        """After every real detection, identify the closest player bbox."""
        if not player_bboxes:
            self.possessor_bbox = None
            self.possessor_age = 0
            return
        best, best_d = None, float("inf")
        for bb in player_bboxes:
            d = _point_to_bbox_distance(ball_x, ball_y, bb)
            if d < best_d:
                best_d, best = d, bb
        if best is not None and best_d <= self.possession_max_dist_px:
            self.possessor_bbox = tuple(best[:4])
            self.possessor_age = 0
        else:
            self.possessor_bbox = None
            self.possessor_age = 0

    @staticmethod
    def _find_possessor_bbox(
        pred_xy: tuple[float, float],
        player_bboxes: list,
        max_dist_px: float = 30.0,
    ) -> Optional[tuple]:
        """Return the player bbox the prediction is inside or very close to."""
        if not player_bboxes:
            return None
        best, best_d = None, float("inf")
        px, py = pred_xy
        for bb in player_bboxes:
            d = _point_to_bbox_distance(px, py, bb)
            if d < best_d:
                best_d, best = d, bb
        if best is not None and best_d <= max_dist_px:
            return best
        return None

    def _empty_result(self) -> dict:
        return {
            "frame_id":     self.frame_counter,
            "ball_px":      None,
            "ball_py":      None,
            "confidence":   0.0,
            "is_estimated": False,
        }

    # ── Region detection core ──────────────────────────────────────────────

    def _detect_in_region(
        self,
        frame: np.ndarray,
        x0: int, y0: int, x1: int, y1: int,
        pred_xy: Optional[tuple[float, float]],
        player_bboxes: Optional[Iterable[tuple]],
        imgsz: int,
    ) -> dict:
        """Run YOLO on `frame[y0:y1, x0:x1]` and return the best ball
        candidate (in *original-frame* pixel coordinates)."""

        if x1 <= x0 or y1 <= y0:
            return {"ball_px": None, "ball_py": None, "confidence": 0.0}

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return {"ball_px": None, "ball_py": None, "confidence": 0.0}

        results = self.model(
            crop,
            imgsz=imgsz,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        candidates: list[tuple[float, float, float]] = []  # (cx, cy, conf)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != self.ball_class_id:
                continue
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            bw, bh = (bx2 - bx1), (by2 - by1)

            # Ball-size sanity gate (largest side, in original-frame px)
            largest = max(bw, bh)
            if largest < self.min_ball_px or largest > self.max_ball_px:
                continue

            cx = (bx1 + bx2) / 2.0 + x0
            cy = (by1 + by2) / 2.0 + y0

            # Player-bbox FP filter
            if player_bboxes and self._inside_any_player(
                cx, cy, player_bboxes, shrink=0.20
            ):
                continue

            candidates.append((cx, cy, conf))

        if not candidates:
            return {"ball_px": None, "ball_py": None, "confidence": 0.0}

        # Score: conf + alpha * proximity_bonus(prediction)
        if pred_xy is not None and self.alpha > 0.0:
            px, py = pred_xy
            r = self.gate_radius

            def score(c):
                cx, cy, cf = c
                d = float(np.hypot(cx - px, cy - py))
                bonus = max(0.0, 1.0 - d / r)
                return cf + self.alpha * bonus
        else:
            def score(c):
                return c[2]

        cx, cy, cf = max(candidates, key=score)
        return {"ball_px": cx, "ball_py": cy, "confidence": cf}

    @staticmethod
    def _inside_any_player(
        cx: float, cy: float,
        player_bboxes: Iterable[tuple],
        shrink: float = 0.0,
    ) -> bool:
        """
        Returns True if (cx, cy) is inside any player bbox, optionally
        shrunk by `shrink` fraction on each side (so 'near the feet'
        detections are NOT rejected — only 'on the torso/head' ones).
        """
        for bb in player_bboxes:
            x, y, w, h = bb[0], bb[1], bb[2], bb[3]
            sx = w * shrink * 0.5
            sy = h * shrink * 0.5
            if (x + sx) <= cx <= (x + w - sx) and \
               (y + sy) <= cy <= (y + h - sy):
                return True
        return False

    # ── Misc ───────────────────────────────────────────────────────────────

    def reset(self, full: bool = True):
        """Clear tracker state.

        full=True  → wipe everything (call between independent clips)
        full=False → drop only the lock; keep frame counter so timing
                     in the caller is preserved (used internally when
                     max_miss_frames is exceeded)
        """
        self.kf = _build_kalman_filter()
        self.kf_initialised = False
        self.miss_count = 0
        self.possessor_bbox = None
        self.possessor_age = 0
        if full:
            self.frame_counter = 0
            self._last_full_frame = -10**9

    def draw_ball(self, frame: np.ndarray, ball_result: dict) -> np.ndarray:
        """Annotate `frame` in-place with the ball position."""
        if ball_result["ball_px"] is None:
            return frame
        px = int(ball_result["ball_px"])
        py = int(ball_result["ball_py"])
        is_est = ball_result["is_estimated"]
        colour = (0, 165, 255) if is_est else (0, 255, 255)
        label  = "ball~" if is_est else "ball"
        conf_s = f"{ball_result['confidence']:.2f}" if not is_est else "est"
        cv2.circle(frame, (px, py), 12, colour, 2)
        cv2.circle(frame, (px, py), 3,  colour, -1)
        cv2.putText(frame, f"{label} {conf_s}", (px + 14, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return frame


# ─── Module-level geometry helpers ────────────────────────────────────────────

def _point_to_bbox_distance(px: float, py: float, bbox) -> float:
    """Euclidean distance from (px, py) to the bbox interior (0 if inside)."""
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return float((dx * dx + dy * dy) ** 0.5)


def _bbox_equal(a, b) -> bool:
    return (a[0] == b[0] and a[1] == b[1]
            and a[2] == b[2] and a[3] == b[3])


# ─── ONNX export helper (optional, run once) ──────────────────────────────────

def export_to_onnx(
    model_path: str | Path = DEFAULT_ROBOFLOW_PATH,
    imgsz: int = 640,
    half: bool = False,
) -> Path:
    """
    Export a YOLO `.pt` weights file to ONNX (experimental).

    ⚠ On the test hardware (Intel i7-1255U) the dynamic-shape ONNX
    export was actually *slower* than the PyTorch backend, because the
    BallDetector mixes 640x640 full-frame and 320x320 ROI inferences,
    which forces a dynamic ONNX graph — and the resulting kernels are
    less optimised than torch's native ones.
    Static-shape ONNX (dynamic=False) is faster *but* cannot accept
    the smaller ROI crops, defeating the main optimisation here.

    Kept as a hook in case a future model is small enough that
    quantised ONNX (INT8) becomes worthwhile.
    """
    from ultralytics import YOLO
    model_path = Path(model_path)
    model = YOLO(str(model_path))
    onnx_path = model.export(format="onnx", imgsz=imgsz, half=half, dynamic=True)
    print(f"[export] ONNX written → {onnx_path}")
    return Path(onnx_path)


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== BallDetector self-test ===")
    detector = BallDetector()

    # Blank frame — just confirms no crash
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    r = detector.track_ball(dummy)
    print(f"Blank frame result: {r}")

    # If a video path is provided, run on first 100 frames
    if len(sys.argv) > 1:
        cap = cv2.VideoCapture(sys.argv[1])
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.track_ball(frame)
            detector.draw_ball(frame, result)
            tag = "[EST]" if result["is_estimated"] else "     "
            if result["ball_px"] is not None:
                print(f"  Frame {result['frame_id']:3d} {tag} "
                      f"pos=({result['ball_px']:6.1f},{result['ball_py']:6.1f}) "
                      f"conf={result['confidence']:.2f}")
            else:
                print(f"  Frame {result['frame_id']:3d}        not detected")
            cv2.imshow("Ball Tracking", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    print("Self-test passed.")
