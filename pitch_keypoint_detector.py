"""
Pitch keypoint detector.

Wraps the Roboflow `football-pitch-detection.pt` YOLOv8-pose model so
the rest of the project doesn't have to know about Ultralytics quirks.

Each call returns up to 32 keypoints in canonical order (matching
`pitch_config.PitchConfig.vertices_m`), each with an (x_px, y_px,
confidence) triple.  Callers can filter on confidence themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("models/football-pitch-detection.pt")


# ─── Roboflow keypoint ordering ──────────────────────────────────────────────
#
# Empirical finding (validated by `debug_keypoints.py` on a kickoff frame):
# the YOLO model emits keypoints DIRECTLY in `vertices` order — i.e. YOLO
# output position `i` corresponds to vertex (i + 1) in the FIFA pitch model.
# (Despite the `labels` list in the Roboflow source code suggesting a
# permutation, that list is metadata for *visualisation*, not for the
# tensor channel order the trained model uses.)
#
# Identity permutation kept as a constant so future tweaks have one place
# to live.
ROBOFLOW_KP_TO_VERTEX_IDX: tuple[int, ...] = tuple(range(32))


@dataclass
class DetectedKeypoint:
    index:      int          # 0-based index into pitch_config vertices
    pixel_xy:   Tuple[float, float]
    confidence: float


@dataclass
class PitchKeypointResult:
    keypoints: List[DetectedKeypoint]   # always exactly 32 entries
    image_shape: Tuple[int, int]        # (H, W)

    def confident(self, threshold: float = 0.5) -> List[DetectedKeypoint]:
        return [kp for kp in self.keypoints if kp.confidence >= threshold]

    def num_confident(self, threshold: float = 0.5) -> int:
        return sum(1 for kp in self.keypoints if kp.confidence >= threshold)


class PitchKeypointDetector:
    """Single-frame inference wrapper around the Roboflow pitch YOLO model."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        imgsz: int = 640,
        verbose: bool = True,
    ):
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Pitch keypoint model not found at {path}. "
                f"Download with:\n"
                f"  python -c \"import gdown; gdown.download("
                f"'https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf', "
                f"'{path}', quiet=False)\""
            )
        self.model_path = path
        self.imgsz      = int(imgsz)
        self.model      = YOLO(str(path))
        if self.model.task != "pose":
            raise RuntimeError(
                f"Expected a 'pose' model; got task='{self.model.task}'"
            )
        if verbose:
            print(f"[PitchKeypointDetector] Loaded {path}")
            try:
                kpt_shape = self.model.model.kpt_shape
                print(f"[PitchKeypointDetector] kpt_shape = {kpt_shape}")
            except AttributeError:
                pass

    # ── Inference ────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.30,
    ) -> PitchKeypointResult:
        """
        Run pose inference on one BGR frame.  Returns 32 DetectedKeypoint
        entries (low-confidence ones still included so the caller can
        decide what to filter).
        """
        res = self.model(frame, imgsz=self.imgsz, conf=conf_threshold,
                         verbose=False)[0]

        H, W = frame.shape[:2]
        if res.keypoints is None or res.keypoints.conf is None \
                or len(res.keypoints.xy) == 0:
            empty = [
                DetectedKeypoint(i, (float("nan"), float("nan")), 0.0)
                for i in range(32)
            ]
            return PitchKeypointResult(keypoints=empty, image_shape=(H, W))

        # The model emits one "pitch" instance per frame; we take the first.
        # xy: (1, 32, 2)   conf: (1, 32)
        # NB: indexed by YOLO OUTPUT POSITION (0..31), which is NOT the same
        # as the canonical vertex order — see ROBOFLOW_KP_TO_VERTEX_IDX above.
        xy   = res.keypoints.xy.cpu().numpy()[0]
        conf = res.keypoints.conf.cpu().numpy()[0]

        out: List[DetectedKeypoint] = []
        for yolo_idx in range(32):
            vertex_idx = ROBOFLOW_KP_TO_VERTEX_IDX[yolo_idx]
            x, y = float(xy[yolo_idx, 0]), float(xy[yolo_idx, 1])
            c    = float(conf[yolo_idx])
            out.append(DetectedKeypoint(vertex_idx, (x, y), c))

        # Sort by canonical vertex index so callers get them in pitch order
        # (otherwise vertex 14 and 19 come out at positions 30, 31).
        out.sort(key=lambda kp: kp.index)
        return PitchKeypointResult(keypoints=out, image_shape=(H, W))


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2

    det = PitchKeypointDetector()
    img = cv2.imread("dataset/train/v_dw7LOz17Omg_c053/img1/000001.jpg")
    if img is None:
        raise SystemExit("test image not found")
    res = det.detect(img)
    print(f"image: {res.image_shape}")
    print(f"confident keypoints (>=0.5): {res.num_confident(0.5)}")
    for kp in res.confident(0.5):
        print(f"  kp{kp.index+1:02d}  px=({kp.pixel_xy[0]:7.1f},"
              f" {kp.pixel_xy[1]:7.1f})  conf={kp.confidence:.2f}")
