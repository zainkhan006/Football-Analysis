"""
Ball detection viewer — Feature 11 (L1 only, no Kalman, no touch).

Plays the SportsMOT clip frame-by-frame at 25 fps with the ball detected
and drawn as a bounding box.

Model used:
  A football-specific YOLOv8n fine-tuned on broadcast footage, downloaded
  once from a public URL into the local 'models/' folder.  If the download
  fails the script falls back to COCO YOLOv8n (class 32 = sports ball).

Controls while the window is open:
  Q or ESC  — quit
  SPACE     — pause / resume
"""

import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────

SEQUENCE_DIR = Path("dataset/train/v_gQNyhv8y0QY_c013")
FPS          = 25
CONF        = 0.15   # minimum confidence to consider a detection
MODEL_PATH  = Path("models/football-ball-detection.pt")
FRAME_LIMIT = 300    # process only first N frames (875 = full clip)

# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """
    Returns (model, ball_class_id, label).
    Uses Roboflow football-ball-detection.pt (fine-tuned on broadcast footage).
    Falls back to COCO YOLOv8n if the file is missing.
    """
    if MODEL_PATH.exists():
        model    = YOLO(str(MODEL_PATH))
        names    = model.names
        ball_cls = next((k for k, v in names.items() if "ball" in v.lower()), 0)
        print(f"[model] Loaded Roboflow football-ball-detection model")
        print(f"[model] Classes: {names}")
        print(f"[model] Ball class → {ball_cls} = '{names[ball_cls]}'")
        return model, ball_cls, "Roboflow-football"
    else:
        print(f"[model] WARNING: {MODEL_PATH} not found — falling back to COCO YOLOv8n")
        model = YOLO("yolov8n.pt")
        return model, 32, "COCO-fallback"


# ─── Frame loader ─────────────────────────────────────────────────────────────

def get_frame_paths(seq_dir: Path) -> list:
    img_dir = seq_dir / "img1"
    paths   = sorted(img_dir.glob("*.jpg"))
    return paths


# ─── Draw helpers ─────────────────────────────────────────────────────────────

def _inside_player(cx: int, cy: int, player_bboxes: list) -> bool:
    """Return True if (cx,cy) falls inside any player bounding box."""
    for (px, py, pw, ph) in player_bboxes:
        if px <= cx <= px + pw and py <= cy <= py + ph:
            return True
    return False


def draw_best_detection(frame: np.ndarray, results, ball_cls: int,
                        conf_thresh: float,
                        player_bboxes: list | None = None) -> tuple:
    """
    Pick the single highest-confidence ball detection that does NOT sit
    inside a player bounding box, and draw it.
    Returns (annotated_frame, detected: bool, conf: float).
    """
    best_conf = 0.0
    best_box  = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id != ball_cls or conf < conf_thresh or conf <= best_conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if player_bboxes and _inside_player(cx, cy, player_bboxes):
            continue   # centre is inside a player — almost certainly a FP
        best_conf = conf
        best_box  = box

    if best_box is None:
        return frame, False, 0.0

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
    cv2.putText(frame, f"ball {best_conf:.2f}",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)
    return frame, True, best_conf


# ─── Main viewer ──────────────────────────────────────────────────────────────

def main():
    model, ball_cls, model_label = load_model()

    frame_paths = get_frame_paths(SEQUENCE_DIR)
    if not frame_paths:
        print(f"[error] No frames found in {SEQUENCE_DIR / 'img1'}")
        sys.exit(1)

    frame_paths = [p for p in frame_paths if int(p.stem) <= FRAME_LIMIT]
    total     = len(frame_paths)
    total_det = 0
    paused    = False

    print(f"[viewer] {total} frames | model: {model_label} | conf>={CONF}")
    print("[viewer] Q/ESC = quit  |  SPACE = pause\n")

    # Load GT player boxes for false-positive filtering
    gt = defaultdict(list)
    gt_file = SEQUENCE_DIR / "gt" / "gt.txt"
    if gt_file.exists():
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    fid = int(parts[0])
                    gt[fid].append((int(parts[2]), int(parts[3]),
                                    int(parts[4]), int(parts[5])))
        print(f"[viewer] Loaded GT player boxes for {len(gt)} frames (FP filter active)")
    else:
        print("[viewer] No GT file found — FP filter disabled")

    cv2.namedWindow("Ball Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Detection", 1280, 720)

    for i, img_path in enumerate(frame_paths):
        frame_num     = int(img_path.stem)
        frame         = cv2.imread(str(img_path))
        if frame is None:
            continue

        player_bboxes = gt.get(frame_num, [])
        results               = model(frame, verbose=False)[0]
        frame, detected, conf = draw_best_detection(
            frame, results, ball_cls, CONF, player_bboxes)
        if detected:
            total_det += 1

        # HUD
        cv2.putText(frame, f"Frame {frame_num}/{total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Ball: {'conf=' + f'{conf:.2f}' if detected else 'not detected'}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if detected else (100, 100, 100), 2)
        cv2.putText(frame, f"Total detected: {total_det}/{i+1} frames",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Ball Detection", frame)

        while True:
            key = cv2.waitKey(1) & 0xFF   # 1ms wait — display rate = inference rate
            if key in (ord('q'), 27):
                print(f"\n[viewer] Quit. Ball detected in {total_det}/{i+1} frames.")
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            if not paused:
                break

    print(f"\n[viewer] Done. Ball detected in {total_det}/{total} frames "
          f"({100*total_det/total:.1f}%)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
