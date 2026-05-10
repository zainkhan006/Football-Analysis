"""
Referee Detection from YOLO - GT Mismatch Analysis
===================================================
Logic:
1. Get team colors from Feature 8 clusters (home/away centroids)
2. Run YOLO detection and GT comparison
3. Find YOLO boxes that DON'T match any GT box
4. Calculate center distance from each unmatched YOLO box to all GT box centers
5. The YOLO box with lowest average distance to GT boxes = Referee candidate
6. Validate candidate: referee cannot be at most bottom or most top compared to GT boxes
7. Output side-by-side image:
   - LEFT: All YOLO boxes NOT in GT
   - RIGHT: Referee box if validated, otherwise blank (no referee)

Usage
-----
    python referee_detection.py --seq v_HdiyOtliFiw_c003 --frames 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

import cv2
from ultralytics import YOLO

# Import Feature 8 components
import sys
sys.path.append(str(Path(__file__).parent))
from feature_8_final import (
    FeatureConfig, crop_jersey, _describe, _kmeans_robust,
    _load_all_gt, _collect_frame_paths, _frame_path_for_idx
)


# ══════════════════════════════════════════════════════════════════════
# Team Color Extraction from Feature 8 Clusters
# ══════════════════════════════════════════════════════════════════════

def get_team_centroids_from_sequence(
    seq_name: str,
    frame_indices: List[int],
    cfg: FeatureConfig = FeatureConfig(),
    anchor_frames: int = 5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run Feature 8 logic to get team centroids.
    
    Returns
    -------
    home_centroid, away_centroid
    """
    here = Path(__file__).parent
    img_dir = here / "videos" / seq_name / "img1"
    gt_path = here / "videos" / seq_name / "gt" / "gt.txt"
    
    if not img_dir.is_dir() or not gt_path.is_file():
        print(f"Cannot find sequence {seq_name}")
        return None, None
    
    all_frame_paths = _collect_frame_paths(str(img_dir))
    frame_gt = _load_all_gt(str(gt_path))
    
    # Collect anchor features
    anchor_feats: List[np.ndarray] = []
    frames_for_anchor = frame_indices[:anchor_frames]
    
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
    
    if len(anchor_feats) < cfg.min_tracks_for_kmeans:
        print(f"Not enough anchor features ({len(anchor_feats)})")
        return None, None
    
    # Get team centroids
    feat_matrix = np.stack(anchor_feats)
    _, centroids, _ = _kmeans_robust(feat_matrix, cfg)
    
    return centroids[0], centroids[1]


# ══════════════════════════════════════════════════════════════════════
# YOLO vs GT Matching Helpers
# ══════════════════════════════════════════════════════════════════════

def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union."""
    ax1, ay1, aw, ah = boxA
    ax2, ay2 = ax1 + aw, ay1 + ah
    
    bx1, by1, bw, bh = boxB
    bx2, by2 = bx1 + bw, by1 + bh
    
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Return (cx, cy) of bounding box."""
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def get_box_y_range(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Return (y_min, y_max) of bounding box."""
    _, y, _, h = box
    return (y, y + h)


def calculate_average_distance_to_gt(
    yolo_box: Tuple[int, int, int, int],
    gt_boxes: List[Tuple[int, int, int, int]]
) -> float:
    """
    Calculate average Euclidean distance from YOLO box center to all GT box centers.
    """
    yolo_cx, yolo_cy = get_box_center(yolo_box)
    
    if not gt_boxes:
        return float('inf')
    
    total_distance = 0.0
    for gt_box in gt_boxes:
        gt_cx, gt_cy = get_box_center(gt_box)
        distance = np.sqrt((yolo_cx - gt_cx) ** 2 + (yolo_cy - gt_cy) ** 2)
        total_distance += distance
    
    return total_distance / len(gt_boxes)


def find_unmatched_yolo_boxes(
    gt_boxes: List[Tuple[int, int, int, int]],
    yolo_boxes: List[Tuple[int, int, int, int]],
    iou_thresh: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """
    Find YOLO boxes that don't match any GT box.
    Returns list of unmatched YOLO boxes.
    """
    matched_yolo = set()
    
    # Greedy matching
    for gt_idx, gt_box in enumerate(gt_boxes):
        best_iou = 0.0
        best_yolo_idx = -1
        
        for yolo_idx, yolo_box in enumerate(yolo_boxes):
            if yolo_idx in matched_yolo:
                continue
            iou = compute_iou(gt_box, yolo_box)
            if iou > best_iou:
                best_iou = iou
                best_yolo_idx = yolo_idx
        
        if best_iou >= iou_thresh and best_yolo_idx >= 0:
            matched_yolo.add(best_yolo_idx)
    
    # Return unmatched YOLO boxes
    unmatched = [yolo_boxes[i] for i in range(len(yolo_boxes)) if i not in matched_yolo]
    return unmatched


def is_vertical_outlier(
    candidate_box: Tuple[int, int, int, int],
    gt_boxes: List[Tuple[int, int, int, int]]
) -> Tuple[bool, str]:
    """
    Check if candidate box is at the most top or most bottom compared to GT boxes.
    
    Returns:
        (is_outlier, reason) where reason is "top", "bottom", or "none"
    """
    if not gt_boxes:
        return False, "none"
    
    # Get Y position (top of box) for all GT boxes
    gt_y_centers = [get_box_center(box)[1] for box in gt_boxes]
    candidate_y_center = get_box_center(candidate_box)[1]
    
    min_gt_y = min(gt_y_centers)
    max_gt_y = max(gt_y_centers)
    
    # Check if candidate is above all GT boxes (most top)
    if candidate_y_center < min_gt_y:
        return True, "top"
    
    # Check if candidate is below all GT boxes (most bottom)
    if candidate_y_center > max_gt_y:
        return True, "bottom"
    
    return False, "none"


def select_referee_from_unmatched(
    unmatched_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]]
) -> Tuple[Optional[Tuple[int, int, int, int]], float, bool, str]:
    """
    Select the referee as the unmatched YOLO box with smallest average distance to GT boxes.
    Then validate it's not a vertical outlier.
    
    Returns:
        (referee_box, avg_distance, is_outlier, outlier_reason)
    """
    if not unmatched_boxes or not gt_boxes:
        return None, float('inf'), False, "none"
    
    # First find best candidate by distance
    best_box = None
    best_distance = float('inf')
    
    for box in unmatched_boxes:
        avg_dist = calculate_average_distance_to_gt(box, gt_boxes)
        if avg_dist < best_distance:
            best_distance = avg_dist
            best_box = box
    
    # Now validate the candidate
    if best_box is not None:
        is_outlier, outlier_reason = is_vertical_outlier(best_box, gt_boxes)
        if is_outlier:
            # Reject this candidate - it's a vertical outlier
            return None, best_distance, True, outlier_reason
    
    return best_box, best_distance, False, "none"


# ══════════════════════════════════════════════════════════════════════
# Visualization: Side-by-Side Output
# ══════════════════════════════════════════════════════════════════════

def create_referee_visualization(
    frame: np.ndarray,
    unmatched_boxes: List[Tuple[int, int, int, int]],
    referee_box: Optional[Tuple[int, int, int, int]],
    avg_distance: float,
    is_outlier: bool,
    outlier_reason: str,
    fi: int,
    gt_count: int,
    yolo_count: int,
    gt_boxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Create side-by-side image:
    - LEFT: All YOLO boxes NOT in GT (unmatched)
    - RIGHT: Only the selected referee bounding box (if valid), otherwise blank
    """
    h, w = frame.shape[:2]
    
    # Create two copies
    left_img = frame.copy()
    right_img = frame.copy()
    
    # ── LEFT IMAGE: All unmatched YOLO boxes ──────────────────────────────
    for idx, (x, y, bw, bh) in enumerate(unmatched_boxes):
        # Draw in orange/blue
        color = (60, 120, 220)  # Blue-ish
        cv2.rectangle(left_img, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(left_img, f"YOLO_{idx}", (x, max(y - 5, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Draw GT boxes on left image for reference (as semi-transparent)
    for idx, (x, y, bw, bh) in enumerate(gt_boxes):
        color = (60, 200, 60)  # Green
        cv2.rectangle(left_img, (x, y), (x + bw, y + bh), color, 1)
    
    # Add legend to left image
    cv2.putText(left_img, "UNMATCHED YOLO BOXES (Blue)", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 120, 220), 2)
    cv2.putText(left_img, f"Count: {len(unmatched_boxes)}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(left_img, "GT Boxes (Green outline)", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 200, 60), 1)
    
    # Calculate vertical range of GT boxes for visualization
    if gt_boxes:
        gt_y_centers = [get_box_center(box)[1] for box in gt_boxes]
        min_gt_y = min(gt_y_centers)
        max_gt_y = max(gt_y_centers)
        
        # Draw vertical range indicator on left image
        cv2.line(left_img, (w - 20, min_gt_y), (w - 20, max_gt_y), (0, 255, 255), 2)
        cv2.putText(left_img, "GT Y-Range", (w - 80, min_gt_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    
    # ── RIGHT IMAGE: Only referee box (if valid) ──────────────────────────
    if referee_box is not None and not is_outlier:
        x, y, bw, bh = referee_box
        # Draw referee in RED with thicker line
        color = (0, 0, 255)  # Red
        cv2.rectangle(right_img, (x, y), (x + bw, y + bh), color, 3)
        cv2.putText(right_img, "REFEREE", (x, max(y - 8, 25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Add center point
        cx, cy = get_box_center(referee_box)
        cv2.circle(right_img, (cx, cy), 4, color, -1)
        
        # Add distance text
        cv2.putText(right_img, f"Avg dist to GT: {avg_distance:.1f}px", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add validation text
        cv2.putText(right_img, "VALIDATED: Within player vertical range", (10, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
    else:
        # No referee detected - show message and reason
        cv2.putText(right_img, "NO REFEREE DETECTED", (w//2 - 100, h//2 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if is_outlier and outlier_reason != "none":
            if outlier_reason == "top":
                msg = f"Best candidate REJECTED: At TOP of frame (above all GT boxes)"
            elif outlier_reason == "bottom":
                msg = f"Best candidate REJECTED: At BOTTOM of frame (below all GT boxes)"
            else:
                msg = "Best candidate REJECTED: Vertical outlier"
            
            cv2.putText(right_img, msg, (w//2 - 200, h//2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(right_img, "Likely linesman, ball boy, or sideline object", (w//2 - 220, h//2 + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        elif not referee_box and not is_outlier:
            cv2.putText(right_img, "No unmatched YOLO boxes found", (w//2 - 130, h//2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Add legend to right image
    cv2.putText(right_img, "REFEREE VERIFICATION", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(right_img, "Must be within GT vertical range", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Add frame info at bottom
    info_text = f"Frame: {fi:06d} | GT: {gt_count} | YOLO: {yolo_count} | Unmatched: {len(unmatched_boxes)}"
    cv2.putText(left_img, info_text, (w//2 - 150, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Stack horizontally
    side_by_side = np.hstack([left_img, right_img])
    
    # Add top title bar
    h_side, w_side = side_by_side.shape[:2]
    title_bar = np.zeros((40, w_side, 3), dtype=np.uint8)
    
    if referee_box is not None and not is_outlier:
        title = f"Frame {fi:06d} - REFEREE VALIDATED (Avg Dist: {avg_distance:.1f}px, Within GT Range)"
        title_color = (0, 255, 0)
    elif is_outlier:
        title = f"Frame {fi:06d} - REFEREE REJECTED (Candidate at {outlier_reason}, outside GT vertical range)"
        title_color = (0, 0, 255)
    else:
        title = f"Frame {fi:06d} - NO REFEREE CANDIDATE"
        title_color = (200, 200, 200)
    
    cv2.putText(title_bar, title, (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_color, 2)
    
    final = np.vstack([title_bar, side_by_side])
    
    return final


# ══════════════════════════════════════════════════════════════════════
# Main Referee Detection Function
# ══════════════════════════════════════════════════════════════════════

def detect_referee(
    seq_name: str,
    n_frames: int = 50,
    iou_thresh: float = 0.5,
    conf: float = 0.25,
    model_name: str = "yolov8m.pt",
    output_dir: Optional[str] = None
) -> None:
    """
    Main function: Detect referee as unmatched YOLO box closest to GT boxes.
    Validates that referee is not a vertical outlier.
    Outputs side-by-side visualizations.
    """
    here = Path(__file__).parent
    img_dir = here / "videos" / seq_name / "img1"
    gt_path = here / "videos" / seq_name / "gt" / "gt.txt"
    
    if not img_dir.is_dir():
        raise FileNotFoundError(f"img1/ not found at {img_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"gt.txt not found at {gt_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = here / "output" / f"{seq_name}_referee_detection"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GT
    print(f"Loading ground truth from {gt_path}")
    gt_all = _load_all_gt(str(gt_path))
    
    # Extract just boxes (without track IDs for distance calculation)
    gt_boxes_only: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for fi, entries in gt_all.items():
        gt_boxes_only[fi] = [bbox for _, bbox in entries]
    
    # Get frame indices to process
    all_paths = _collect_frame_paths(str(img_dir))
    all_indices = [frame_index_of(p) for p in all_paths if frame_index_of(p) >= 0]
    frame_indices = sorted(all_indices)[:n_frames]
    
    # Get team centroids from Feature 8
    print("Extracting team centroids from Feature 8...")
    home_centroid, away_centroid = get_team_centroids_from_sequence(seq_name, frame_indices)
    if home_centroid is not None:
        print("Team centroids extracted successfully")
    
    # Load YOLO model
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    print(f"\nProcessing {len(frame_indices)} frames from '{seq_name}'")
    print("-" * 70)
    
    referee_detections = []
    rejected_candidates = []
    
    for fi in frame_indices:
        fp = _frame_path_for_idx(all_paths, fi)
        if fp is None:
            continue
        
        frame = cv2.imread(fp)
        if frame is None:
            continue
        
        # Get GT boxes for this frame
        gt_boxes = gt_boxes_only.get(fi, [])
        if not gt_boxes:
            continue
        
        # Run YOLO detection
        results = model(frame, classes=[0], conf=conf, verbose=False)
        yolo_boxes = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                yolo_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        if not yolo_boxes:
            continue
        
        # Find unmatched YOLO boxes
        unmatched_boxes = find_unmatched_yolo_boxes(gt_boxes, yolo_boxes, iou_thresh)
        
        if not unmatched_boxes:
            # No unmatched boxes - no referee candidate
            # Still save visualization showing no candidate
            vis_img = create_referee_visualization(
                frame, [], None, 0.0, False, "none", fi,
                len(gt_boxes), len(yolo_boxes), gt_boxes
            )
            output_path = output_dir / f"frame_{fi:06d}_referee.jpg"
            cv2.imwrite(str(output_path), vis_img)
            continue
        
        # Select and validate referee candidate
        referee_box, avg_distance, is_outlier, outlier_reason = select_referee_from_unmatched(
            unmatched_boxes, gt_boxes
        )
        
        if referee_box is not None and not is_outlier:
            # Valid referee detected
            referee_detections.append((fi, referee_box, avg_distance))
            status = f"✓ VALID"
            print(f"Frame {fi:06d}: {status} | Referee at y={get_box_center(referee_box)[1]} | Unmatched: {len(unmatched_boxes)} | Avg dist: {avg_distance:.1f}px")
        elif is_outlier:
            # Candidate was rejected
            rejected_candidates.append((fi, avg_distance, outlier_reason))
            status = f"✗ REJECTED ({outlier_reason})"
            print(f"Frame {fi:06d}: {status} | Best candidate at y={get_box_center(unmatched_boxes[0])[1] if unmatched_boxes else 'N/A'} | Avg dist: {avg_distance:.1f}px")
        else:
            # No candidate found
            status = f"✗ NO CANDIDATE"
            print(f"Frame {fi:06d}: {status}")
        
        # Create visualization (will show referee or rejection reason)
        vis_img = create_referee_visualization(
            frame, unmatched_boxes, referee_box, avg_distance, is_outlier, outlier_reason, fi,
            len(gt_boxes), len(yolo_boxes), gt_boxes
        )
        
        # Save image
        output_path = output_dir / f"frame_{fi:06d}_referee.jpg"
        cv2.imwrite(str(output_path), vis_img)
    
    # Summary report
    print("\n" + "=" * 70)
    print("REFEREE DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total frames processed: {len(frame_indices)}")
    print(f"  Frames with VALID referee: {len(referee_detections)}")
    print(f"  Frames with REJECTED candidate: {len(rejected_candidates)}")
    print(f"  Frames with no candidate: {len(frame_indices) - len(referee_detections) - len(rejected_candidates)}")
    
    if referee_detections:
        print(f"\n  ✓ VALID referee detected in frames:")
        for fi, box, dist in referee_detections[:10]:
            cx, cy = get_box_center(box)
            print(f"    Frame {fi:06d}: center=({cx},{cy}), avg_dist={dist:.1f}px")
        if len(referee_detections) > 10:
            print(f"    ... and {len(referee_detections) - 10} more frames")
    
    if rejected_candidates:
        print(f"\n  ✗ REJECTED candidates (vertical outliers):")
        for fi, dist, reason in rejected_candidates[:10]:
            print(f"    Frame {fi:06d}: rejected due to {reason}, avg_dist={dist:.1f}px")
        if len(rejected_candidates) > 10:
            print(f"    ... and {len(rejected_candidates) - 10} more frames")
    
    print(f"\n  Output images saved to: {output_dir}")
    print("=" * 70)


def frame_index_of(path: str) -> int:
    """Parse frame index from filename."""
    try:
        return int(Path(path).stem)
    except ValueError:
        return -1


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect referee as unmatched YOLO box closest to GT boxes (with vertical outlier rejection)"
    )
    parser.add_argument(
        "--seq", type=str, default="v_1yHWGw8DH4A_c047",
        help="Sequence folder name inside videos/"
    )
    parser.add_argument(
        "--frames", type=int, default=50,
        help="Number of leading frames to process (default: 50)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for GT-YOLO matching (default: 0.5)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m.pt",
        help="YOLO model weights"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for images"
    )
    
    args = parser.parse_args()
    
    detect_referee(
        seq_name=args.seq,
        n_frames=args.frames,
        iou_thresh=args.iou,
        conf=args.conf,
        model_name=args.model,
        output_dir=args.output
    )