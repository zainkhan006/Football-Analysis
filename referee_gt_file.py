"""
Generate Referee Ground Truth for ALL Sequences
=================================================
Iterates through all folders in videos/ directory and creates referee_gt.txt
for each sequence that has a valid gt.txt file.

File location: videos/<seq_name>/gt/referee_gt.txt

Usage
-----
    python generate_referee_gt_all.py
    python generate_referee_gt_all.py --frames 100 --iou 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO

# Import detection logic from referee_detection
import sys
sys.path.append(str(Path(__file__).parent))

# Import the functions we need
from referee_detection import (
    compute_iou,
    get_box_center,
    calculate_average_distance_to_gt,
    find_unmatched_yolo_boxes,
    is_vertical_outlier,
    select_referee_from_unmatched,
    get_team_centroids_from_sequence,
    _load_all_gt,
    _collect_frame_paths,
    _frame_path_for_idx
)


def generate_referee_ground_truth_for_sequence(
    seq_path: Path,
    n_frames: int = 50,
    iou_thresh: float = 0.5,
    conf: float = 0.25,
    model_name: str = "yolov8m.pt",
    output_confidence: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Generate referee ground truth file for a single sequence.
    
    Returns:
        Dictionary with statistics: {total_frames, referee_detections, detection_rate}
    """
    seq_name = seq_path.name
    img_dir = seq_path / "img1"
    gt_path = seq_path / "gt" / "gt.txt"
    referee_gt_path = seq_path / "gt" / "referee_gt.txt"
    
    if not img_dir.is_dir():
        if verbose:
            print(f"  ⚠️  Skipping {seq_name}: no img1/ folder")
        return None
    
    if not gt_path.is_file():
        if verbose:
            print(f"  ⚠️  Skipping {seq_name}: no gt/gt.txt file")
        return None
    
    if verbose:
        print(f"\n  Processing: {seq_name}")
    
    # Load GT
    gt_all = _load_all_gt(str(gt_path))
    
    # Extract just boxes (without track IDs for distance calculation)
    gt_boxes_only: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for fi, entries in gt_all.items():
        gt_boxes_only[fi] = [bbox for _, bbox in entries]
    
    # Get frame indices to process
    all_paths = _collect_frame_paths(str(img_dir))
    all_indices = []
    for p in all_paths:
        try:
            idx = int(Path(p).stem)
            all_indices.append(idx)
        except ValueError:
            continue
    
    frame_indices = sorted(all_indices)[:n_frames]
    
    if not frame_indices:
        if verbose:
            print(f"  ⚠️  No valid frames found")
        return None
    
    # Get team centroids (optional, for logging)
    home_centroid, away_centroid = get_team_centroids_from_sequence(seq_name, frame_indices)
    
    # Load YOLO model (reuse if possible, but for simplicity load per sequence)
    model = YOLO(model_name)
    
    # Collection for referee ground truth
    referee_entries: Dict[int, Tuple[int, int, int, int, float]] = {}
    
    for fi in frame_indices:
        fp = _frame_path_for_idx(all_paths, fi)
        if fp is None:
            continue
        
        frame = cv2.imread(fp)
        if frame is None:
            continue
        
        # Get GT boxes
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
            continue
        
        # Select and validate referee candidate
        referee_box, avg_distance, is_outlier, outlier_reason = select_referee_from_unmatched(
            unmatched_boxes, gt_boxes
        )
        
        if referee_box is not None and not is_outlier:
            # Valid referee detected
            x, y, w, h = referee_box
            confidence = avg_distance if output_confidence else 1.0
            referee_entries[fi] = (x, y, w, h, confidence)
    
    # Write referee ground truth file
    if referee_entries:
        with open(referee_gt_path, 'w', encoding='utf-8') as f:
            for fi in sorted(referee_entries.keys()):
                x, y, w, h, conf_val = referee_entries[fi]
                # MOT format: frame, track_id, x, y, w, h, confidence, -1, -1, -1
                line = f"{fi},0,{x},{y},{w},{h},{conf_val:.3f},-1,-1,-1\n"
                f.write(line)
        
        # Write summary file
        summary_path = referee_gt_path.parent / "referee_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"REFEREE GROUND TRUTH SUMMARY - {seq_name}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Frames processed: {len(frame_indices)}\n")
            f.write(f"  IoU threshold: {iou_thresh}\n")
            f.write(f"  YOLO confidence: {conf}\n")
            f.write(f"  Model: {model_name}\n\n")
            
            f.write(f"Detection Results:\n")
            f.write(f"  Total referee detections: {len(referee_entries)}\n")
            f.write(f"  Detection rate: {len(referee_entries)/len(frame_indices)*100:.1f}%\n\n")
            
            f.write("Frame-by-frame referee detections:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Frame':>6} | {'X':>4} | {'Y':>4} | {'W':>4} | {'H':>4} | {'Confidence':>10} | {'Center X':>7} | {'Center Y':>7}\n")
            f.write("-" * 70 + "\n")
            
            for fi in sorted(referee_entries.keys()):
                x, y, w, h, conf_val = referee_entries[fi]
                cx = x + w // 2
                cy = y + h // 2
                f.write(f"{fi:>6} | {x:>4} | {y:>4} | {w:>4} | {h:>4} | {conf_val:>10.3f} | {cx:>7} | {cy:>7}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        if verbose:
            print(f"    ✓ Referee GT written: {len(referee_entries)}/{len(frame_indices)} frames ({len(referee_entries)/len(frame_indices)*100:.1f}%)")
        
        return {
            'seq_name': seq_name,
            'total_frames': len(frame_indices),
            'referee_detections': len(referee_entries),
            'detection_rate': len(referee_entries)/len(frame_indices)*100
        }
    else:
        if verbose:
            print(f"    ✗ No referee detections found")
        return None


def generate_all_referee_ground_truth(
    videos_root: Optional[str] = None,
    n_frames: int = 50,
    iou_thresh: float = 0.5,
    conf: float = 0.25,
    model_name: str = "yolov8m.pt",
    output_confidence: bool = True,
    verbose: bool = True
) -> None:
    """
    Iterate through all sequence folders in videos/ directory and generate
    referee_gt.txt for each.
    """
    if videos_root is None:
        here = Path(__file__).parent
        videos_root = here / "videos"
    else:
        videos_root = Path(videos_root)
    
    if not videos_root.is_dir():
        raise FileNotFoundError(f"videos/ folder not found at {videos_root}")
    
    # Get all subdirectories in videos/
    seq_dirs = [d for d in videos_root.iterdir() if d.is_dir()]
    
    if not seq_dirs:
        print(f"No sequence folders found in {videos_root}")
        return
    
    print(f"\n{'='*70}")
    print(f"GENERATING REFEREE GROUND TRUTH FOR ALL SEQUENCES")
    print(f"{'='*70}")
    print(f"  Found {len(seq_dirs)} sequence(s)")
    print(f"  Frames per sequence: {n_frames}")
    print(f"  IoU threshold: {iou_thresh}")
    print(f"  YOLO confidence: {conf}")
    print(f"{'='*70}")
    
    results = []
    
    for seq_dir in sorted(seq_dirs):
        result = generate_referee_ground_truth_for_sequence(
            seq_path=seq_dir,
            n_frames=n_frames,
            iou_thresh=iou_thresh,
            conf=conf,
            model_name=model_name,
            output_confidence=output_confidence,
            verbose=verbose
        )
        if result:
            results.append(result)
    
    # Print overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sequence':<35} | {'Frames':>8} | {'Referee Detections':>18} | {'Rate':>6}")
    print(f"{'-'*70}")
    
    total_frames = 0
    total_detections = 0
    
    for r in results:
        total_frames += r['total_frames']
        total_detections += r['referee_detections']
        print(f"{r['seq_name']:<35} | {r['total_frames']:>8} | {r['referee_detections']:>18} | {r['detection_rate']:>5.1f}%")
    
    print(f"{'-'*70}")
    print(f"{'TOTAL':<35} | {total_frames:>8} | {total_detections:>18} | {total_detections/total_frames*100:>5.1f}%")
    print(f"{'='*70}")
    
    # Save master summary
    master_summary = videos_root / "referee_gt_master_summary.txt"
    with open(master_summary, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MASTER SUMMARY - REFEREE GROUND TRUTH FOR ALL SEQUENCES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Frames per sequence: {n_frames}\n")
        f.write(f"  IoU threshold: {iou_thresh}\n")
        f.write(f"  YOLO confidence: {conf}\n")
        f.write(f"  Model: {model_name}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"{'Sequence':<40} | {'Frames':>8} | {'Referee Detections':>18} | {'Rate':>6}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            f.write(f"{r['seq_name']:<40} | {r['total_frames']:>8} | {r['referee_detections']:>18} | {r['detection_rate']:>5.1f}%\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<40} | {total_frames:>8} | {total_detections:>18} | {total_detections/total_frames*100:>5.1f}%\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n  Master summary saved to: {master_summary}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate referee ground truth files for ALL sequences in videos/ folder"
    )
    parser.add_argument(
        "--frames", type=int, default=50,
        help="Number of leading frames to process per sequence (default: 50)"
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
        "--quiet", action="store_true",
        help="Suppress per-sequence verbose output"
    )
    parser.add_argument(
        "--no-confidence", action="store_true",
        help="Don't store confidence in the GT file"
    )
    
    args = parser.parse_args()
    
    generate_all_referee_ground_truth(
        n_frames=args.frames,
        iou_thresh=args.iou,
        conf=args.conf,
        model_name=args.model,
        output_confidence=not args.no_confidence,
        verbose=not args.quiet
    )