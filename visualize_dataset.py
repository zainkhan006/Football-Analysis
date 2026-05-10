"""
SportsMOT Football Dataset Visualization Tool
Visualizes MOT (Multiple Object Tracking) format ground truth annotations
on video frames with player bounding boxes and tracking IDs
"""

import cv2
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


class FootballMOTVisualizer:
    def __init__(self, dataset_path):
        """
        Initialize the visualizer
        
        Args:
            dataset_path: Path to the sequence folder (e.g., v_gQNyhv8y0QY_c013)
        """
        self.dataset_path = Path(dataset_path)
        self.img_dir = self.dataset_path / "img1"
        self.gt_file = self.dataset_path / "gt" / "gt.txt"
        self.seqinfo_file = self.dataset_path / "seqinfo.ini"
        
        # Load sequence info
        self.seq_info = self._load_seqinfo()
        
        # Load ground truth
        self.gt_data = self._load_gt()
        
        # Print statistics
        self._print_statistics()
    
    def _load_seqinfo(self):
        """Parse seqinfo.ini file"""
        info = {}
        with open(self.seqinfo_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=')
                    info[key.strip()] = value.strip()
        return info
    
    def _load_gt(self):
        """
        Load ground truth annotations
        Returns dict: frame_num -> list of detections
        """
        gt_data = defaultdict(list)
        
        with open(self.gt_file, 'r') as f:
            for line in f:
                # Parse: frame_id, track_id, x, y, width, height, conf, class, visibility
                values = [int(float(x.strip())) for x in line.strip().split(',')]
                
                frame_id = values[0]
                track_id = values[1]
                x, y, w, h = values[2:6]
                conf = values[6]
                cls = values[7]
                visibility = values[8]
                
                gt_data[frame_id].append({
                    'track_id': track_id,
                    'bbox': (x, y, w, h),  # (x, y, width, height)
                    'confidence': conf,
                    'class': cls,
                    'visibility': visibility
                })
        
        return gt_data
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("FOOTBALL MOT DATASET STATISTICS")
        print("="*60)
        
        print(f"\nSequence Name: {self.seq_info.get('name', 'N/A')}")
        print(f"Frame Rate (fps): {self.seq_info.get('frameRate', 'N/A')}")
        print(f"Total Frames: {self.seq_info.get('seqLength', 'N/A')}")
        print(f"Resolution: {self.seq_info.get('imWidth', 'N/A')}×{self.seq_info.get('imHeight', 'N/A')}")
        print(f"Image Format: {self.seq_info.get('imExt', '.jpg')}")
        
        # Calculate video duration
        total_frames = int(self.seq_info.get('seqLength', 0))
        fps = int(self.seq_info.get('frameRate', 25))
        duration_sec = total_frames / fps
        print(f"Estimated Duration: {duration_sec:.2f} seconds ({int(duration_sec//60)}m {int(duration_sec%60)}s)")
        
        # Annotations statistics
        total_annotations = sum(len(dets) for dets in self.gt_data.values())
        unique_track_ids = set()
        for dets in self.gt_data.values():
            for det in dets:
                unique_track_ids.add(det['track_id'])
        
        print(f"\nAnnotations Statistics:")
        print(f"  Total Detections: {total_annotations}")
        print(f"  Unique Track IDs (Players): {len(unique_track_ids)}")
        print(f"  Track IDs: {sorted(list(unique_track_ids))}")
        print(f"  Frames with Annotations: {len(self.gt_data)}")
        
        # Calculate avg players per frame
        avg_players = total_annotations / len(self.gt_data) if self.gt_data else 0
        print(f"  Average Players per Frame: {avg_players:.2f}")
        
        print("="*60 + "\n")
    
    def display_frame_with_boxes(self, frame_num, delay=500):
        """
        Display a single frame with bounding boxes
        
        Args:
            frame_num: Frame number to display (1-indexed)
            delay: Display duration in milliseconds
        """
        # Load image
        img_path = self.img_dir / f"{frame_num:06d}.jpg"
        if not img_path.exists():
            print(f"Frame {frame_num} not found at {img_path}")
            return
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            return
        
        # Get detections for this frame
        detections = self.gt_data.get(frame_num, [])
        
        # Draw bounding boxes
        colors = self._generate_colors(len(set(d['track_id'] for d in detections)))
        
        for det in detections:
            track_id = det['track_id']
            x, y, w, h = det['bbox']
            
            # Get color for this track
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw track ID label
            label = f"ID:{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(img, (x, y - 25), (x + text_size[0] + 5, y), color, -1)
            cv2.putText(img, label, (x + 2, y - 8), font, font_scale, (255, 255, 255), thickness)
        
        # Add frame info
        info_text = f"Frame: {frame_num}/{self.seq_info.get('seqLength', 'N/A')} | Players: {len(detections)}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow(f"Frame {frame_num}", img)
        cv2.waitKey(delay)
    
    def play_sequence(self, start_frame=1, end_frame=None, frame_delay=40):
        """
        Play the sequence from start to end frame with annotations
        
        Args:
            start_frame: Starting frame (1-indexed)
            end_frame: Ending frame (if None, plays until last frame)
            frame_delay: Delay between frames in ms (default 40ms ≈ 25fps)
        """
        max_frame = int(self.seq_info.get('seqLength', 875))
        if end_frame is None:
            end_frame = max_frame
        
        print(f"Playing sequence from frame {start_frame} to {end_frame}...")
        print("Press 'q' to quit, 'p' to pause, any other key to continue")
        
        for frame_num in range(start_frame, end_frame + 1):
            img_path = self.img_dir / f"{frame_num:06d}.jpg"
            if not img_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Get detections
            detections = self.gt_data.get(frame_num, [])
            
            # Draw boxes
            colors = self._generate_colors(12)  # Max ~12 players on field
            
            for det in detections:
                track_id = det['track_id']
                x, y, w, h = det['bbox']
                color = colors[track_id % len(colors)]
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label = f"ID:{track_id}"
                cv2.putText(img, label, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
            
            # Frame info
            info_text = f"Frame: {frame_num}/{max_frame} | Players: {len(detections)}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Football MOT Sequence", img)
            
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                print("Playback stopped by user")
                break
            elif key == ord('p'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)
    
    def print_frame_annotations(self, frame_num):
        """Print all annotations for a specific frame"""
        detections = self.gt_data.get(frame_num, [])
        
        print(f"\n{'='*70}")
        print(f"Frame {frame_num} Annotations ({len(detections)} players detected)")
        print(f"{'='*70}")
        
        if not detections:
            print("No annotations for this frame")
            return
        
        print(f"{'Track ID':<10} {'X':<8} {'Y':<8} {'Width':<8} {'Height':<8} {'Area':<10}")
        print("-" * 70)
        
        for det in sorted(detections, key=lambda x: x['track_id']):
            track_id = det['track_id']
            x, y, w, h = det['bbox']
            area = w * h
            print(f"{track_id:<10} {x:<8} {y:<8} {w:<8} {h:<8} {area:<10}")
        
        print("=" * 70 + "\n")
    
    def export_frame_with_boxes(self, frame_num, output_path):
        """Save a frame with bounding boxes to file"""
        img_path = self.img_dir / f"{frame_num:06d}.jpg"
        if not img_path.exists():
            print(f"Frame {frame_num} not found")
            return
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            return
        
        detections = self.gt_data.get(frame_num, [])
        colors = self._generate_colors(len(set(d['track_id'] for d in detections)))
        
        for det in detections:
            track_id = det['track_id']
            x, y, w, h = det['bbox']
            color = colors[track_id % len(colors)]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"ID:{track_id}"
            cv2.putText(img, label, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        info_text = f"Frame: {frame_num} | Players: {len(detections)}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, img)
        print(f"Saved annotated frame to: {output_path}")
    
    @staticmethod
    def _generate_colors(n_colors):
        """Generate distinct colors for different tracks"""
        colors = []
        for i in range(n_colors):
            hue = int(180 * i / n_colors)
            hsv = np.uint8([[[hue, 255, 255]]])
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, rgb)))
        return colors


def main():
    """Main demonstration"""
    
    # Path to the dataset
    dataset_path = Path("c:/Users/lahma/OneDrive/Documents/SportsMOT_example/dataset/train/v_gQNyhv8y0QY_c013")
    
    # Create visualizer
    viz = FootballMOTVisualizer(dataset_path)
    
    # --- EXAMPLES ---
    
    # 1. Display a single frame with bounding boxes
    print("\n1. Displaying frame 10 with bounding boxes...")
    viz.display_frame_with_boxes(10, delay=3000)  # Display for 3 seconds
    
    # 2. Print annotations for a specific frame
    print("\n2. Printing annotations for frame 10:")
    viz.print_frame_annotations(10)
    
    # 3. Play a sequence (first 100 frames)
    print("\n3. Starting playback of frames 1-100...")
    print("(This will display frames with player bounding boxes)")
    # Uncomment to run interactively:
    # viz.play_sequence(start_frame=1, end_frame=100, frame_delay=40)
    
    # 4. Export an annotated frame
    print("\n4. Exporting annotated frame 50 to output...")
    output_path = str(dataset_path / "frame_50_annotated.jpg")
    viz.export_frame_with_boxes(50, output_path)
    
    # 5. Get track statistics
    print("\n5. Track-specific statistics:")
    track_frames = defaultdict(int)
    for frame_detections in viz.gt_data.values():
        for det in frame_detections:
            track_frames[det['track_id']] += 1
    
    print(f"\n{'Track ID':<12} {'Appearances':<15} {'Duration (sec)':<15}")
    print("-" * 42)
    fps = int(viz.seq_info.get('frameRate', 25))
    for track_id in sorted(track_frames.keys()):
        appearances = track_frames[track_id]
        duration = appearances / fps
        print(f"{track_id:<12} {appearances:<15} {duration:<15.2f}")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
