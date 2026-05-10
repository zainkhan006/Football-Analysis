"""
Quick Start: Using the FootballMOT Dataset
Copy & paste these examples to quickly work with the dataset
"""

from pathlib import Path
from visualize_dataset import FootballMOTVisualizer
import cv2

# ============================================
# SETUP
# ============================================
dataset_path = Path("dataset/train/v_gQNyhv8y0QY_c013")
viz = FootballMOTVisualizer(dataset_path)


# ============================================
# EXAMPLE 1: View a Single Frame
# ============================================
# Shows frame 10 with bounding boxes and player IDs drawn on it
viz.display_frame_with_boxes(766, delay=3000)  # Display for 3 seconds


# ============================================
# EXAMPLE 2: Get Player Info from Frame 10
# ============================================
# Prints a table of all players detected in frame 10
viz.print_frame_annotations(10)

# Output:
# Frame 10 Annotations (11 players detected)
# Track ID   X        Y        Width    Height   Area
# 0          215      374      35       93       3255
# 1          712      553      72       115      8280
# ... etc


# ============================================
# EXAMPLE 3: Play the Video with Annotations
# ============================================
# Plays frames 1-100 with bounding boxes (like a video)
# Press 'q' to quit, 'p' to pause
viz.play_sequence(
    start_frame=1, 
    end_frame=875, 
    frame_delay=25  # milliseconds (25 fps = 40ms)
)


# ============================================
# EXAMPLE 4: Save an Annotated Frame
# ============================================
# Exports frame 50 with bounding boxes to a JPG file
viz.export_frame_with_boxes(50, "frame_50_annotated.jpg")


# ============================================
# EXAMPLE 5: Access Raw Annotation Data
# ============================================
# Get all detections for frame 10
frame_num = 10
detections = viz.gt_data[frame_num]

for detection in detections:
    track_id = detection['track_id']
    x, y, w, h = detection['bbox']
    print(f"Player {track_id}: x={x}, y={y}, width={w}, height={h}")

# Output:
# Player 0: x=215, y=374, width=35, height=93
# Player 1: x=712, y=553, width=72, height=115
# ... etc


# ============================================
# EXAMPLE 6: Get Statistics
# ============================================
# How many times each player appears
from collections import defaultdict

track_frames = defaultdict(int)
for frame_detections in viz.gt_data.values():
    for det in frame_detections:
        track_frames[det['track_id']] += 1

print(f"Player 0 appears in: {track_frames[0]} frames")
print(f"Player 4 appears in: {track_frames[4]} frames (entire video)")


# ============================================
# EXAMPLE 7: Process All Frames
# ============================================
# Loop through all frames and count players
for frame_num, detections in sorted(viz.gt_data.items()):
    player_count = len(detections)
    print(f"Frame {frame_num}: {player_count} players detected")


# ============================================
# EXAMPLE 8: Get Specific Player's Trajectory
# ============================================
# Track where player ID 0 is throughout the video
player_trajectory = []

for frame_num in sorted(viz.gt_data.keys()):
    for det in viz.gt_data[frame_num]:
        if det['track_id'] == 0:
            x, y, w, h = det['bbox']
            # Center of bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            player_trajectory.append({
                'frame': frame_num,
                'center': (center_x, center_y),
                'bbox': (x, y, w, h)
            })

print(f"Player 0 trajectory: {len(player_trajectory)} positions")
for pos in player_trajectory[:5]:  # First 5
    print(f"  Frame {pos['frame']}: center at {pos['center']}")


# ============================================
# EXAMPLE 9: Custom Frame Processing
# ============================================
# Load a frame and manually draw boxes in your own way
import cv2

frame_num = 766
img_path = dataset_path / "img1" / f"{frame_num:06d}.jpg"
img = cv2.imread(str(img_path))

# Your custom processing here...
for det in viz.gt_data[frame_num]:
    track_id = det['track_id']
    x, y, w, h = det['bbox']
    # Do custom visualization
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Custom Processing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================
# EXAMPLE 10: Dataset Information
# ============================================
# Access metadata from seqinfo.ini
print(f"Sequence: {viz.seq_info['name']}")
print(f"FPS: {viz.seq_info['frameRate']}")
print(f"Total Frames: {viz.seq_info['seqLength']}")
print(f"Resolution: {viz.seq_info['imWidth']}x{viz.seq_info['imHeight']}")

# Output:
# Sequence: v_gQNyhv8y0QY_c013
# FPS: 25
# Total Frames: 875
# Resolution: 1280x720


# ============================================
# ANNOTATION FORMAT REFERENCE
# ============================================
"""
Ground Truth Format (gt.txt):
<frame_id>, <track_id>, <x>, <y>, <width>, <height>, <conf>, <class>, <visibility>

Example line:
1, 0, 39, 388, 64, 88, 1, 1, 1

Meaning:
- frame_id=1: In frame 1
- track_id=0: Player with ID 0
- x=39, y=388: Top-left corner of bounding box
- width=64, height=88: Size of bounding box
- conf=1: Confidence score (always 1)
- class=1: Player class (always 1)
- visibility=1: Fully visible (always 1)
"""
