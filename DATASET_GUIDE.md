# SportsMOT Football Dataset Analysis

## 📊 Dataset Overview

### What You Have
- **Single Football Video**: `v_gQNyhv8y0QY_c013`
- **Format**: MOT (Multiple Object Tracking) - broken down into individual frames
- **Duration**: 35 seconds
- **Frame Rate**: 25 fps
- **Resolution**: 1280×720 pixels
- **Total Frames**: 875

---

## 📁 Folder Structure Explained

```
v_gQNyhv8y0QY_c013/
├── seqinfo.ini          # Video sequence metadata
├── img1/                # Individual frames (000001.jpg to 000875.jpg)
└── gt/
    └── gt.txt           # Ground truth annotations (player locations)
```

### seqinfo.ini
Contains metadata:
- Video name/ID
- Frame count and rate
- Resolution
- Image directory and format

---

## 🎯 Ground Truth Format (gt.txt)

Each line represents ONE player detection in ONE frame:

```
<frame_id>, <track_id>, <x>, <y>, <width>, <height>, <confidence>, <class>, <visibility>
```

### Example:
```
10, 0, 215, 374, 35, 93, 1, 1, 1
```
Means: In **frame 10**, there's a bounding box for player **track ID 0** at pixel position **(215, 374)** with size **35×93 pixels**.

### Components:
| Field | Example | Meaning |
|-------|---------|---------|
| frame_id | 10 | Frame number (1-indexed) |
| track_id | 0 | Player ID (consistent across frames for tracking) |
| x | 215 | Left edge of bounding box |
| y | 374 | Top edge of bounding box |
| width | 35 | Bounding box width |
| height | 93 | Bounding box height |
| confidence | 1 | Detection confidence score |
| class | 1 | Object class (1 = player) |
| visibility | 1 | Visibility flag (1 = fully visible) |

---

## 📈 Dataset Statistics

### Annotations Summary:
- **Total Detections**: 10,557 bounding boxes total
- **Unique Track IDs**: 20 (20 unique players tracked)
- **Average Players per Frame**: 12.07 (typical lineup is ~11v11)
- **All 875 Frames**: Have annotations

### Player Appearance Duration:
The longest tracked player (ID 4) appears in **all 875 frames** (35 seconds)

| Player ID | Appearances | Duration (seconds) |
|-----------|-------------|-------------------|
| 4 | 875 | 35.00 (full video) |
| 2 | 772 | 30.88 |
| 5 | 788 | 31.52 |
| 1 | 270 | 10.80 |
| 10 | 222 | 8.88 |

---

## 💻 Code Features

The provided `visualize_dataset.py` script includes:

### 1. **Display Single Frames**
```python
viz.display_frame_with_boxes(10)  # Show frame 10 with bounding boxes
```
- Draws bounding boxes with player IDs
- Shows frame number and player count

### 2. **Print Frame Annotations**
```python
viz.print_frame_annotations(10)  # Print all detections in frame 10
```
- Lists all players in a frame with their positions

### 3. **Play Sequence Interactively**
```python
viz.play_sequence(start_frame=1, end_frame=100, frame_delay=40)
```
- Plays frames like a video with annotations
- Press 'q' to quit, 'p' to pause

### 4. **Export Annotated Frames**
```python
viz.export_frame_with_boxes(50, "output.jpg")
```
- Saves a frame with bounding boxes drawn

### 5. **Track Statistics**
- Per-player appearance counts
- Duration each player is tracked
- Unique player IDs

---

## 🎮 How to Use

### Basic Usage:
```python
from pathlib import Path
from visualize_dataset import FootballMOTVisualizer

# Initialize
dataset_path = Path("dataset/train/v_gQNyhv8y0QY_c013")
viz = FootballMOTVisualizer(dataset_path)

# View frame 50
viz.display_frame_with_boxes(50, delay=2000)

# Check what's in frame 50
viz.print_frame_annotations(50)

# Play first 200 frames
viz.play_sequence(start_frame=1, end_frame=200)

# Save annotated frame
viz.export_frame_with_boxes(100, "frame_100_with_boxes.jpg")
```

---

## 🔍 What the Colors Mean

Each player (track ID) gets a unique color:
- **Red-ish colors**: Some players
- **Green-ish colors**: Other players
- **Blue-ish colors**: Remaining players

The color stays consistent for each player throughout the video, making it easy to track them.

---

## 📊 Example Output from Frame 10

| Player ID | X | Y | Width | Height | Bounding Box Area |
|-----------|-----|-------|-------|--------|-------------------|
| 0 | 215 | 374 | 35 | 93 | 3,255 px² |
| 1 | 712 | 553 | 72 | 115 | 8,280 px² |
| 2 | 620 | 109 | 22 | 73 | 1,606 px² |
| 3 | 956 | 394 | 66 | 98 | 6,468 px² |
| ... | ... | ... | ... | ... | ... |

Players with larger bounding boxes = closer to camera or larger in frame.

---

## 🎬 Next Steps for Your CV Project

1. **Player Detection**: Train models to detect players given frames
2. **Tracking**: Track players across frames using track IDs
3. **Action Recognition**: Classify player actions (running, shooting, etc.)
4. **Ball Detection**: Add ball tracking (not in current gt.txt)
5. **Team Classification**: Distinguish between teams (not in current annotations)

---

## 📌 Important Notes

- Track IDs are **consistent across frames** (same player = same ID)
- The format follows **MOT challenges** standard
- You have **20 unique players** = ~2 teams + substitutes/referees
- **All frames are annotated** - no gaps in tracking data
- Confidence and visibility are always 1 = clean, high-quality annotations
