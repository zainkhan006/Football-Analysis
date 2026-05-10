# Ground Truth File (gt.txt) - Complete Explanation

## What is gt.txt?

The `gt/gt.txt` file contains **ground truth annotations** for every player detection in every frame.
It's in MOT (Multiple Object Tracking) format - the standard for tracking datasets.

---

## File Format

**Raw format** (one player per line):
```
frame_id, track_id, x, y, width, height, confidence, class, visibility
```

**Example from your dataset (first 20 lines of gt.txt):**
```
1, 0, 39, 388, 64, 88, 1, 1, 1      ← Frame 1: Player 0 at (39,388) size 64x88
1, 1, 588, 554, 62, 110, 1, 1, 1    ← Frame 1: Player 1 at (588,554) size 62x110
1, 2, 424, 108, 24, 74, 1, 1, 1     ← Frame 1: Player 2 at (424,108) size 24x74
1, 3, 831, 396, 74, 98, 1, 1, 1     ← Frame 1: Player 3 at (831,396) size 74x98
1, 4, 818, 225, 42, 84, 1, 1, 1     ← Frame 1: Player 4 at (818,225) size 42x84
1, 5, 721, 188, 36, 80, 1, 1, 1     ← Frame 1: Player 5 at (721,188) size 36x80
1, 6, 994, 67, 32, 68, 1, 1, 1      ← Frame 1: Player 6 at (994,67) size 32x68
1, 7, 1053, 91, 41, 73, 1, 1, 1     ← Frame 1: Player 7 at (1053,91) size 41x73
1, 8, 1042, 73, 36, 75, 1, 1, 1     ← Frame 1: Player 8 at (1042,73) size 36x75
1, 9, 191, 44, 28, 69, 1, 1, 1      ← Frame 1: Player 9 at (191,44) size 28x69
1, 11, 1231, 136, 49, 83, 1, 1, 1   ← Frame 1: Player 11 at (1231,136) size 49x83
2, 0, 62, 387, 68, 87, 1, 1, 1      ← Frame 2: Player 0 at (62,387) size 68x87
2, 1, 608, 555, 64, 112, 1, 1, 1    ← Frame 2: Player 1 at (608,555) size 64x112
2, 2, 450, 108, 24, 74, 1, 1, 1     ← Frame 2: Player 2 at (450,108) size 24x74
...
```

---

## Detailed Field Explanation

### 1. **frame_id** (1st column)
- **What it is**: Frame number in the sequence
- **Range**: 1 to 875 (your dataset has 875 frames)
- **Example**: `1, 2, 3, ...`
- **Purpose**: Which image frame are we looking at?

### 2. **track_id** (2nd column)
- **What it is**: Unique player identifier that stays consistent across frames
- **Range**: 0 to 19 (your dataset has 20 unique players)
- **Example**: `0, 1, 2, 3, 4, 5, ...`
- **Purpose**: Track the same player across multiple frames
- **Key insight**: Same player = same ID across the entire video

### 3. **x** (3rd column)
- **What it is**: Horizontal position of the bounding box (left edge)
- **Unit**: Pixels from the left
- **Range**: 0 to 1280 (video width)
- **Example**: `39, 588, 424, 831, ...`
- **Meaning**: How far from the left edge of the frame

### 4. **y** (4th column)
- **What it is**: Vertical position of the bounding box (top edge)
- **Unit**: Pixels from the top
- **Range**: 0 to 720 (video height)
- **Example**: `388, 554, 108, 396, ...`
- **Meaning**: How far down from the top of the frame

### 5. **width** (5th column)
- **What it is**: Width of the bounding box
- **Unit**: Pixels
- **Example**: `64, 62, 24, 74, ...`
- **Formula**: x_max = x + width
- **Purpose**: How wide is the player?

### 6. **height** (6th column)
- **What it is**: Height of the bounding box
- **Unit**: Pixels
- **Example**: `88, 110, 74, 98, ...`
- **Formula**: y_max = y + height
- **Purpose**: How tall is the player?

### 7. **confidence** (7th column)
- **What it is**: Detection confidence score
- **Range**: 0 to 1 (typically), or binary: 0 or 1
- **Your data**: Always `1` (100% confident)
- **Meaning**: How sure is the detector this is a real player?

### 8. **class** (8th column)
- **What it is**: Object class label
- **Your data**: Always `1`
- **Possible values**: 1 = Player, 0 = Ball, -1 = Not used, etc.
- **Purpose**: What type of object is this?

### 9. **visibility** (9th column)
- **What it is**: Visibility flag
- **Your data**: Always `1`
- **Meanings**: 1 = Fully visible, 0 = Partially occluded
- **Purpose**: Is the player fully visible in the frame?

---

## Visual Example

Let's visualize one line of data:

```
1, 0, 39, 388, 64, 88, 1, 1, 1
```

### As a diagram:

```
Frame 1 (1280×720 image):

       0 pixels                                    1280 pixels
       ↓                                              ↓
    0  +─────────────────────────────────────────────+
       │                                             │
       │     Player 0:                               │
       │     ┌─ x=39, y=388                          │ 388 pixels down
  388  │     │  width=64  ─────┐                     │
       │     │                  │                    │
       │     │  ┌────────────┐  │                    │
       │     │  │            │  │ height=88          │
       │     │  │  (Player)  │  │                    │
       │     │  │            │  │                    │
       │     └──┤────────────┤──┘                    │
  476  │        └────────────┘   (x+w, y+h)         │
       │        ↑              ↑                     │
       │        39+64=103      388+88=476            │
       │                                             │
  720  +─────────────────────────────────────────────+
```

### Corner coordinates:
- **Top-left**: (x=39, y=388)
- **Top-right**: (x+w=103, y=388)
- **Bottom-left**: (x=39, y+h=476)
- **Bottom-right**: (x+w=103, y+h=476)
- **Center**: ((39+64/2=71, 388+88/2=432))

---

## Key Statistics from Your Data

### Frame 1 Analysis:
```
1, 0, 39, 388, 64, 88, 1, 1, 1
1, 1, 588, 554, 62, 110, 1, 1, 1
1, 2, 424, 108, 24, 74, 1, 1, 1
1, 3, 831, 396, 74, 98, 1, 1, 1
1, 4, 818, 225, 42, 84, 1, 1, 1
1, 5, 721, 188, 36, 80, 1, 1, 1
1, 6, 994, 67, 32, 68, 1, 1, 1
1, 7, 1053, 91, 41, 73, 1, 1, 1
1, 8, 1042, 73, 36, 75, 1, 1, 1
1, 9, 191, 44, 28, 69, 1, 1, 1
1, 11, 1231, 136, 49, 83, 1, 1, 1
```

**In Frame 1:**
- **11 players detected** (track IDs: 0,1,2,3,4,5,6,7,8,9,11)
- **Note**: No track ID 10 = player 10 is off-camera or not yet on field
- **Smallest player**: ID 2 (24×74 pixels) - far or partially visible
- **Largest player**: ID 1 (62×110 pixels) - closest to camera
- **Bounding box areas**:
  - ID 0: 64×88 = 5,632 px²
  - ID 1: 62×110 = 6,820 px² (biggest)
  - ID 2: 24×74 = 1,776 px² (smallest)

---

## How Players Move (Tracking Example)

Same player across different frames:

```
Frame 1: 1, 4, 818, 225, 42, 84, 1, 1, 1   ← Player 4 at (818, 225)
Frame 2: 2, 4, 839, 227, 48, 82, 1, 1, 1   ← Player 4 at (839, 227) - moved right
Frame 3: 3, 4, 861, 228, 50, 83, 1, 1, 1   ← Player 4 at (861, 228) - moved more right
Frame 4: 4, 4, 885, 228, 48, 83, 1, 1, 1   ← Player 4 at (885, 228) - still moving right
Frame 5: 5, 4, 905, 228, 45, 81, 1, 1, 1   ← Player 4 at (905, 228) - continues right
```

**Trajectory of Player 4:**
```
Frame 1: (818, 225) →
Frame 2: (839, 227) →  (moved +21 pixels right, +2 pixels down)
Frame 3: (861, 228) →  (moved +22 pixels right, +1 pixel down)
Frame 4: (885, 228) →  (moved +24 pixels right, no change)
Frame 5: (905, 228) →  (moved +20 pixels right, no change)
```
→ **Player 4 is running to the right across the field!**

---

## Summary Table

| Field | Index | Type | Range | Your Data |
|-------|-------|------|-------|-----------|
| frame_id | 1 | Integer | 1-875 | All values 1-875 |
| track_id | 2 | Integer | 0-19 | IDs 0-19 (20 players) |
| x | 3 | Integer | 0-1280 | Varies |
| y | 4 | Integer | 0-720 | Varies |
| width | 5 | Integer | > 0 | Varies |
| height | 6 | Integer | > 0 | Varies |
| confidence | 7 | Integer | 0 or 1 | Always 1 |
| class | 8 | Integer | 1 | Always 1 (player) |
| visibility | 9 | Integer | 0 or 1 | Always 1 (visible) |

---

## Loading & Parsing Example

```python
# Read gt.txt and parse it
with open('gt/gt.txt', 'r') as f:
    for line in f:
        values = [int(float(x.strip())) for x in line.strip().split(',')]
        
        frame_id, track_id, x, y, w, h, conf, cls, vis = values
        
        print(f"Frame {frame_id}: Player {track_id} at ({x},{y}) size {w}×{h}")
        
        # Calculate bounding box corners
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        
        # Calculate center
        center_x = x + w // 2
        center_y = y + h // 2
        
        print(f"  Corners: ({x_min},{y_min}) to ({x_max},{y_max})")
        print(f"  Center: ({center_x},{center_y})")
```

---

## What This Means for Your CV Project

1. **Multi-target tracking**: Each player has a consistent ID
2. **Bounding box format**: (x_min, y_min, width, height) - not corners!
3. **Complete annotations**: All 875 frames have detections
4. **20 unique objects**: Dynamic number visible per frame
5. **Ready for training**: Ground truth is clean (100% confidence)
