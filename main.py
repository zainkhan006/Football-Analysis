import cv2
import numpy as np
import os

# Path to one football sequence
SEQ_PATH = r"C:\Users\Zain Ul Ibad\Desktop\projects\cv_project\sportsmot_publish\dataset\val\v_i2_L4qquVg0_c006"

img_path = os.path.join(SEQ_PATH, "img1")
ann_path = os.path.join(SEQ_PATH, "gt", "gt.txt")

# Load annotations
anns = np.loadtxt(ann_path, delimiter=",")

# Group annotations by frame
from collections import defaultdict
frame_anns = defaultdict(list)
for row in anns:
    frame_id = int(row[0])
    track_id = int(row[1])
    x, y, w, h = row[2], row[3], row[4], row[5]
    frame_anns[frame_id].append((track_id, x, y, w, h))

# Visualize first 100 frames
for frame_id in sorted(frame_anns.keys()):
    img_file = os.path.join(img_path, f"{frame_id:06d}.jpg")
    frame = cv2.imread(img_file)
    if frame is None:
        print(f"Could not load frame {frame_id}")
        continue

    for (track_id, x, y, w, h) in frame_anns[frame_id]:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("SportsMOT Football Test", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()