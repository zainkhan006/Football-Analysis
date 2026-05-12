# CV Project — Work Division & Feature Specification

**Team:** Zain Ul Ebad (29217) · Lamaan Ali Bakhsh (29233) · Mohammad Sameem (29209)  
**Dataset:** SportsMOT (football split) + YouTube clips (max 10-minute sequences)  
**Hardware note:** No GPUs on any machine. All models must run on CPU. Use YOLOv8n (nano) throughout.

---

## Part 1 — Feature List

### Feature 1 — Player Detection

YOLOv8n runs on every frame of the input clip and draws bounding boxes around every person visible on the pitch. A "person" here means outfield players, goalkeepers, referees, and fourth officials. The model is not fine-tuned from scratch — a pretrained YOLOv8n checkpoint (trained on COCO, which includes the `person` class) is used directly, since broadcast football footage is well within its training distribution.

Because there is no GPU, inference is run on CPU. To keep processing time reasonable on a 10-minute clip (roughly 15,000 frames at 25 FPS), every Nth frame is processed (N=2 or N=3 depending on measured throughput). For SportsMOT validation, the ground-truth bounding boxes from `gt/gt.txt` are used directly, bypassing detection entirely and allowing the downstream pipeline to be validated cleanly.

**Output per frame:** a list of bounding boxes in pixel coordinates `[x, y, w, h]` with associated confidence scores.

---

### Feature 2 — Multi-Object Tracking

ByteTrack is applied on top of the per-frame detections from Feature 1. ByteTrack associates detections across consecutive frames and assigns each detected person a consistent integer track ID that persists throughout the clip. This is what lets the pipeline say "player with ID 7 was at position (x, y) in frame 100 and position (x', y') in frame 101" — without it, detections are just anonymous boxes with no continuity.

ByteTrack handles brief occlusions well by maintaining a buffer of "lost" tracks and re-associating them when the player reappears. For the manually selected target player (Feature 5 & 6), the user clicks on that player's bounding box in the first frame and the system locks onto that track ID. If ByteTrack loses that specific track due to a long occlusion, the user can re-initialise.

**Output per frame:** same bounding boxes as Feature 1 but each now has an assigned `track_id` that is consistent across the entire clip.

---

### Feature 3 — Pitch Keypoint Detection & Homography Estimation

**Assigned to: Lamaan Ali Bakhsh**

This is the geometric core of the entire pipeline. A homography is a 3×3 matrix that maps any pixel coordinate in the camera image to a real-world coordinate on the pitch (in metres), accounting for the camera's angle, zoom, and position. Without it, pixel movement cannot be converted to real-world distance, and player positions cannot be projected onto a top-down tactical map.

The homography is computed by detecting specific landmarks on the pitch — corner flags, penalty spot, penalty arc endpoints, centre circle intersections, and the four corners of the penalty box — using a YOLOv8n model fine-tuned for keypoint detection on football pitches (pretrained weights available from the SoccerNet pitch localisation challenge). Once at least 4 landmark correspondences are established between their detected pixel positions and their known real-world positions (using FIFA-standard pitch dimensions), OpenCV's `findHomography` with RANSAC computes the matrix.

For static-camera footage (SportsMOT tactical clips), the homography is computed once at the start and held fixed for the entire clip. For YouTube broadcast footage with a moving or panning camera, it is recomputed every K frames (or triggered when the camera moves significantly, as detected by Feature 4).

**Output:** a 3×3 homography matrix `H`. Any pixel point `(px, py)` can be transformed to real-world metres `(wx, wy)` via `cv2.perspectiveTransform`.

---

### Feature 4 — Optical Flow Camera Motion Compensation

When a broadcast camera pans to follow play, every player's pixel position shifts — even stationary ones. A player standing still in the real world will appear to move 50 pixels to the left if the camera pans right. If this is not corrected before computing speed or distance, every player appears to be sprinting whenever the camera moves.

Lucas-Kanade sparse optical flow (via `cv2.calcOpticalFlowPyrLK`) solves this. In each frame, a set of feature points are tracked on stationary background elements — pitch line markings, advertising boards, corner flag bases. These points have zero real-world velocity, so their apparent pixel displacement between frames is entirely due to camera motion. This displacement is computed as a translation vector and subtracted from every player's pixel displacement before any speed calculation.

This feature is most important for YouTube clips where the camera constantly follows the ball. For SportsMOT tactical camera footage (fixed, wide-angle), this compensation is minimal but is still applied for correctness.

**Output:** a per-frame camera motion vector `(dx_cam, dy_cam)` in pixels, used to correct player displacement in Feature 5.

---

### Feature 5 — Player Movement Analysis (Speed, Top Speed, Distance Covered)

Once the player's position is known in real-world metres (via Feature 3's homography) and camera motion is corrected (via Feature 4), the physics is simple arithmetic.

**Distance covered:** For each consecutive pair of frames, the Euclidean distance between the player's real-world foot position is computed. Summing these frame-to-frame displacements across the entire clip gives total distance in metres.

**Speed:** Distance between frames divided by the time interval between frames (1 / frame_rate), then multiplied by 3.6 to convert m/s to km/h. A rolling average over a small window (5–7 frames) smooths noise.

**Top speed:** The maximum recorded speed value over the entire clip, reported in km/h.

**Sprint detection:** Any sustained period (≥10 consecutive frames) where speed exceeds 25 km/h is classified as a sprint. Sprint count and total sprint distance are also logged.

All metrics apply only to the single manually-selected target player. Other players' positions are tracked but their individual movement metrics are not computed (they are used for team assignment and formation analysis instead).

**Output:** `{distance_m, top_speed_kmh, avg_speed_kmh, sprint_count, sprint_distance_m}` for the target player.

---

### Feature 6 — Zone Heatmap & Zone Occupancy

The pitch is divided into a 9-zone grid based on standard football positional analysis conventions. The zones are:

| Zone | Location |
|---|---|
| Defensive Half | Own half, all widths |
| Middle Third | Central band across full width |
| Final Third | Attacking third, all widths |
| Left Flank | Wide left channel |
| Left Half Space | Between left flank and centre |
| Centre | Central corridor |
| Right Half Space | Between right flank and centre |
| Right Flank | Wide right channel |
| Left Advanced / Right Advanced | Overlapping zones for advanced wide positions |

The target player's real-world coordinate every frame is projected onto a top-down pitch template (105m × 68m, standard FIFA dimensions). A Gaussian kernel is applied at each position to accumulate a density map. After processing the full clip, the accumulated density map is rendered as a broadcast-style thermal heatmap (colour gradient: dark navy → blue → cyan → green → yellow → orange → red → white at peak density) overlaid on the pitch template.

Separately, zone occupancy is computed as the percentage of total tracked frames the player spent in each of the 9 zones.

**Output:** a thermal heatmap image, a zone occupancy table `{zone_name: percentage}`, and touch location markers (from Feature 7) overlaid on the same pitch template.

---

### Feature 7 — Touch Detection & Touch Map

A "touch" is registered whenever the ball (tracked by Feature 11) is within a threshold pixel distance of the target player's foot position (the bottom-centre of their bounding box). When this proximity condition is met for at least 2 consecutive frames, one touch is logged.

The real-world coordinate of each touch (obtained via homography) is stored. These touch locations are rendered as small markers on the same top-down pitch template as the heatmap (Feature 6). Summary statistics — total touch count, touches by zone, touches per minute — are computed from the log.

**Output:** `{touch_count, touch_locations_list, touches_by_zone}`.

---

### Feature 8 — Team Assignment via K-Means HSV Clustering

The 22 outfield players plus the referee need to be labelled by which team they belong to. This is done entirely from visual appearance, without any jersey number reading.

For each tracked person, the top half of their bounding box (the jersey region, avoiding shorts and socks) is cropped. The crop is converted from BGR to HSV colour space. HSV is used rather than RGB because the H (hue) channel is robust to lighting variation — a green jersey under bright sunlight and a green jersey under stadium floodlights have similar hue values but very different RGB values. The mean HSV values of the crop are computed and collected across all tracked persons in a given frame.

K-means clustering with k=3 is run on these HSV vectors: Cluster 1 = home team, Cluster 2 = away team, Cluster 3 = referee/GK (since referees and goalkeepers often wear a third distinct colour). The cluster assignments are propagated to subsequent frames via track ID continuity — once a track ID is assigned to a cluster in one frame, it keeps that label unless the colour changes significantly (which handles the case of substitutes entering mid-clip with a different jersey).

**Output:** per-frame label `{track_id: "home" | "away" | "referee/GK"}`.

---

### Feature 9 — Jersey Colour Classification & Clash Detection

This builds on top of Feature 8's clustering to produce human-readable jersey metadata.

**Home/Away/Alternate identification:** Once the two team clusters are established, the dominant hue of each cluster's centroid is looked up against a simple colour name mapping (e.g., hue 0–15° = red, 15–45° = orange, 45–75° = yellow, 75–150° = green, 150–260° = blue/purple, 260–300° = pink/magenta, 300–360° = red again). This gives a plain-language label: "Team A: blue jerseys, Team B: red jerseys."

**GK identification:** Goalkeepers almost always wear a jersey colour distinct from both outfield teams. Among the tracks assigned to a given team cluster, the goalkeeper is typically the one whose jersey colour is a statistical outlier relative to the cluster centroid. A simple distance threshold in HSV space identifies them.

**Referee identification:** Referees are usually isolated in their own cluster (the k=3 third cluster). If the third cluster contains only 1–2 tracks, those are referees.

**Clash detection:** If the HSV distance between the two team cluster centroids falls below a threshold (meaning the two teams' jersey colours are too visually similar), a clash warning is raised. This is relevant for clips where teams are wearing similar colours, which can degrade team assignment accuracy. The threshold is tuned to approximately 30 HSV hue degrees of separation.

**Output:** `{home_colour, away_colour, home_GK_colour, referees_identified, clash_warning: bool}`.

---

### Feature 10 — Formation Analysis (Both Teams)

Formation analysis runs on all tracked players (not just the target player), divided by team label from Feature 8.

For each frame, the real-world (x, y) coordinates of all outfield players from one team are projected onto the top-down pitch template. A temporal window of frames (e.g., 50 frames = 2 seconds) is used to smooth over transient movements — formations are positional tendencies, not frame-by-frame snapshots.

The 10 outfield players (excluding GK, who is identified by Feature 9) are sorted by their average x-coordinate (depth on the pitch) and clustered into lines using a simple 1D k-means on the x-axis. The number of clusters (defensive line, midfield line, attacking line) is determined by the data — a 2-cluster solution gives a 2-line formation (like 4-6 or 5-5), a 3-cluster solution gives a 3-line formation (like 4-4-2, 4-3-3), and a 4-cluster solution gives a 4-line formation (like 4-2-3-1). The number of players in each cluster gives the formation string (e.g., "4-3-3", "4-4-2", "3-5-2").

Formation is computed separately for each team, and can be computed for different phases: in-possession (frames where the team's cluster has the ball nearby) vs. out-of-possession.

**Output:** `{team_A_formation, team_B_formation}` per phase, plus a top-down visualisation showing player positions and formation lines.

---

### Feature 11 — Ball Detection & Tracking

A YOLOv8n model is used to detect the ball in each frame. Since the COCO-pretrained YOLOv8n does not have a football class, a specialised ball detection checkpoint is used — fine-tuned on publicly available football detection datasets (Roboflow's football dataset, which does not require GPU to run inference on). If the pretrained checkpoint performs well enough, no further fine-tuning is done.

Ball detection is harder than player detection: the ball is small, fast, and frequently occluded (behind players, net, or out of frame). To handle missed frames, a Kalman filter maintains a predicted ball position when detection fails for up to 10 consecutive frames. This predicted position is flagged as an estimate, not a confirmed detection.

Ball tracking uses a single-object tracker (since there is only one ball) rather than ByteTrack. The ball's pixel position per frame is passed to Feature 7 (touch detection) and Feature 10 (possession phase detection for formation analysis).

**Output per frame:** `{ball_px, ball_py, confidence, is_estimated: bool}`.

---

## Part 2 — Work Division

### Zain Ul Ebad (29217)

**Owns:** Features 1, 2, 4, 5, 6 + Pipeline Integration (Phase L4)  
**Role:** Foundation pipeline — everything downstream depends on Zain's output, + final assembly.

Zain has the SportsMOT dataset downloaded and is the only team member who can do dataset-level validation. His work produces the two core data structures the entire project relies on:
- Per-frame detection + tracking data (player positions with consistent IDs)
- Real-world coordinate mapping via homography
- Integrating all modules at the end.

Once Zain's modules are producing clean output, Sameem plugs team labels in and Lamaan plugs ball positions in.

---

### Mohammad Sameem (29209)

**Owns:** Features 8, 9, 10  
**Role:** Team intelligence — labelling who is who and what shape each team is in.

Sameem does not need the SportsMOT dataset. He can develop and test Features 8 and 9 using any football frame grabbed from YouTube (a single still image is enough to develop the jersey clustering logic). Feature 10 needs real-world player coordinates, which comes from Zain — but Sameem can develop the formation algorithm using a hardcoded mock coordinate input while waiting.

---

### Lamaan Ali Bakhsh (29233)

**Owns:** Features 3, 7, 11  
**Role:** Ball intelligence

Lamaan does not need the SportsMOT dataset. Ball detection development and testing can be done on any football clip from YouTube — the model runs on single images. Touch detection logic is pure geometry and can be tested with mock coordinate inputs.
---

## Part 3 — Phases Per Team Member

### Zain — Phases

> All of Zain's phases can run independently from the start.

**Phase Z1 — Environment & Dataset Validation**  
Install YOLOv8, ByteTrack, and OpenCV in the `tf` environment. Load the SportsMOT football split and verify the directory structure matches the expected MOT Challenge 17 format (this was already confirmed during EDA). Write a loader that reads `gt/gt.txt` and `seqinfo.ini` for a given sequence.

_Deliverable: working dataset loader, confirmed environment._

**Phase Z2 — Player Detection (YOLOv8n)**  
Load a pretrained YOLOv8n checkpoint. Run inference on a SportsMOT clip and on a short YouTube clip. Filter detections to the `person` class only. Evaluate on SportsMOT using the GT boxes as ground truth: compute detection precision and recall. Tune the confidence threshold to balance false positives (detecting crowd) vs. false negatives (missing players).

_Deliverable: `detect(frame) → [(x, y, w, h, confidence), ...]`. Precision/recall numbers on SportsMOT._

**Phase Z3 — Multi-Object Tracking (ByteTrack)**  
Integrate ByteTrack on top of Z2's detections. Verify that track IDs are consistent across frames by visually overlaying ID labels on a clip. On SportsMOT, compare assigned track IDs against GT track IDs to measure ID switch count. Implement the target-player lock-on: the user provides a bounding box in frame 1 and the system identifies and locks the closest-matching track ID.

_Deliverable: `track(frames) → [{frame_id, track_id, bbox}]`. ID switch count on SportsMOT._

**Phase Z4 — Pitch Keypoint Detection & Homography**  
Load a YOLOv8n-pose checkpoint pretrained on football pitch landmarks (SoccerNet). Run keypoint detection on a SportsMOT frame and on a YouTube clip. Collect at least 4 landmark pixel positions and their known real-world positions. Compute H via `cv2.findHomography(..., cv2.RANSAC)`. Validate by projecting known pitch landmarks back through H and measuring reprojection error in centimetres.

_Deliverable: `compute_homography(frame) → H (3x3 matrix)`. Reprojection error ≤ 50 cm._

**Phase Z5 — Optical Flow Camera Motion Compensation**  
Implement Lucas-Kanade optical flow on background pitch features. Extract stationary-background keypoints (using `cv2.goodFeaturesToTrack` on the previous frame), track them into the next frame, and compute the median displacement vector as the camera motion vector. Subtract this from all player displacements before any speed calculation. Test by deliberately panning a static video and confirming that a stationary player's computed speed is near zero.

_Deliverable: `camera_motion(prev_frame, next_frame) → (dx_cam, dy_cam)`._

**Phase Z6 — Speed, Distance, Sprint Detection**  
For the target player's track, apply H to convert their foot pixel position (bottom-centre of bbox) to real-world metres each frame. Apply camera motion correction from Z5. Compute frame-to-frame displacement, apply a 5-frame rolling average to smooth, then derive speed in km/h. Accumulate total distance. Flag sprints (≥10 consecutive frames above 25 km/h). Output the full metrics dict.

_Deliverable: `{distance_m, top_speed_kmh, avg_speed_kmh, sprint_count, sprint_distance_m}`._

**Phase Z7 — Zone Classification & Thermal Heatmap**  
Define the 9-zone grid on the 105×68m pitch coordinate space. For each frame, map the target player's real-world position to a zone. Accumulate a 2D Gaussian kernel at each position on a pitch-sized density array. At the end of the clip, normalise and render the density map as a thermal heatmap (navy → white colour ramp) overlaid on a pitch template image. Render zone occupancy as a bar chart.

_Deliverable: heatmap image, `{zone_name: percentage}` dict._

**Handoff to team:** After Z3 and Z4, Zain produces a per-frame JSON/dict: `{frame_id, track_id, bbox_px, world_xy, speed_kmh, zone}`. This is the data structure Sameem and Lamaan build on top of.

---

### Sameem — Phases

> Phases S1–S4 are fully independent. S5–S7 need Zain's `world_xy` output.

**Phase S1 — Jersey Crop Pipeline**  
Write a function that takes a bounding box `(x, y, w, h)` and an image frame, crops the top 40% of the bounding box (the jersey region), resizes it to a fixed size (e.g., 32×32), converts BGR→HSV, and returns the per-pixel HSV values. Test on manually grabbed football frames from YouTube. Confirm that the crop correctly captures the jersey and excludes shorts, socks, and pitch background.

_Deliverable: `crop_jersey(frame, bbox) → hsv_array`._

**Phase S2 — K-Means Team Assignment**  
For each frame, collect jersey HSV crops from all tracked bounding boxes. Compute the mean HSV vector per crop. Run `sklearn.cluster.KMeans(n_clusters=3)` on these vectors. Label the three clusters as home, away, and referee/GK (the cluster with the fewest members is referee/GK in most frames). Propagate cluster labels via track ID — once a track ID has a stable label across 5+ frames, lock it. Test on a YouTube clip with clearly contrasting jersey colours first, then test on a harder clip.

_Deliverable: `assign_teams(frame, bboxes, track_ids) → {track_id: "home"|"away"|"ref_gk"}`._

**Phase S3 — Jersey Colour Naming**  
Map each team cluster's centroid HSV hue to a colour name using a lookup table (hue ranges to colour strings: red, orange, yellow, green, blue, purple, pink). Output a human-readable string per team: "Team A: blue jerseys, Team B: red jerseys."

_Deliverable: `{team_A_colour_name, team_B_colour_name}`._

**Phase S4 — GK, Referee Identification & Clash Detection**  
Within each team cluster, identify the goalkeeper as the track whose jersey HSV is a statistical outlier from the cluster centroid (distance threshold in HSV space). The referee/GK cluster (k=3 output) contains both referees and GKs — split them by track count (≥3 tracks = includes GK, usually; referees are typically 2–3 individuals). For clash detection, compute the HSV hue distance between the two team centroids. If below 30 hue degrees, raise a `clash_warning = True` flag.

_Deliverable: `{home_GK_trackid, away_GK_trackid, referee_trackids, clash_warning}`._

**Phase S5 — Formation Grid & Player Positioning (needs Zain's Z4 output)**  
Using Zain's `world_xy` per frame for all tracked players, separate players by team label (from S2). For each team, exclude the GK track (from S4). Apply a temporal smoothing window of 50 frames to get stable average positions. Sort the 10 outfield player positions by their x-coordinate (depth on pitch). This sorted list is the input to the formation classifier.

_Can do without Zain: implement with hardcoded mock world coordinates (e.g., a numpy array of 10 (x,y) pairs). Swap in real data when Zain delivers Z4._

_Deliverable: per-team list of smoothed `[(world_x, world_y)]` for 10 outfield players._

**Phase S6 — Formation Line Detection**  
Run 1D k-means on the x-coordinates of the 10 outfield players to detect positional lines (defensive, midfield, attacking). Try k=2, k=3, k=4 and select the best k using the elbow method on within-cluster sum of squares. Count players per cluster to produce the formation string (e.g., k=3 with counts [4, 3, 3] → "4-3-3", counts [4, 4, 2] → "4-4-2").

_Deliverable: `{team_A_formation: "4-3-3", team_B_formation: "4-4-2"}`._

**Phase S7 — Formation Visualisation & Phase Separation (needs Lamaan's L2 output)**  
Draw the detected formation on the top-down pitch template: player dots coloured by team, lines connecting players in the same detected cluster. Optionally separate in-possession vs. out-of-possession formation if ball position (from Lamaan's L2) is available to determine which team has the ball in a given window.

_Can do without Lamaan: visualise formation without phase separation first, add phase split after Lamaan delivers L2._

_Deliverable: formation overlay image._

---

### Lamaan — Phases

> Phases L1–L3 are fully independent. L4–L5 need outputs from both Zain and Sameem.

**Phase L1 — Ball Detection**  
Load a YOLOv8n checkpoint pretrained on sports ball detection (available from Roboflow Universe without GPU-intensive training). Run inference on a football YouTube clip. Filter to the ball class. Tune confidence threshold — err on the side of higher precision (fewer false positives) since a missed ball frame is recoverable by the Kalman filter, but a false positive (detecting a player's head as a ball) corrupts touch detection.

_Deliverable: `detect_ball(frame) → (ball_x, ball_y, confidence) or None`._

**Phase L2 — Ball Tracking with Kalman Filter**  
Wrap ball detection in a Kalman filter (`cv2.KalmanFilter`, 4-state: x, y, dx, dy). When detection succeeds, update the filter. When detection fails for up to 10 consecutive frames, use the filter's prediction as an estimated position (flagged `is_estimated=True`). After 10 consecutive misses, reset the filter. Output the ball's pixel position (real or estimated) per frame.

_Deliverable: `{frame_id, ball_px, ball_py, is_estimated}`._

**Phase L3 — Touch Detection Logic**  
A touch is logged when: (1) the ball is detected with `is_estimated=False`, and (2) the Euclidean pixel distance between the ball centre and the target player's foot position (bottom-centre of their bbox) is below a threshold (tune between 30–60 pixels, since the ball is about 20–25 pixels in diameter in typical broadcast footage), and (3) this proximity condition holds for at least 2 consecutive frames (to reject single-frame noise).

_Develop with mock inputs: generate fake ball positions and bbox positions in a test script, confirm touch detection fires at the right moments._

_Deliverable: `detect_touch(ball_pos, player_foot_pos, prev_frames) → bool`. List of `{frame_id, touch_world_xy}` for the full clip._

**Phase L4 — Pipeline Integration (needs Z3, Z4, Z7 from Zain + S2, S4 from Sameem)**

**Assigned to: Zain Ul Ibad**

Write the master `pipeline.py` that connects all modules in sequence:

```
input clip
  → Z2: detect players
  → Z3: track players (get track IDs)
  → Z4: compute homography
  → Z5: compute camera motion per frame
  → Z6: compute speed/distance for target player
  → Z7: compute zone and accumulate heatmap
  → S2: assign team labels per track ID
  → S4: identify GK, referees, clash warning
  → L1/L2: detect and track ball
  → L3: detect touches for target player
  → S5/S6: compute formation per team
output: metrics dict + heatmap image + formation image
```

Define a clear shared data contract: the per-frame dict that all modules read from and write to, so each module can be swapped independently. Handle the case where ball detection fails entirely for the clip (gracefully skip touch detection and formation phase split rather than crashing).

_Deliverable: working end-to-end `pipeline.py` that takes a clip path and outputs all results._

**Phase L5 — Output Rendering**  
Produce the final visual outputs:

- **Annotated video:** original clip with bounding boxes, track IDs, team colour labels, target player highlighted, ball position, and real-time speed overlay — rendered as an output `.mp4`.
- **Stats card:** static image with the target player's metrics (distance, top speed, sprint count, touch count, zone occupancy).
- **Heatmap image:** the thermal heatmap with touch markers from Zain's Z7 and Lamaan's L3 combined.
- **Formation image:** top-down pitch view with both teams' formations from Sameem's S7.

_Deliverable: output directory containing `annotated_clip.mp4`, `stats_card.png`, `heatmap.png`, `formation.png`._

---

## Dependency Summary

```
Z1 → Z2 → Z3 ─────────────────────────────────────┐
          ↓                                         │
          Z4 → Z5 → Z6                              │
          ↓                                         │
          Z7 (heatmap)                              │
                                                    ↓
S1 → S2 → S3 → S4 ──────────────────────────── L4 (integration)
          ↓ (needs Z4)                              ↑
          S5 → S6 → S7                              │
                                                    │
L1 → L2 → L3 ──────────────────────────────────────┘
                                                    ↓
                                                   L5 (output)
```

Phases that can run in parallel without waiting:
- **Z1–Z7** (all Zain, sequential within his own track)
- **S1–S4** (Sameem, no external dependencies)
- **L1–L3** (Lamaan, no external dependencies)

Phases that require handoff:
- **S5–S7** start after **Z4** is done (Zain sends the homography-transformed coordinates)
- **L4** starts after **Z3, Z4, Z7** (Zain) and **S2, S4** (Sameem) are both done
- **L5** starts after **L4** and **S7** are both done

---

## Clip Processing Note

The pipeline processes clips up to 10 minutes (≈15,000 frames at 25 FPS). On CPU without a GPU, YOLOv8n inference takes approximately 60–150ms per frame depending on machine. Processing every frame would take 15–37 minutes for a 10-minute clip. The recommended approach is to process every 2nd frame (stride=2), which halves processing time with negligible loss of tracking accuracy since players do not move discontinuously between adjacent frames. Speed calculations adjust for the effective frame rate (25/stride FPS).