# CV Project

## Overview

This project combines two complementary computer vision ideas into a single unified pipeline: **single-player match analysis** and **team formation mapping**. Together, they produce a system that can take a football video clip as input and output both a detailed analytical profile of one chosen player and a tactical bird's-eye map showing how both teams are structured and moving relative to each other.

The motivation for combining the two is that their technical foundations are almost entirely shared. Both require player detection, multi-object tracking, and homography estimation to project pixel coordinates into real-world pitch coordinates. Once that shared infrastructure is built, the player analysis and the formation analysis are simply two different analytical layers sitting on top of the same pipeline — they do not need to be built separately, and building them together means the overall project is more substantial without being proportionally harder.

The input is a video clip sourced from YouTube, shot from a static tactical camera that captures the full pitch in a single wide-angle frame. The output is a dual-view dashboard: one panel showing the original video with CV overlays, and a second panel showing a top-down tactical map with the heatmap, movement trail, touch markers, and formation lines derived from it.

---

## Dataset

### Primary dataset — SportsMOT

SportsMOT is the primary formal benchmark dataset for this project. It provides short football video clips with per-frame bounding box annotations and consistent player track IDs maintained across frames. Because the bounding boxes are pre-annotated, the project uses SportsMOT specifically to validate the analytical pipeline — that is, the homography estimation, speed and distance calculation, zone occupancy classification, and heatmap generation — without needing to also solve player detection on that footage at the same time. This allows clean, quantitative evaluation of each analytical stage against ground-truth position data.

SportsMOT does not provide team colour labels, ball position annotations, jersey number data, or pitch keypoint annotations. This means the K-means team assignment, ball detection, and homography estimation modules must be implemented independently and validated on real match footage.

### Secondary footage — YouTube clips

YouTube clips sourced from publicly available football match broadcasts serve as the real-world validation footage, particularly for the formation mapping component and for demonstrating the full end-to-end pipeline. These clips provide the team colour variation, pitch line geometry, and tactical context that SportsMOT does not supply. The two sources are complementary rather than redundant: SportsMOT validates the analytics rigorously, and the YouTube clips demonstrate the system working in realistic conditions.

---

## Implementation

### How the pipeline works

The pipeline is best understood as a sequence of stages where each stage produces outputs that feed directly into the next. Nothing is skipped and nothing is assumed — every piece of information the system uses about players, pitch geometry, or team identity is computed from the raw video.

**Stage 1 — Player detection.** YOLOv8 is run on every frame of the input video clip. It detects all people on the pitch — outfield players, goalkeepers, and referees — and outputs a bounding box for each detection, defined by pixel coordinates and a confidence score. The model is used in its pre-trained form for person detection and does not require custom training for this stage.

**Stage 2 — Multi-object tracking.** ByteTrack is applied to the stream of per-frame detections. It assigns a persistent numerical track ID to each detected player and maintains that ID across frames using Intersection-over-Union matching and Kalman filter motion prediction. Even when a player is briefly occluded, ByteTrack uses low-confidence detections to keep the track alive rather than terminating it. The output of this stage is a data structure mapping each track ID to a sequence of bounding boxes over time. For the player analysis component, the user selects their target player by clicking on a bounding box in the first frame, and the corresponding track ID is locked for the rest of the sequence.

**Stage 3 — Team assignment via K-means HSV clustering.** Once bounding boxes are established, the jersey colour of each player is extracted by cropping the upper half of their bounding box — this isolates the torso and jersey while excluding the shorts and pitch background, which would otherwise contaminate the colour signal. Each cropped region is converted from RGB to HSV colour space, because HSV separates colour (hue) from lighting conditions (value and saturation), making it more robust to the varying illumination of outdoor football footage. The dominant HSV colour of each crop is passed to a K-means classifier with k=3, producing three clusters: home team, away team, and a third cluster that catches referees and goalkeepers in distinctive colours. This assignment is computed once early in the sequence and used to colour-code all downstream visualisations.

**Stage 4 — Homography estimation.** A homography is a 3×3 matrix that mathematically describes a projective transformation between two flat planes — in this case, between the camera's perspective view of the pitch and a flat top-down template of a standard FIFA pitch with known real-world dimensions (105m × 68m). Computing this matrix automatically from video requires identifying at least four corresponding points between the camera image and the template. A YOLOv8 model fine-tuned on pitch landmark detection identifies up to 32 characteristic keypoints on the pitch — corner flags, penalty spot, centre circle intersections, penalty arc, and touchline corners of the penalty areas. These detected pixel coordinates are paired with their known real-world metric coordinates on the template, and OpenCV's `findHomography` function with RANSAC-based outlier rejection solves for the 3×3 matrix. Because the camera is static, this matrix is computed once per clip and held constant throughout. The quality of the estimation is evaluated using reprojection error — the average distance in centimetres between where known pitch points project to under the computed H matrix versus where they actually appear on the template.

**Stage 5 — Coordinate projection.** With the homography matrix established, the foot position of every tracked player (the bottom-centre of their bounding box, which approximates ground contact) is multiplied by H to produce their real-world (x, y) coordinates in metres on the pitch template. All subsequent analysis — speed, distance, zone classification, heatmap generation, and formation detection — operates on these real-world coordinates rather than on pixel positions. This is what makes the system's measurements physically meaningful.

**Stage 6 — Player analysis (target player).** For the user-selected target player, the system computes a set of metrics frame by frame. Distance covered is the sum of Euclidean displacements between consecutive real-world positions: the distance between the player's real-world coordinate at frame n and frame n-1, accumulated over the clip. Instantaneous speed is that displacement divided by the inter-frame time interval (1/FPS), then multiplied by 3.6 to convert metres per second to km/h. The pitch is divided into a 3×2 grid of zones — defensive third, middle third, and attacking third on one axis; left channel, central lane, and right channel on the other — and the player's zone at each frame is logged to produce zone occupancy percentages. A thermal heatmap is generated by accumulating Gaussian kernel density contributions at the player's real-world position every frame, then rendering the result using a black-to-navy-to-blue-to-cyan-to-green-to-yellow-to-orange-to-red-to-white thermal colour ramp. Touch detection is implemented by checking whether the ball's real-world position falls within a threshold distance of the player's feet at each frame.

**Stage 7 — Formation mapping (both teams).** All 22 tracked players are projected into real-world coordinates simultaneously. Team assignment from Stage 3 colour-codes each player's position. The system divides the 90-second clip into sliding windows and, within each window, computes the average real-world position of each player. Players are then sorted by their average x-coordinate (horizontal pitch position) and grouped into defensive, midfield, and forward lines using a grid-based zone occupancy count. The line groupings produce a formation label — for example, if four players average in the defensive third, three in the midfield band, and three in the attacking third, the system classifies this as a 4-3-3. An artificial neural network trained on labelled formation distributions maps the zone-occupancy vector to one of a predefined set of formation strings (4-3-3, 4-4-2, 4-2-3-1, 3-5-2, 5-3-2). Dashed formation lines connecting players in the same tactical line are drawn on the top-down view, providing a visual representation of the inferred shape.

**Stage 8 — Visualisation and output.** The final output is a dual-panel display. The left panel renders the original video frame with all CV overlays drawn on top: colour-coded bounding boxes for all 22 players, a pulsing highlight ring around the target player, a speed label above the target player's box, and a touch flash when a touch is detected. The right panel renders the top-down tactical map with the thermal heatmap, the target player's movement trail, touch markers (drawn as white X symbols at each touch location), ghost dots for all other players, and formation lines connecting teammates. A colour scale bar is displayed alongside the heatmap. Four statistical panels accompany the visualisation: live player stats (distance, current speed, peak speed, touch count, sprint count), zone occupancy percentages, a speed-profile chart over time, and a CV pipeline log showing processing events as they occur.

---

## Workflow (Phases)

### Phase 1 — Environment setup and dataset preparation

Install all required libraries: Python 3.10+, OpenCV, NumPy, Matplotlib, Ultralytics (YOLOv8), a ByteTrack implementation, SciPy, and scikit-learn. Download the SportsMOT dataset and organise clips into a local directory structure. Download or source 2–3 YouTube football clips shot from a static tactical camera for real-world validation. Confirm YOLOv8 runs successfully on a sample frame and produces bounding boxes at acceptable inference speed. Write a data loader that reads SportsMOT annotation files and yields per-frame bounding box dictionaries.

### Phase 2 — Homography estimation module

Implement pitch keypoint detection using YOLOv8 fine-tuned on pitch landmarks — either using an existing Roboflow checkpoint or annotating a small set of frames manually. Write the homography solver using OpenCV's `findHomography` with RANSAC. Define the full-pitch template with hardcoded FIFA-standard real-world coordinates for each keypoint class. Implement reprojection error evaluation. Validate on at least five frames from different YouTube clips, targeting a reprojection error below 15 cm. Since the camera is static, confirm the matrix is stable across frames and implement a single-frame compute-then-hold strategy.

### Phase 3 — Tracking pipeline on SportsMOT

Using SportsMOT's ground-truth bounding boxes as input (bypassing detection), integrate ByteTrack to assign persistent track IDs. Implement the coordinate projection module: for each track ID and frame, multiply the foot position by H to produce real-world (x, y) in metres. Compute per-player speed and distance from real-world displacement. Validate speed estimates by checking they fall within physically plausible ranges for football players (0–34 km/h). Implement zone classification and accumulate zone occupancy fractions per track. This phase validates the entire analytics pipeline on clean data before introducing detection noise.

### Phase 4 — Detection pipeline on YouTube clips

Run YOLOv8 person detection on the YouTube clips. Integrate ByteTrack for multi-object tracking on the raw detections. Implement K-means HSV team assignment on jersey crops. Implement the user-click interface that allows the user to select a target player by clicking on a bounding box in the first frame. Validate that the selected track is maintained throughout the clip with acceptable identity switches. If identity switches occur near the target player's track, implement a re-initialisation mechanism.

### Phase 5 — Player analysis features

Implement the full player analysis stack: thermal heatmap accumulation and rendering, movement trail drawing, touch detection against ball position, sprint detection thresholding (above 21 km/h), and zone occupancy visualisation. Implement the speed profile chart rendered as a canvas overlay. Test on both SportsMOT clips (using ground-truth tracks) and YouTube clips (using predicted tracks). Compare SportsMOT results against ground truth to compute quantitative accuracy metrics for speed estimation.

### Phase 6 — Formation mapping

Implement simultaneous real-world projection of all 22 tracked players. Apply team colour labels from K-means to separate the two squads. Implement the sliding-window averaging logic that computes mean player positions over time. Write the zone-occupancy-to-formation-label classifier — either a lightweight ANN trained on synthetic formation distributions or a rule-based system as a baseline. Render formation lines on the top-down view and display the formation string label for each team. Validate on YouTube clips where the formation is visually identifiable.

### Phase 7 — Extra features implementation

Implement the five extra features detailed in the section below, one at a time, each with its own validation step. Each extra feature is independent and can be added without destabilising the core pipeline.

### Phase 8 — Integration, evaluation, and report

Integrate all modules into a single runnable script with command-line arguments for input video path and target player selection mode. Run the full pipeline on at least three clips. Record quantitative results: reprojection error, speed estimation error against SportsMOT ground truth, formation classification accuracy, and team assignment accuracy. Write the LaTeX project report.

---

## Core Features

**Dual-panel real-time visualisation.** The left panel renders the input video with CV overlays. The right panel renders the top-down analytical map derived from homography projection.

**YOLOv8 player detection.** Detects all players on the pitch per frame, outputting bounding boxes with confidence scores.

**ByteTrack multi-object tracking.** Maintains persistent track IDs across the full clip using IoU matching and Kalman filter motion prediction.

**Homography estimation from pitch keypoints.** Detects pitch landmarks automatically and computes the 3×3 perspective transformation matrix mapping pixel space to real-world metre coordinates.

**User-selected target player tracking.** The user clicks on any player in the first frame to lock a track ID, which is then followed for the entire sequence.

**K-means HSV team assignment.** Clusters players into home, away, and referee groups based on jersey colour in HSV space.

**Thermal heatmap.** Accumulates the target player's real-world position over the clip and renders it using a broadcast-style black-to-white thermal colour ramp (navy → blue → cyan → green → yellow → orange → red → white).

**Movement trail.** Draws the target player's recent trajectory as a fading coloured path on both the camera view and the top-down map.

**Touch detection.** Logs frames where the ball's position falls within a threshold distance of the target player's feet, marking touch locations on the tactical map.

**Speed and distance metrics.** Computes instantaneous speed in km/h, peak speed, total distance covered in metres, and sprint count (frames above 21 km/h) from real-world displacements.

**Zone occupancy.** Classifies the target player's pitch position into a 3×2 grid of zones every frame and reports occupancy percentages.

**Speed profile chart.** Renders a live time-series chart of the target player's speed with threshold lines for walking, jogging, running, and sprinting bands.

**Formation mapping.** Projects all 22 players into real-world coordinates simultaneously, assigns team labels, averages positions over a sliding window, and classifies the formation of each team using a zone-occupancy vector fed to an ANN classifier.

**Formation line overlay.** Draws dashed connecting lines between players in the same tactical line on the top-down map, with the inferred formation string (e.g., 4-3-3) labelled for each team.

**Pipeline status log.** Displays a running log of processing events as they occur: detection count, track initialisation, H matrix computation, team assignment convergence, touch events, and sprint detections.

---

## Extra Features

These five features are additional contributions that go beyond the core pipeline. They are included specifically to compensate for the pre-annotated bounding boxes in the SportsMOT dataset, each representing a distinct and non-trivial CV implementation.

### Extra Feature 1 — Jersey colour team assignment via K-means HSV clustering

Although SportsMOT provides bounding boxes, it does not provide team labels. Implementing K-means clustering on HSV jersey crops — cropping the top half of each bounding box, converting to HSV, extracting the dominant colour, and clustering into k=3 groups — is a real implementation that must be built from scratch. It is also a prerequisite for the formation mapping component, since formation analysis is meaningless without knowing which players belong to which team. Evaluation is done by manually labelling a sample of frames and computing team assignment accuracy.

### Extra Feature 2 — Ball detection module

SportsMOT has no ball annotations whatsoever. A separate YOLOv8 model, either fine-tuned on a ball-annotated dataset sourced from Roboflow or trained on manually annotated frames from the YouTube clips, is integrated into the pipeline to detect the ball's position each frame. This ball position feeds directly into the touch detection feature and also enables future extensions such as ball trajectory analysis and possession tracking. The ball detection module is evaluated by measuring detection rate and localisation accuracy on held-out frames.

### Extra Feature 3 — Pitch keypoint detection for homography

The most technically demanding extra feature. SportsMOT bounding boxes tell you where players are in pixel space but say nothing about pitch geometry. Detecting pitch landmarks — the 32 characteristic points described above — and solving for the homography matrix is the foundational step that makes all real-world metric calculations possible, and the dataset does not do this for you. The implementation uses a YOLOv8 keypoint detection model. Quality is measured quantitatively via reprojection error in centimetres.

### Extra Feature 4 — Optical flow camera motion compensation

For YouTube clips where the camera may pan or zoom slightly even on a so-called static mount, naive pixel displacement calculations will misattribute camera motion as player motion. The Lucas-Kanade optical flow algorithm is used to track stationary background features — pitch markings, advertising boards, corner flags — between consecutive frames. The estimated camera displacement vector is subtracted from each player's pixel displacement before the homography-based speed calculation is applied. This ensures that a stationary goalkeeper does not appear to be moving when the camera drifts. This feature is only activated for clips where camera motion is detected above a threshold; for truly static footage it has no effect. Evaluation compares speed estimates on clips with and without the compensation applied.

### Extra Feature 5 — Voronoi-based territory visualisation

Given all 22 players' real-world coordinates projected onto the pitch template at any given frame, a Voronoi tessellation is computed using SciPy's `Voronoi` function. Each cell in the tessellation represents the region of the pitch closest to that player — in other words, the territory that player is responsible for defending or could reach first. Cells are colour-coded by team and overlaid transparently on the top-down tactical map. This produces the kind of pitch control visualisation used by Second Spectrum in Premier League broadcasts. It is computed per-frame and rendered as a semi-transparent overlay that can be toggled on or off. The implementation requires clipping Voronoi cells to the pitch boundary polygon, which is a non-trivial geometric operation handled using Shapely.

---

## Tools and Libraries

The project is implemented in Python. The core libraries used are OpenCV for image processing and homography computation, Ultralytics for YOLOv8 detection and keypoint detection, ByteTrack for multi-object tracking, NumPy for all array and matrix operations, Matplotlib for heatmap and chart rendering, scikit-learn for K-means clustering and the ANN formation classifier, SciPy for Voronoi tessellation in the extra feature, and Shapely for pitch boundary polygon clipping.

---

## Evaluation Metrics

The pipeline is evaluated quantitatively at each stage. Homography quality is measured by reprojection error in centimetres, targeting below 15 cm on static footage. Speed estimation accuracy is evaluated on SportsMOT clips by comparing computed speeds to ground-truth displacements, reported as mean absolute error in km/h. Team assignment accuracy is evaluated by comparing K-means output to manually labelled ground-truth team identities on a sample of frames, reported as percentage correct. Formation classification accuracy is evaluated by comparing the system's formation string output to the visually ground-truthed formation on each YouTube clip. Ball detection is evaluated by detection rate and mean localisation error on held-out annotated frames.

---

## Scope and Limitations

The pipeline operates on shorter video sequences rather than full match clips. This is a deliberate scope decision that keeps implementation manageable without sacrificing the quality of any individual output. All outputs — heatmaps, formation maps, speed profiles — are generated from the full length of whatever clip is provided, so extending to longer clips is a matter of processing time rather than architectural change.

The system assumes a static or near-static tactical camera with the full pitch visible. It is not designed for broadcast footage with frequent camera cuts, zoom changes, or tracking shots that follow the ball. The homography estimation exploits the fact that pitch geometry is constant across frames, and this assumption breaks under moving cameras.

Re-identification across full 90-minute matches — knowing with certainty that track ID 7 in the first half is the same player as track ID 7 in the second half — is not attempted. For shorter sequences, ByteTrack maintains identity reliably. For the target player specifically, re-initialisation by the user handles any identity switch that occurs.
