"""
Feature 10 - Formation Detection and Visualisation
====================================================
Detects each team's formation from player world-coordinates and renders an
interactive playback window with:
    - the main video frame, with horizontal lines drawn across each
      defensive line of players (one team at a time, both shown together)
    - two minimaps on the right side, one per team, showing the average
      player positions on a 2D pitch with the formation string above

Pipeline per frame:
    1. take tracked player boxes from tracking.py
    2. assign each track to a team using feature_8's TeamAssigner
    3. drop the goalkeeper (1 per team) since GKs distort line clustering
    4. project foot points to world coordinates using Lamaan's homography
    5. cluster each team's y-coordinates into defensive lines
    6. count players per line, smooth the count over a rolling window,
       map to a formation string

Run:
    python feature_10.py staticCam/<video name>.mp4 homographies/napoli_roma_2.npz
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import deque
import config
from tracking import trackYolo
from feature_8 import TeamAssigner
from homography import PitchHomography
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════
# Formation detection knobs
# ══════════════════════════════════════════════════════════════════════

# Rolling window of frames over which the line counts are averaged.
# 1s of footage at 25fps. Smaller = jumpy formations, larger = sluggish.
formationSmoothingWindow = 50
pitchMarginM = 3.0

# K-means line counts to try. A team plays in 2-4 horizontal lines.
# We pick the k whose within-cluster variance "elbow" gives the best fit.
minLineCount = 2
maxLineCount = 4

# Known orthodox formations. If the detected line counts don't match any
# of these, we report "unknown" rather than print a nonsense string.
knownFormations = {
    (4, 4, 2): "4-4-2",
    (4, 3, 3): "4-3-3",
    (4, 2, 3, 1): "4-2-3-1",
    (4, 5, 1): "4-5-1",
    (4, 1, 4, 1): "4-1-4-1",
    (3, 5, 2): "3-5-2",
    (3, 4, 3): "3-4-3",
    (5, 3, 2): "5-3-2",
    (5, 4, 1): "5-4-1",
    (4, 4, 1, 1): "4-4-1-1",
}

# Visual constants
homeColor    = (60,  200, 60)    # green tint
awayColor    = (60,  120, 220)   # blue tint
lineThickness = 2
minimapWidth  = 280
minimapHeight = 180
panelBgColor  = (28, 28, 28)
panelTextColor = (240, 240, 240)


# ══════════════════════════════════════════════════════════════════════
# Line clustering
# ══════════════════════════════════════════════════════════════════════

def clusterLinesByY(worldYs):
    """
    Cluster the team's y-coordinates into horizontal defensive lines.

    Tries every k in [minLineCount, maxLineCount] and picks the k that
    minimises the within-cluster variance relative to the variance you'd
    expect from one more line — a simple elbow test.

    Returns a list of player counts per line, sorted from defensive end
    (low y) to attacking end (high y). E.g. [4, 3, 3] for a 4-3-3.
    """
    n = len(worldYs)
    if(n < minLineCount):
        return []

    samples = np.array(worldYs, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)

    bestK = minLineCount
    bestVariance = float("inf")
    bestLabels = None
    bestCentres = None

    for k in range(minLineCount, min(maxLineCount, n) + 1):
        compactness, labels, centres = cv2.kmeans(
            samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        # Elbow rule: divide compactness by k. The first k where this drops
        # sharply is the right line count.
        score = compactness / k
        if(score < bestVariance):
            bestVariance = score
            bestK = k
            bestLabels = labels.flatten()
            bestCentres = centres.flatten()

    if(bestLabels is None):
        return []

    # Sort clusters from defensive (low y, near own goal) to attacking
    centreOrder = np.argsort(bestCentres)
    counts = []
    for clusterIdx in centreOrder:
        members = int(np.sum(bestLabels == clusterIdx))
        if(members > 0):
            counts.append(members)
    return counts


def countsToFormationString(counts):
    """Map a tuple of line counts to a known formation name."""
    if(not counts):
        return "unknown"
    key = tuple(counts)
    if(key in knownFormations):
        return knownFormations[key]
    return "unknown"

class TrackPositionAverager:
    """
    Maintains a rolling average of each track's world-y coordinate over a
    fixed window. Clustering on averaged positions instead of per-frame
    positions stops lines from jumping around as players take a step.
    """
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.history = {}   # trackId -> deque of (worldX, worldY)

    def update(self, trackIdsAndWorldXY):
        """trackIdsAndWorldXY: list of (trackId, worldX, worldY)"""
        seenIds = set()
        for tid, wx, wy in trackIdsAndWorldXY:
            seenIds.add(tid)
            if(tid not in self.history):
                self.history[tid] = deque(maxlen=self.windowSize)
            self.history[tid].append((wx, wy))
        # Drop tracks not seen this frame for a while - prevents stale data
        # building up forever as track IDs change
        toRemove = []
        for tid in self.history:
            if(tid not in seenIds):
                if(len(self.history[tid]) > 0):
                    self.history[tid].popleft()
                if(len(self.history[tid]) == 0):
                    toRemove.append(tid)
        for tid in toRemove:
            del self.history[tid]

    def getAveraged(self, trackIds):
        """Return list of (trackId, avgWorldX, avgWorldY) for given ids."""
        out = []
        for tid in trackIds:
            if(tid in self.history and len(self.history[tid]) > 0):
                xs = [p[0] for p in self.history[tid]]
                ys = [p[1] for p in self.history[tid]]
                out.append((tid, float(np.mean(xs)), float(np.mean(ys))))
        return out

class FormationSmoother:
    """
    Rolling-window stabiliser. Per-frame line counts are noisy because a
    single player making a run shifts a cluster. We buffer the last N
    counts and report the most frequent one.
    """
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.buffer = deque(maxlen=windowSize)

    def add(self, counts):
        if(counts):
            self.buffer.append(tuple(counts))

    def get(self):
        if(not self.buffer):
            return None
        # Most common pattern in the window
        votes = {}
        for c in self.buffer:
            votes[c] = votes.get(c, 0) + 1
        bestPattern = max(votes, key=votes.get)
        return list(bestPattern)


# ══════════════════════════════════════════════════════════════════════
# Goalkeeper exclusion
# ══════════════════════════════════════════════════════════════════════

def removeGoalkeeper(playerData):
    """
    Drop the most extreme player on the y-axis (closest to either goal).
    Each team has exactly one GK and they sit far behind everyone else,
    so the deepest player is almost always the GK.

    playerData: list of (trackId, worldX, worldY)
    Returns the same list minus the deepest player.
    """
    if(len(playerData) < 3):
        return playerData

    ys = [p[2] for p in playerData]
    yMin = min(ys)
    yMax = max(ys)
    # We don't know which goal each team defends, so check both extremes.
    # The GK is the player who is the biggest outlier from the rest.
    sortedY = sorted(ys)
    lowGap = sortedY[1] - sortedY[0]
    highGap = sortedY[-1] - sortedY[-2]

    if(lowGap > highGap):
        return [p for p in playerData if(p[2] != yMin)]
    else:
        return [p for p in playerData if(p[2] != yMax)]


# ══════════════════════════════════════════════════════════════════════
# Pixel-space line drawing on the video frame
# ══════════════════════════════════════════════════════════════════════

def drawLinesOnFrame(frame, teamPlayers, worldYs, lineCounts, color):
    """
    Draw horizontal lines on the video frame connecting players who are in
    the same defensive line.

    teamPlayers: list of (trackId, foot_px_x, foot_px_y) in image space
    worldYs: parallel list of world-y coordinates (depth on pitch)
    lineCounts: e.g. [4, 3, 3] meaning 4 in line 1, 3 in line 2, 3 in line 3
    """
    if(not lineCounts or len(teamPlayers) != len(worldYs)):
        return

    # Re-cluster using the lineCounts so we know which player belongs to
    # which line. We already clustered once, but re-doing it here keeps
    # the visualisation self-contained and avoids passing labels around.
    samples = np.array(worldYs, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    k = len(lineCounts)
    if(k < 2 or k > len(samples)):
        return

    _, labels, centres = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    centreOrder = np.argsort(centres.flatten())

    # For each line, collect the pixel positions of its members and connect
    # them with a polyline sorted left-to-right
    for clusterIdx in centreOrder:
        members = [teamPlayers[i] for i in range(len(teamPlayers))
                   if(labels[i] == clusterIdx)]
        if(len(members) < 2):
            continue
        members.sort(key=lambda p: p[1])  # sort by image x
        pts = [(int(p[1]), int(p[2])) for p in members]
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, lineThickness, cv2.LINE_AA)
        # Small filled circles at each player position
        for px, py in pts:
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════
# Minimap rendering
# ══════════════════════════════════════════════════════════════════════

def drawPitchTemplate(canvas):
    """Sketch a basic pitch outline on the minimap canvas."""
    h, w = canvas.shape[:2]
    pitchColor = (40, 90, 40)
    lineColor  = (200, 200, 200)
    canvas[:] = pitchColor
    cv2.rectangle(canvas, (4, 4), (w - 4, h - 4), lineColor, 1)
    cv2.line(canvas, (w // 2, 4), (w // 2, h - 4), lineColor, 1)
    cv2.circle(canvas, (w // 2, h // 2), min(w, h) // 8, lineColor, 1)


def renderMinimap(teamWorldPositions, lineCounts, color, formationStr, teamName):
    """
    Render a 2D bird's-eye minimap for one team showing average player
    positions as dots with horizontal lines connecting same-line players.

    teamWorldPositions: list of (worldX, worldY)
    """
    canvas = np.zeros((minimapHeight, minimapWidth, 3), dtype=np.uint8)
    drawPitchTemplate(canvas)

    # Header strip with team name and formation string
    headerH = 24
    header = np.zeros((headerH, minimapWidth, 3), dtype=np.uint8)
    header[:] = panelBgColor
    cv2.putText(header, f"{teamName}  {formationStr}",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                panelTextColor, 1, cv2.LINE_AA)

    if(not teamWorldPositions):
        return np.vstack([header, canvas])

    # Map world coords to minimap pixels. World x range: 0..pitchWidthM,
    # World y range: 0..pitchHeightM (these are set in config.py).
    xs = [p[0] for p in teamWorldPositions]
    ys = [p[1] for p in teamWorldPositions]
    drawMargin = 8
    drawableW = minimapWidth - 2 * drawMargin
    drawableH = minimapHeight - 2 * drawMargin

    def worldToMini(wx, wy):
        nx = wx / config.pitchWidthM
        ny = wy / config.pitchHeightM
        mx = int(drawMargin + nx * drawableW)
        my = int(drawMargin + ny * drawableH)
        return (mx, my)

    # Cluster by y for the line connection lines
    if(lineCounts and len(lineCounts) >= 2 and len(ys) >= 2):
        samples = np.array(ys, dtype=np.float32).reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        k = len(lineCounts)
        if(k <= len(samples)):
            _, labels, centres = cv2.kmeans(
                samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            labels = labels.flatten()
            centreOrder = np.argsort(centres.flatten())
            for clusterIdx in centreOrder:
                members = [(xs[i], ys[i]) for i in range(len(xs))
                           if(labels[i] == clusterIdx)]
                if(len(members) < 2):
                    continue
                members.sort(key=lambda p: p[0])
                pts = [worldToMini(mx, my) for mx, my in members]
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

    # Player dots
    for wx, wy in teamWorldPositions:
        px, py = worldToMini(wx, wy)
        cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)

    return np.vstack([header, canvas])


# ══════════════════════════════════════════════════════════════════════
# Composite frame
# ══════════════════════════════════════════════════════════════════════

def composeFrame(videoFrame, homeMini, awayMini):
    """Stack the video frame next to the two stacked minimaps."""
    vh, vw = videoFrame.shape[:2]
    panel = np.vstack([homeMini, awayMini])
    ph, pw = panel.shape[:2]
    # Pad whichever side is shorter so heights match
    if(ph < vh):
        pad = np.zeros((vh - ph, pw, 3), dtype=np.uint8)
        pad[:] = panelBgColor
        panel = np.vstack([panel, pad])
    elif(ph > vh):
        pad = np.zeros((ph - vh, vw, 3), dtype=np.uint8)
        videoFrame = np.vstack([videoFrame, pad])
    return np.hstack([videoFrame, panel])


# ══════════════════════════════════════════════════════════════════════
# Sequence runner with interactive playback
# ══════════════════════════════════════════════════════════════════════

def runFormationViewer(videoPath, homographyPath):
    """
    Walk through the staticCam video frame-by-frame and render the
    formation view.

    Controls:
        SPACE   pause/play
        N       next frame (when paused)
        P       previous frame (when paused)
        Q       quit
    """
    cap = cv2.VideoCapture(str(videoPath))
    if(not cap.isOpened()):
        print(f"could not open video {videoPath}")
        raise SystemExit(1)
    seqLen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"loaded video {Path(videoPath).name} with {seqLen} frames")

    homography = PitchHomography.load(homographyPath)
    print(f"loaded homography from {homographyPath}")

    assigner = TeamAssigner()
    yoloModel = YOLO(config.yoloWeightsPath)

    # Pre-compute everything because seeking back/forward needs random access
    cachedDisplay = []

    homeSmoother = FormationSmoother(formationSmoothingWindow)
    awaySmoother = FormationSmoother(formationSmoothingWindow)
    homeAverager = TrackPositionAverager(formationSmoothingWindow)
    awayAverager = TrackPositionAverager(formationSmoothingWindow)

    print("processing frames - this can take a minute on a long clip")

    frameIdx = 0
    while(True):
        ret, frame = cap.read()
        if(not ret):
            break
        if(frameIdx > 3000):
            break

        # Get tracks for this frame
        tracks = trackYolo(frame, yoloModel)

        # Assign teams
        labels = assigner.assign(frame, tracks)
        print(f"frame {frameIdx}  total tracks {len(tracks)}  home {sum(1 for v in labels.values() if v == 'home')}  away {sum(1 for v in labels.values() if v == 'away')}  gk {sum(1 for v in labels.values() if v == 'gk')}")

        # Build (trackId, footPxX, footPxY, worldX, worldY, team) per track
        homePlayersPx = []
        awayPlayersPx = []
        homeWorld = []
        awayWorld = []

        for t in tracks:
            tid = int(t["trackId"])
            x, y, w, h = t["bbox"]
            footPxX = x + w / 2.0
            footPxY = y + h
            wx, wy = homography.pixel_to_world(footPxX, footPxY)
            if(wx < -pitchMarginM or wx > config.pitchWidthM + pitchMarginM):
                continue
            if(wy < -pitchMarginM or wy > config.pitchHeightM + pitchMarginM):
                continue
            team = labels.get(tid, "gk")
            if(team == "home"):
                homePlayersPx.append((tid, footPxX, footPxY))
                homeWorld.append((tid, wx, wy))
            elif(team == "away"):
                awayPlayersPx.append((tid, footPxX, footPxY))
                awayWorld.append((tid, wx, wy))

        # Drop the GK from each team
        homeWorld = removeGoalkeeper(homeWorld)
        awayWorld = removeGoalkeeper(awayWorld)
        print(f"frame {frameIdx}  on-pitch home {len(homeWorld)}  away {len(awayWorld)}")
        # Keep the pixel and world lists in sync after GK removal
        homeKeepIds = {p[0] for p in homeWorld}
        awayKeepIds = {p[0] for p in awayWorld}
        homePlayersPx = [p for p in homePlayersPx if(p[0] in homeKeepIds)]
        awayPlayersPx = [p for p in awayPlayersPx if(p[0] in awayKeepIds)]

        # Feed current-frame positions into the running averager
        homeAverager.update(homeWorld)
        awayAverager.update(awayWorld)

        # Cluster on AVERAGED positions, not per-frame positions
        minPlayersForFormation = 6
        homeAvg = homeAverager.getAveraged([p[0] for p in homeWorld])
        awayAvg = awayAverager.getAveraged([p[0] for p in awayWorld])
        homeAvgYs = [p[2] for p in homeAvg]
        awayAvgYs = [p[2] for p in awayAvg]

        # Hard gate on CURRENT-frame visibility, not smoother state
        homeHasEnough = len(homeWorld) >= minPlayersForFormation
        awayHasEnough = len(awayWorld) >= minPlayersForFormation

        homeCounts = clusterLinesByY(homeAvgYs) if homeHasEnough else []
        awayCounts = clusterLinesByY(awayAvgYs) if awayHasEnough else []
        homeSmoother.add(homeCounts)
        awaySmoother.add(awayCounts)
        homeSmoothed = homeSmoother.get() if homeHasEnough else None
        awaySmoothed = awaySmoother.get() if awayHasEnough else None
        homeFormation = countsToFormationString(homeSmoothed) if homeSmoothed else "--"
        awayFormation = countsToFormationString(awaySmoothed) if awaySmoothed else "--"

        # Build avgY lookup so drawLinesOnFrame can use the averaged y per track
        homeAvgYById = {p[0]: p[2] for p in homeAvg}
        awayAvgYById = {p[0]: p[2] for p in awayAvg}
        homeAvgYsForPx = [homeAvgYById.get(p[0], 0.0) for p in homePlayersPx]
        awayAvgYsForPx = [awayAvgYById.get(p[0], 0.0) for p in awayPlayersPx]

        # Draw lines on the video frame using averaged y for stable clustering
        annotated = frame.copy()
        if(homeSmoothed):
            drawLinesOnFrame(annotated, homePlayersPx, homeAvgYsForPx, homeSmoothed, homeColor)
        if(awaySmoothed):
            drawLinesOnFrame(annotated, awayPlayersPx, awayAvgYsForPx, awaySmoothed, awayColor)

        # Render minimaps using AVERAGED positions so dots don't jitter
        homeWorldXY = [(p[1], p[2]) for p in homeAvg]
        awayWorldXY = [(p[1], p[2]) for p in awayAvg]
        homeMini = renderMinimap(homeWorldXY, homeSmoothed, homeColor,
                                 homeFormation, "HOME")
        awayMini = renderMinimap(awayWorldXY, awaySmoothed, awayColor,
                                 awayFormation, "AWAY")

        cachedDisplay.append(composeFrame(annotated, homeMini, awayMini))

        frameIdx += 1
        if(frameIdx % 50 == 0):
            print(f"  processed frame {frameIdx}/{seqLen}")

    cap.release()

    print("done processing - opening viewer")
    print("controls: SPACE play/pause | N next | P prev | Q quit")

    # Interactive playback loop
    windowName = "Feature 10 - Formation Viewer"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    idx = 0
    playing = True
    while(True):
        if(cachedDisplay[idx] is None):
            idx = (idx + 1) % len(cachedDisplay)
            continue
        cv2.imshow(windowName, cachedDisplay[idx])
        key = cv2.waitKey(40 if playing else 0) & 0xFF
        if(key == ord("q")):
            break
        elif(key == ord(" ")):
            playing = not playing
        elif(key == ord("n")):
            playing = False
            idx = min(idx + 1, len(cachedDisplay) - 1)
        elif(key == ord("p")):
            playing = False
            idx = max(idx - 1, 0)
        else:
            if(playing):
                idx = (idx + 1) % len(cachedDisplay)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if(__name__ == "__main__"):
    if(len(sys.argv) < 3):
        print("usage: python feature_10.py <videoPath> <homographyPath>")
        raise SystemExit(1)

    videoPath = Path(sys.argv[1])
    homographyPath = Path(sys.argv[2])

    if(not videoPath.is_file()):
        print(f"video file not found at {videoPath}")
        raise SystemExit(1)
    if(not homographyPath.is_file()):
        print(f"homography file not found at {homographyPath}")
        raise SystemExit(1)

    runFormationViewer(videoPath, homographyPath)