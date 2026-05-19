import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import config
import dataLoader
import opticalFlow
import movement
import detection
import tracking
from homography import PitchHomography


def buildHeatmap(worldPositions, pitchLengthM, pitchWidthM, resolution, sigma):
    gridW = int(pitchLengthM * resolution)
    gridH = int(pitchWidthM * resolution)
    heatmap = np.zeros((gridH, gridW), dtype=np.float32)

    for x, y in worldPositions:
        gx = min(max(int(x * resolution), 0), gridW - 1)
        gy = min(max(int(y * resolution), 0), gridH - 1)
        heatmap[gy, gx] += 1.0

    if(heatmap.sum() == 0):
        return heatmap

    heatmap = gaussian_filter(heatmap, sigma=sigma * resolution)
    heatmap = heatmap / heatmap.max()
    return heatmap


def zoneOccupancy(worldPositions, pitchLengthM, pitchWidthM):
    if(len(worldPositions) == 0):
        return {}

    thirdsL = [0, pitchLengthM / 3.0, 2 * pitchLengthM / 3.0, pitchLengthM]
    channelsW = [0, pitchWidthM * 0.20, pitchWidthM * 0.37,
                 pitchWidthM * 0.63, pitchWidthM * 0.80, pitchWidthM]
    thirdLabels = ["defensive", "middle", "final"]
    channelLabels = ["left-flank", "left-half-space", "centre", "right-half-space", "right-flank"]

    counts = {}
    for tl in thirdLabels:
        for cl in channelLabels:
            counts[f"{tl}/{cl}"] = 0

    for x, y in worldPositions:
        col = 0
        for i in range(3):
            if(x >= thirdsL[i] and x < thirdsL[i + 1]):
                col = i
                break
            else:
                if(x >= thirdsL[3]):
                    col = 2

        row = 0
        for i in range(5):
            if(y >= channelsW[i] and y < channelsW[i + 1]):
                row = i
                break
            else:
                if(y >= channelsW[5]):
                    row = 4

        label = f"{thirdLabels[col]}/{channelLabels[row]}"
        counts[label] += 1

    total = sum(counts.values())
    return {k: (v / total) * 100.0 for k, v in counts.items()}


def drawPitchTemplate(ax, pitchLengthM, pitchWidthM):
    ax.set_facecolor("#3a6b2f")
    ax.set_xlim(-2, pitchLengthM + 2)
    ax.set_ylim(-2, pitchWidthM + 2)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), pitchLengthM, pitchWidthM, color="white", fill=False, linewidth=1.8))
    ax.axvline(pitchLengthM / 2, color="white", linewidth=1.2)
    ax.add_patch(plt.Circle((pitchLengthM / 2, pitchWidthM / 2), 9.15, color="white", fill=False, linewidth=1.2))
    ax.add_patch(plt.Rectangle((0, (pitchWidthM - 40.3) / 2), 16.5, 40.3, color="white", fill=False, linewidth=1.2))
    ax.add_patch(plt.Rectangle((pitchLengthM - 16.5, (pitchWidthM - 40.3) / 2), 16.5, 40.3, color="white", fill=False, linewidth=1.2))
    ax.add_patch(plt.Rectangle((0, (pitchWidthM - 18.32) / 2), 5.5, 18.32, color="white", fill=False, linewidth=1.2))
    ax.add_patch(plt.Rectangle((pitchLengthM - 5.5, (pitchWidthM - 18.32) / 2), 5.5, 18.32, color="white", fill=False, linewidth=1.2))


def renderHeatmapImage(heatmap, pitchLengthM, pitchWidthM, outPath, title="player heatmap"):
    fig, ax = plt.subplots(figsize=(11, 7))
    drawPitchTemplate(ax, pitchLengthM, pitchWidthM)
    extent = [0, pitchLengthM, 0, pitchWidthM]
    ax.imshow(heatmap, extent=extent, origin="lower", cmap="jet",
              alpha=0.75, interpolation="bilinear")
    ax.set_xlabel("pitch length (m)")
    ax.set_ylabel("pitch width (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(str(outPath), dpi=120)
    plt.close()


def renderZoneBreakdown(zoneStats, pitchLengthM, pitchWidthM, outPath):
    if(len(zoneStats) == 0):
        return

    thirdsL = [0, pitchLengthM / 3.0, 2 * pitchLengthM / 3.0, pitchLengthM]
    channelsW = [0, pitchWidthM * 0.20, pitchWidthM * 0.37,
                 pitchWidthM * 0.63, pitchWidthM * 0.80, pitchWidthM]
    thirdLabels = ["defensive", "middle", "final"]
    channelLabels = ["left-flank", "left-half-space", "centre", "right-half-space", "right-flank"]

    fig, ax = plt.subplots(figsize=(13, 8))
    plt.subplots_adjust(left=0.18)
    maxPct = max(zoneStats.values()) if zoneStats else 1.0
    for col, tl in enumerate(thirdLabels):
        for row, cl in enumerate(channelLabels):
            label = f"{tl}/{cl}"
            pct = zoneStats.get(label, 0.0)
            x0, x1 = thirdsL[col], thirdsL[col + 1]
            y0, y1 = channelsW[row], channelsW[row + 1]
            shade = pct / maxPct if maxPct > 0 else 0.0
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#e74c3c", alpha=0.15 + 0.55 * shade, edgecolor="white", linewidth=0.6))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{pct:.1f}%", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    for x in thirdsL[1:-1]:
        ax.axvline(x, color="white", linewidth=0.6, linestyle="--", alpha=0.5)
    for y in channelsW[1:-1]:
        ax.axhline(y, color="white", linewidth=0.6, linestyle="--", alpha=0.5)

    for col, tl in enumerate(thirdLabels):
        midX = (thirdsL[col] + thirdsL[col + 1]) / 2
        ax.text(midX, -1.5, tl + " third", ha="center", fontsize=9, color="black", fontweight="bold")
    for row, cl in enumerate(channelLabels):
        midY = (channelsW[row] + channelsW[row + 1]) / 2
        ax.text(-3.5, midY, cl, ha="right", va="center", fontsize=8.5, color="black", fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("")
    drawPitchTemplate(ax, pitchLengthM, pitchWidthM)
    ax.set_title("zone occupancy breakdown  (5 channels x 3 thirds)")
    plt.tight_layout()
    plt.savefig(str(outPath), dpi=120)
    plt.close()


def videoFrameRate(videoPath):
    """reads the frame rate from a video file, falls back to the config default"""
    cap = cv2.VideoCapture(str(videoPath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if(fps <= 0):
        return float(config.frameRate)
    return float(fps)


def iterateFrames(inputPath, isVideo):
    """yields (frameId, frame) from either a sportsmot sequence folder or a video file"""
    if(isVideo):
        cap = cv2.VideoCapture(str(inputPath))
        frameId = 1
        while(True):
            ret, frame = cap.read()
            if(not ret):
                break
            if(frameId % 2  == 0):
                yield frameId, frame
            frameId += 1
        cap.release()
    else:
        for data in dataLoader.loadSequence(inputPath):
            yield data["frameId"], data["frame"]


def isOnPitch(bbox, cameraMotion, homography, pitchLengthM, pitchWidthM, marginM):
    """returns True if a detection projects to within the pitch bounds plus a margin"""
    footPx = movement.footPoint(bbox)
    compensated = movement.compensateCameraMotion(footPx, cameraMotion)
    worldX, worldY = movement.pixelToWorld(compensated, homography)
    if(worldX < -marginM or worldX > pitchLengthM + marginM):
        return False
    if(worldY < -marginM or worldY > pitchWidthM + marginM):
        return False
    return True


def saveAnnotatedFrame(frame, tracks, outPath):
    """draws track ids on a frame and saves it as a visual reference for picking a player"""
    annotated = frame.copy()
    for trackId, x, y, w, h in tracks:
        cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(annotated, f"id {trackId}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imwrite(str(outPath), annotated)


if(__name__ == "__main__"):
    # a player needs at least 2 seconds of visibility for a meaningful heatmap (50 frames at 25 fps)
    minVisibleFrames = 50
    # tolerance beyond the pitch line, absorbs homography error and keeps genuine touchline players
    pitchMarginM = 3.0

    if(len(sys.argv) < 2):
        print("usage: python heatmap.py <inputPath> [homographyPath]")
        print("  inputPath can be a sportsmot sequence folder or a staticCam video file")
        print("  homographyPath is required for video input, optional for sportsmot")
        sys.exit(1)

    inputPath = Path(sys.argv[1])
    homographyArg = Path(sys.argv[2]) if(len(sys.argv) >= 3) else None

    if(not inputPath.exists()):
        print(f"input path does not exist {inputPath}")
        sys.exit(1)

    isVideo = inputPath.is_file()

    if(isVideo):
        seqName = inputPath.stem
        frameRate = videoFrameRate(inputPath)
        useGtMode = False
        print(f"staticCam video input {inputPath.name}, {frameRate:.0f} fps")
        if(homographyArg is None):
            print("homography path is required for video input, pass it as the second argument")
            sys.exit(1)
        homoPath = homographyArg
    else:
        info = dataLoader.readSeqInfo(inputPath)
        seqName = info["name"]
        frameRate = info["frameRate"]
        useGtMode = config.useGt
        print(f"sportsmot sequence {seqName}, {info['seqLength']} frames at {frameRate} fps")
        if(homographyArg is not None):
            homoPath = homographyArg
        else:
            homoPath = config.projectRoot / "homographies" / f"{seqName}.npz"

    if(not homoPath.exists()):
        print(f"homography file not found at {homoPath}")
        sys.exit(1)
    homo = PitchHomography.load(str(homoPath))
    print(f"loaded homography from {homoPath.name}")

    model = None
    if(not useGtMode):
        print("loading yolo model for detection and tracking")
        model = detection.loadModel()

    gt = {}
    if(not isVideo and useGtMode):
        gt = dataLoader.readGroundTruth(inputPath)

    print("pass 1, collecting tracks and camera motion across all frames")
    allTracks = []
    cameraMotions = []
    prevFrame = None
    cumX, cumY = 0.0, 0.0
    firstFrameDone = False
    rawCount = 0
    keptCount = 0

    annotatedDir = config.outputsDir / "annotatedFrames"
    annotatedDir.mkdir(parents=True, exist_ok=True)

    for frameId, frame in iterateFrames(inputPath, isVideo):
        if(useGtMode):
            rawTracks = list(gt.get(frameId, []))
        else:
            yoloTracks = tracking.trackYolo(frame, model)
            rawTracks = [(t["trackId"], t["bbox"][0], t["bbox"][1], t["bbox"][2], t["bbox"][3]) for t in yoloTracks]

        if(isVideo):
            motion = (0.0, 0.0)
        else:
            if(prevFrame is None):
                motion = (0.0, 0.0)
            else:
                maskBoxes = [(x, y, w, h) for _, x, y, w, h in rawTracks]
                (dx, dy), _, _ = opticalFlow.computeCameraMotion(prevFrame, frame, maskBoxes)
                cumX += dx
                cumY += dy
                motion = (cumX, cumY)
            prevFrame = frame
        cameraMotions.append(motion)

        tracks = []
        for trackId, x, y, w, h in rawTracks:
            if(isOnPitch((x, y, w, h), motion, homo, config.pitchWidthM, config.pitchHeightM, pitchMarginM)):
                tracks.append((trackId, x, y, w, h))
        rawCount += len(rawTracks)
        keptCount += len(tracks)
        allTracks.append(tracks)
        if(frameId % 100 == 0):
            print(f"  processed frame {frameId}")

        if(not firstFrameDone):
            outFrame = annotatedDir / f"{seqName}_ids.jpg"
            saveAnnotatedFrame(frame, tracks, outFrame)
            print(f"saved annotated reference frame to {outFrame}")
            firstFrameDone = True

    print(f"processed {len(allTracks)} frames")
    print(f"kept {keptCount} on-pitch detections, dropped {rawCount - keptCount} off-pitch detections")

    idCounts = {}
    for tracks in allTracks:
        for trackId, x, y, w, h in tracks:
            idCounts[trackId] = idCounts.get(trackId, 0) + 1

    if(len(idCounts) == 0):
        print("no on-pitch players found in this clip, nothing to generate")
        sys.exit(1)

    print("available player ids and how many frames each is visible for")
    for trackId in sorted(idCounts.keys()):
        print(f"  id {trackId}  visible in {idCounts[trackId]} frames")

    chosen = int(input("enter the player id to generate a heatmap for: "))

    if(chosen not in idCounts):
        print(f"id {chosen} is not in this clip")
        sys.exit(1)

    if(idCounts[chosen] < minVisibleFrames):
        print(f"id {chosen} is only visible for {idCounts[chosen]} frames, needs at least {minVisibleFrames}")
        print("heatmap rejected, pick a player with more visibility")
        sys.exit(1)

    print(f"building heatmap for player {chosen}")
    targetPositions = []
    for tracks in allTracks:
        found = None
        for trackId, x, y, w, h in tracks:
            if(trackId == chosen):
                found = (x, y, w, h)
                break
        targetPositions.append(found)

    result = movement.analysePlayer(targetPositions, cameraMotions, homo, frameRate)
    worldPositions = result["worldPositions"]
    print(f"got {len(worldPositions)} world positions")

    heatmap = buildHeatmap(
        worldPositions,
        config.pitchWidthM,
        config.pitchHeightM,
        config.heatmapPitchResolution,
        config.heatmapGaussianSigma,
    )
    print(f"heatmap shape {heatmap.shape}, max intensity {heatmap.max():.3f}")

    zones = zoneOccupancy(worldPositions, config.pitchWidthM, config.pitchHeightM)
    print("zone occupancy")
    for label, pct in zones.items():
        if(pct > 0.0):
            print(f"  {label:35s}  {pct:5.1f}%")

    outImg = config.outputsDir / f"heatmap_{seqName}_player{chosen}.png"
    renderHeatmapImage(heatmap, config.pitchWidthM, config.pitchHeightM, outImg, title=f"player {chosen} heatmap  -  {seqName}")
    print(f"saved heatmap to {outImg}")

    outZones = config.outputsDir / f"zone_{seqName}_player{chosen}.png"
    renderZoneBreakdown(zones, config.pitchWidthM, config.pitchHeightM, outZones)
    print(f"saved zone breakdown to {outZones}")
    print("heatmap generation done")