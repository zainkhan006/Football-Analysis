import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import config
import dataLoader
import opticalFlow
import movement
from homography import PitchHomography


def buildHeatmap(worldPositions, pitchLengthM, pitchWidthM, resolution, sigma):
    gridW = int(pitchLengthM * resolution)
    gridH = int(pitchWidthM * resolution)
    heatmap = np.zeros((gridH, gridW), dtype=np.float32)

    for x, y in worldPositions:
        gx = int(x * resolution)
        gy = int(y * resolution)
        if(gx < 0 or gx >= gridW):
            continue
        if(gy < 0 or gy >= gridH):
            continue
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


if(__name__ == "__main__"):
    print("running heatmap self-check on test sequence")
    seqPath = config.testSequencePath
    info = dataLoader.readSeqInfo(seqPath)
    gt = dataLoader.readGroundTruth(seqPath)
    frameRate = info["frameRate"]
    print(f"sequence {info['name']}, {info['seqLength']} frames at {frameRate} fps")
    homoPath = config.projectRoot / "homographies" / f"{info['name']}.npz"
    if(not homoPath.exists()):
        print(f"  homography file not found at {homoPath}")
        exit(1)

    homo = PitchHomography.load(str(homoPath))
    print(f"  loaded homography from {homoPath.name}")
    if(1 not in gt or len(gt[1]) == 0):
        print("  no gt boxes in frame 1")
        exit(1)
    targetId = gt[1][0][0]
    print(f"  target player set to track id {targetId}")
    numFrames = min(500, info["seqLength"])
    print(f"loading {numFrames} frames for heatmap analysis")
    frames = []
    targetPositions = []
    for i, data in enumerate(dataLoader.loadSequence(seqPath)):
        if(i >= numFrames):
            break
        frames.append(data["frame"])
        found = None
        for trackId, x, y, w, h in data["boxes"]:
            if(trackId == targetId):
                found = (x, y, w, h)
                break
        targetPositions.append(found)

    framesPresent = sum(1 for p in targetPositions if p is not None)
    print(f"target player visible in {framesPresent}/{numFrames} frames")
    print("computing camera motion")
    cameraMotions = [(0.0, 0.0)]
    cumX, cumY = 0.0, 0.0
    for i in range(1, len(frames)):
        prevBoxes = []
        if(i in gt):
            prevBoxes = [(x, y, w, h) for _, x, y, w, h in gt[i]]
        (dx, dy), _, _ = opticalFlow.computeCameraMotion(frames[i - 1], frames[i], prevBoxes)
        cumX += dx
        cumY += dy
        cameraMotions.append((cumX, cumY))

    print("getting world positions via movement.analysePlayer")
    result = movement.analysePlayer(targetPositions, cameraMotions, homo, frameRate)
    worldPositions = result["worldPositions"]
    print(f"  got {len(worldPositions)} world positions")
    print("building heatmap")
    heatmap = buildHeatmap(
        worldPositions,
        config.pitchWidthM,
        config.pitchHeightM,
        config.heatmapPitchResolution,
        config.heatmapGaussianSigma,
    )
    print(f"  heatmap shape {heatmap.shape}, max intensity {heatmap.max():.3f}")
    print("computing zone occupancy")
    zones = zoneOccupancy(worldPositions, config.pitchWidthM, config.pitchHeightM)
    for label, pct in zones.items():
        if(pct > 0.0):
            print(f"  {label:35s}  {pct:5.1f}%")

    outImg = config.outputsDir / "heatmap_test.png"
    renderHeatmapImage(heatmap, config.pitchWidthM, config.pitchHeightM, outImg, title=f"player {targetId} heatmap  -  {numFrames} frames")
    print(f"saved heatmap to {outImg}")
    outZones = config.outputsDir / "zone_occupancy_test.png"
    renderZoneBreakdown(zones, config.pitchWidthM, config.pitchHeightM, outZones)
    print(f"saved zone breakdown to {outZones}")
    print("heatmap self-check done")