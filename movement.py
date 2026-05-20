# feature 5
# feature 5
import argparse
from pathlib import Path
import numpy as np
import cv2
import config
import dataLoader
import opticalFlow
import detection
import tracking
from homography import PitchHomography
import matplotlib.pyplot as plt


def footPoint(bbox):
    """returns the foot position (bottom-centre) of a bounding box in pixels"""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h)


def pixelToWorld(footPx, homography):
    """converts a pixel foot position to real-world pitch coordinates in metres"""
    worldX, worldY = homography.pixel_to_world(footPx[0], footPx[1])
    return (worldX, worldY)


def compensateCameraMotion(footPx, cameraMotion):
    """subtracts cumulative camera drift from a pixel position"""
    dx, dy = cameraMotion
    return (footPx[0] - dx, footPx[1] - dy)


def computeSpeed(worldPositions, frameRate):
    """computes per-frame speed in km/h from real-world positions"""
    speeds = [0.0]
    for i in range(1, len(worldPositions)):
        x1, y1 = worldPositions[i - 1]
        x2, y2 = worldPositions[i]
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speedMps = dist * frameRate
        speedKmh = speedMps * 3.6
        speeds.append(speedKmh)
    return speeds


def smoothSpeeds(speeds, window):
    """applies a centered rolling mean to speed values"""
    smoothed = []
    half = window // 2
    for i in range(len(speeds)):
        lo = max(0, i - half)
        hi = min(len(speeds), i + half + 1)
        smoothed.append(float(np.mean(speeds[lo:hi])))
    return smoothed


def totalDistance(worldPositions):
    """sums frame-to-frame displacement in metres"""
    total = 0.0
    for i in range(1, len(worldPositions)):
        x1, y1 = worldPositions[i - 1]
        x2, y2 = worldPositions[i]
        total += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total


def detectSprints(speeds, threshold, minFrames):
    """returns list of (startFrame, endFrame, peakSpeed) for each sprint"""
    sprints = []
    inSprint = False
    startIdx = 0
    peak = 0.0
    for i, s in enumerate(speeds):
        if(s >= threshold):
            if(not inSprint):
                inSprint = True
                startIdx = i
                peak = s
            else:
                if(s > peak):
                    peak = s
        else:
            if(inSprint):
                length = i - startIdx
                if(length >= minFrames):
                    sprints.append((startIdx, i - 1, peak))
                inSprint = False
                peak = 0.0
    if(inSprint):
        length = len(speeds) - startIdx
        if(length >= minFrames):
            sprints.append((startIdx, len(speeds) - 1, peak))
    return sprints


def analysePlayer(targetPositionsPx, cameraMotions, homography, frameRate):
    """
    main entry point. takes:
      targetPositionsPx: list of (x, y, w, h) tuples per frame for the target player
                        use None for frames where the player is not detected
      cameraMotions:    list of (dx, dy) cumulative camera drift per frame
      homography:       a PitchHomography instance
      frameRate:        sequence frame rate
    returns a dict with speed profile, distance, and sprint events.
    """
    worldPositions = []
    validFrames = []
    for i, bbox in enumerate(targetPositionsPx):
        if(bbox is None):
            continue
        footPx = footPoint(bbox)
        compensatedPx = compensateCameraMotion(footPx, cameraMotions[i])
        worldPos = pixelToWorld(compensatedPx, homography)
        worldPositions.append(worldPos)
        validFrames.append(i)

    if(len(worldPositions) < 2):
        return {
            "worldPositions": worldPositions,
            "validFrames": validFrames,
            "speeds": [],
            "smoothedSpeeds": [],
            "totalDistanceM": 0.0,
            "peakSpeedKmh": 0.0,
            "avgSpeedKmh": 0.0,
            "sprints": [],
        }

    rawSpeeds = computeSpeed(worldPositions, frameRate)
    smoothed = smoothSpeeds(rawSpeeds, config.speedSmoothingWindow)
    distM = totalDistance(worldPositions)
    sprints = detectSprints(smoothed, config.sprintSpeedThresholdKmh, config.sprintMinFrames)

    return {
        "worldPositions": worldPositions,
        "validFrames": validFrames,
        "speeds": rawSpeeds,
        "smoothedSpeeds": smoothed,
        "totalDistanceM": distM,
        "peakSpeedKmh": float(np.max(smoothed)),
        "avgSpeedKmh": float(np.mean(smoothed)),
        "sprints": sprints,
    }


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="feature 5 player movement analysis")
    parser.add_argument("--input", type=str, default=None,
                        help="optional path to a sportsmot sequence folder or a video file. defaults to config.testSequencePath")
    parser.add_argument("--homo", type=str, default=None,
                        help="optional path to a homography .npz file. required when input is a video, defaults to homographies/<seqName>.npz for sportsmot")
    args = parser.parse_args()

    if(args.input is None):
        inputPath = config.testSequencePath
    else:
        inputPath = Path(args.input)

    if(not inputPath.exists()):
        print(f"input path does not exist {inputPath}")
        exit(1)

    isVideo = inputPath.is_file()

    # resolve frame rate, sequence name, and homography
    if(isVideo):
        seqName = inputPath.stem
        frameRate = dataLoader.videoFrameRate(inputPath)
        print(f"running movement on video {inputPath.name}, {frameRate:.0f} fps")
        if(args.homo is None):
            print("homography path is required for video input. pass --homo <path>")
            exit(1)
        homoPath = Path(args.homo)
    else:
        info = dataLoader.readSeqInfo(inputPath)
        seqName = info["name"]
        frameRate = info["frameRate"]
        print(f"running movement on sportsmot sequence {seqName}, {info['seqLength']} frames at {frameRate} fps")
        if(args.homo is None):
            homoPath = config.projectRoot / "homographies" / f"{seqName}.npz"
        else:
            homoPath = Path(args.homo)

    if(not homoPath.exists()):
        print(f"homography file not found at {homoPath}")
        print("run compute_homographies.py first for sportsmot, or pass an existing .npz with --homo")
        exit(1)
    homo = PitchHomography.load(str(homoPath))
    print(f"loaded homography from {homoPath.name}")

    numFrames = 200

    # collect frames, target positions, and camera motions
    # path branches because video has no gt and needs yolo + bytetrack to find a player to follow
    if(isVideo):
        print("loading yolo model for video tracking")
        model = detection.loadModel()

        print(f"running yolo + bytetrack across up to {numFrames} frames")
        allTracks = []
        idCounts = {}
        framesProcessed = 0
        for data in dataLoader.loadVideo(inputPath):
            if(framesProcessed >= numFrames):
                break
            frame = data["frame"]
            yoloTracks = tracking.trackYolo(frame, model)
            frameTracks = [(t["trackId"], t["bbox"][0], t["bbox"][1], t["bbox"][2], t["bbox"][3]) for t in yoloTracks]
            allTracks.append(frameTracks)
            for trackId, *_ in frameTracks:
                idCounts[trackId] = idCounts.get(trackId, 0) + 1
            framesProcessed += 1
            if(framesProcessed % 50 == 0):
                print(f"  processed frame {framesProcessed}")

        print(f"processed {framesProcessed} frames, found {len(idCounts)} unique track ids")

        if(len(idCounts) == 0):
            print("no tracks found in video, cannot pick target player")
            exit(1)

        # auto-pick the longest-lived track (option A)
        targetId = max(idCounts, key=idCounts.get)
        print(f"auto-picked track id {targetId} as target (visible in {idCounts[targetId]} of {framesProcessed} frames)")

        targetPositions = []
        for frameTracks in allTracks:
            found = None
            for trackId, x, y, w, h in frameTracks:
                if(trackId == targetId):
                    found = (x, y, w, h)
                    break
            targetPositions.append(found)

        # static cam assumption for video, mirrors what heatmap does
        cameraMotions = [(0.0, 0.0)] * len(allTracks)
        numFrames = len(allTracks)

    else:
        gt = dataLoader.readGroundTruth(inputPath)
        if(1 not in gt or len(gt[1]) == 0):
            print("no gt boxes in frame 1, cannot pick target player")
            exit(1)
        targetId = gt[1][0][0]
        print(f"target player set to gt track id {targetId}")

        numFrames = min(numFrames, info["seqLength"])

        print(f"loading {numFrames} frames for analysis")
        frames = []
        targetPositions = []
        for i, data in enumerate(dataLoader.loadSequence(inputPath)):
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

        print("computing camera motion across analysis window")
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
        print(f"total camera drift ({cumX:.1f}, {cumY:.1f}) px")

    # common analysis path
    print("running player analysis")
    result = analysePlayer(targetPositions, cameraMotions, homo, frameRate)

    print(f"  total distance: {result['totalDistanceM']:.1f} m")
    print(f"  average speed:  {result['avgSpeedKmh']:.2f} km/h")
    print(f"  peak speed:     {result['peakSpeedKmh']:.2f} km/h")
    print(f"  sprints detected: {len(result['sprints'])}")
    for i, (s, e, peak) in enumerate(result["sprints"]):
        durationSec = (e - s + 1) / frameRate
        print(f"    sprint {i + 1}: frames {s}-{e} ({durationSec:.1f}s), peak {peak:.1f} km/h")

    # trajectory and speed profile plot
    pitchL = 105.0
    pitchW = 68.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.set_facecolor("#4a7c3f")
    ax1.set_xlim(0, pitchL)
    ax1.set_ylim(0, pitchW)
    ax1.axvline(pitchL / 2, color="white", linewidth=1)
    ax1.add_patch(plt.Circle((pitchL / 2, pitchW / 2), 9.15, color="white", fill=False, linewidth=1))
    wxs = [p[0] for p in result["worldPositions"]]
    wys = [p[1] for p in result["worldPositions"]]
    if(len(wxs) > 0):
        ax1.plot(wxs, wys, color="yellow", linewidth=1.2, alpha=0.7)
        ax1.scatter(wxs[0], wys[0], color="lime", s=60, zorder=5, label="start")
        ax1.scatter(wxs[-1], wys[-1], color="red", s=60, zorder=5, label="end")
    ax1.set_xlabel("pitch length (m)")
    ax1.set_ylabel("pitch width (m)")
    ax1.set_title(f"player {targetId} trajectory  -  {result['totalDistanceM']:.1f}m covered")
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(result["validFrames"], result["smoothedSpeeds"], color="#2980b9", linewidth=1.5, label="smoothed speed")
    ax2.axhline(config.sprintSpeedThresholdKmh, color="red", linestyle="--", linewidth=1, label=f"sprint threshold ({config.sprintSpeedThresholdKmh} km/h)")
    for s, e, peak in result["sprints"]:
        ax2.axvspan(result["validFrames"][s], result["validFrames"][e], alpha=0.2, color="red")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("speed (km/h)")
    ax2.set_title(f"speed profile  -  peak {result['peakSpeedKmh']:.1f} km/h")
    ax2.legend(fontsize=8)
    plt.tight_layout()

    outPath = config.outputsDir / "movement_validation.png"
    plt.savefig(str(outPath), dpi=120)
    plt.close()
    print(f"saved validation plot to {outPath}")
    print("movement analysis done")