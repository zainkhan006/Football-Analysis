# feature 4

# feature 4

import argparse
from pathlib import Path
import cv2
import numpy as np
import config
import dataLoader


def computeCameraMotion(prevFrame, currFrame, bboxes):
    grayPrev = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    grayCurr = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
    mask = np.ones(grayPrev.shape, dtype=np.uint8) * 255
    for x, y, w, h in bboxes:
        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 0, -1)

    prevPts = cv2.goodFeaturesToTrack(
        grayPrev, mask=mask,
        maxCorners=200, qualityLevel=0.01,
        minDistance=10, blockSize=7,
    )

    if prevPts is None or len(prevPts) < 10:
        return (0.0, 0.0), np.empty((0, 2)), np.empty((0, 2))

    nextPts, status, err = cv2.calcOpticalFlowPyrLK(
        grayPrev, grayCurr, prevPts, None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    good = status.flatten() == 1
    goodPrev = prevPts[good].reshape(-1, 2)
    goodCurr = nextPts[good].reshape(-1, 2)
    if len(goodPrev) < 5:
        return (0.0, 0.0), goodPrev, goodCurr
    displacements = goodCurr - goodPrev
    dx = float(np.median(displacements[:, 0]))
    dy = float(np.median(displacements[:, 1]))

    return (dx, dy), goodPrev, goodCurr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="feature 4 optical flow")
    parser.add_argument("--input", type=str, default=None,
                        help="optional path to a sportsmot sequence folder or a video file. defaults to config.testSequencePath")
    args = parser.parse_args()

    if(args.input is None):
        inputPath = config.testSequencePath
    else:
        inputPath = Path(args.input)

    if(not inputPath.exists()):
        print(f"input path does not exist {inputPath}")
        exit(1)

    isVideo = inputPath.is_file()

    numFrames = 60

    if(isVideo):
        print(f"computing optical flow across first {numFrames} frames of video {inputPath.name}")
        print("video input has no gt so player masking is skipped, flow computed on full frame")
        frameGen = dataLoader.loadVideo(inputPath)
    else:
        print(f"computing camera motion across first {numFrames} frames of sportsmot sequence {inputPath.name}")
        frameGen = dataLoader.loadSequence(inputPath)

    prevFrame = None
    prevBoxes = []
    motions = []
    midFrame = None
    midPrevPts = None
    midCurrPts = None
    midBoxes = []
    midMotion = None

    for i, data in enumerate(frameGen):
        if(i >= numFrames):
            break

        currFrame = data["frame"]
        # for sportsmot, use gt boxes as player mask. for video, boxes is empty so mask is the full frame
        currBoxes = [(b[1], b[2], b[3], b[4]) for b in data["boxes"]]

        if(prevFrame is not None):
            (dx, dy), prevPts, currPts = computeCameraMotion(prevFrame, currFrame, prevBoxes)
            motions.append((dx, dy))

            if(i == numFrames // 2):
                midFrame = currFrame.copy()
                midPrevPts = prevPts
                midCurrPts = currPts
                midBoxes = currBoxes
                midMotion = (dx, dy)

        prevFrame = currFrame
        prevBoxes = currBoxes

    motionsArr = np.array(motions)
    magnitudes = np.linalg.norm(motionsArr, axis=1)
    print(f"computed {len(motions)} motion vectors")
    print(f"  mean magnitude {magnitudes.mean():.2f} px per frame")
    print(f"  max magnitude {magnitudes.max():.2f} px")
    print(f"  total camera drift x {motionsArr[:, 0].sum():.1f}, y {motionsArr[:, 1].sum():.1f} px")

    if(midFrame is not None):
        for x, y, w, h in midBoxes:
            cv2.rectangle(midFrame, (int(x), int(y)), (int(x + w), int(y + h)), (120, 120, 120), 1)

        for prev, curr in zip(midPrevPts, midCurrPts):
            aX, aY = int(prev[0]), int(prev[1])
            bX, bY = int(curr[0]), int(curr[1])
            cv2.arrowedLine(midFrame, (aX, aY), (bX, bY), (0, 255, 255), 1, tipLength=0.3)
            cv2.circle(midFrame, (aX, aY), 2, (0, 0, 255), -1)

        frameH, frameW = midFrame.shape[:2]
        centreX, centreY = frameW // 2, frameH // 2
        scale = 20
        dx, dy = midMotion
        cv2.arrowedLine(midFrame, (centreX, centreY),
                        (centreX + int(dx * scale), centreY + int(dy * scale)),
                        (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(midFrame, f"camera motion ({dx:.2f}, {dy:.2f}) px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        outPath = config.outputsDir / "optical_flow_test.jpg"
        cv2.imwrite(str(outPath), midFrame)
        print(f"saved annotated frame to {outPath}")
        print("yellow arrows are tracked features, green is the median camera motion vector")