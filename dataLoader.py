# dataset gt reader

import configparser
import numpy as np
import cv2
import config


def readSeqInfo(seqPath):
    iniPath = seqPath / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(iniPath)
    s = parser["Sequence"]
    return {
        "name": s.get("name"),
        "imDir": s.get("imDir"),
        "frameRate": int(s.get("frameRate")),
        "seqLength": int(s.get("seqLength")),
        "imWidth": int(s.get("imWidth")),
        "imHeight": int(s.get("imHeight")),
        "imExt": s.get("imExt"),
    }


def readGroundTruth(seqPath):
    gtPath = seqPath / "gt" / "gt.txt"
    if not gtPath.exists():
        return {}

    raw = np.loadtxt(gtPath, delimiter=",")
    boxes = {}
    for row in raw:
        frameId = int(row[0])
        trackId = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        if frameId not in boxes:
            boxes[frameId] = []
        boxes[frameId].append((trackId, x, y, w, h))
    return boxes


def loadSequence(seqPath):
    info = readSeqInfo(seqPath)
    gt = readGroundTruth(seqPath)
    imgDir = seqPath / info["imDir"]
    ext = info["imExt"]

    for frameId in range(1, info["seqLength"] + 1):
        framePath = imgDir / f"{frameId:06d}{ext}"
        frame = cv2.imread(str(framePath))
        if frame is None:
            print(f"could not read frame {frameId} at {framePath}")
            continue

        yield {
            "frameId": frameId,
            "frame": frame,
            "boxes": gt.get(frameId, []),
        }


def loadVideo(videoPath):
    """yields frame dicts from a video file. boxes will always be empty since videos have no gt."""
    cap = cv2.VideoCapture(str(videoPath))
    if(not cap.isOpened()):
        print(f"could not open video at {videoPath}")
        return

    frameId = 1
    while(True):
        ret, frame = cap.read()
        if(not ret):
            break
        yield {
            "frameId": frameId,
            "frame": frame,
            "boxes": [],
        }
        frameId += 1

    cap.release()


def videoFrameRate(videoPath):
    """reads the frame rate from a video file, falls back to the config default"""
    cap = cv2.VideoCapture(str(videoPath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if(fps <= 0):
        return float(config.frameRate)
    return float(fps)


if __name__ == "__main__":
    seqPath = config.testSequencePath
    print(f"loading sequence from {seqPath}")
    info = readSeqInfo(seqPath)
    print("sequence info")
    print(f"  name {info['name']}")
    print(f"  frame rate {info['frameRate']} fps")
    print(f"  length {info['seqLength']} frames")
    print(f"  resolution {info['imWidth']}x{info['imHeight']}")
    gt = readGroundTruth(seqPath)
    print(f"ground truth loaded for {len(gt)} frames")
    print("iterating first 5 frames")
    count = 0
    for data in loadSequence(seqPath):
        print(f"  frame {data['frameId']}, image shape {data['frame'].shape}, {len(data['boxes'])} boxes")
        count += 1
        if count >= 5:
            break

    print("dataLoader self-check done")