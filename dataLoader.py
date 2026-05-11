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