#feature 2

#feature 2

import argparse
from pathlib import Path
import cv2
import config
import dataLoader
import detection


def trackYolo(frame, model):
    results = model.track(frame, classes=[0], conf=config.yoloConfidence, imgsz=config.imgsz, persist=True, tracker="bytetrack.yaml", verbose=False)
    tracks = []
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return tracks

    ids = boxes.id.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        tracks.append({
            "trackId": int(ids[i]),
            "bbox": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            "confidence": float(confs[i]),
        })
    return tracks


def trackFromGt(gtBoxes):
    tracks = []
    for trackId, x, y, w, h in gtBoxes:
        tracks.append({
            "trackId": int(trackId),
            "bbox": (x, y, w, h),
            "confidence": 1.0,
        })
    return tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="feature 2 multi-object tracking")
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

    print("loading yolov8n model")
    model = detection.loadModel()
    print("model loaded")

    numFrames = 30

    if(isVideo):
        print(f"tracking first {numFrames} frames of video {inputPath.name} with yolo + bytetrack")
        frameGen = dataLoader.loadVideo(inputPath)
    else:
        print(f"tracking first {numFrames} frames of sportsmot sequence {inputPath.name} with yolo + bytetrack")
        frameGen = dataLoader.loadSequence(inputPath)

    firstFrameTracks = []
    lastFrameTracks = []
    lastFrame = None
    seenIds = set()

    for i, data in enumerate(frameGen):
        if(i >= numFrames):
            break
        tracks = trackYolo(data["frame"], model)
        for t in tracks:
            seenIds.add(t["trackId"])
        if(i == 0):
            firstFrameTracks = tracks
        lastFrameTracks = tracks
        lastFrame = data["frame"].copy()

    firstIds = set(t["trackId"] for t in firstFrameTracks)
    lastIds = set(t["trackId"] for t in lastFrameTracks)
    persisted = firstIds & lastIds
    print(f"saw {len(seenIds)} unique track ids across {numFrames} frames")
    print(f"frame 1 had {len(firstIds)} ids, frame {numFrames} has {len(lastIds)} ids")
    print(f"  {len(persisted)} ids persisted across the full window")

    for t in lastFrameTracks:
        x, y, w, h = t["bbox"]
        tid = t["trackId"]
        cv2.rectangle(lastFrame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(lastFrame, f"id {tid}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    outPath = config.outputsDir / "tracking_test.jpg"
    cv2.imwrite(str(outPath), lastFrame)
    print(f"saved annotated frame to {outPath}")
    print("open the file to verify track ids are drawn next to players")