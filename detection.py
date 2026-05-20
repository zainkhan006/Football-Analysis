# feature 1

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import config
import dataLoader


def loadModel():
    modelPath = config.yoloPlayerModel
    if modelPath.exists():
        return YOLO(str(modelPath))

    print(f"model not found at {modelPath}, falling back to ultralytics auto-download")
    return YOLO("yolov8n.pt")


def detect(frame, model):
    results = model(frame, classes=[0], conf=config.yoloConfidence, imgsz=config.imgsz, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        detections.append({
            "bbox": (x, y, w, h),
            "confidence": conf,
        })
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="feature 1 player detection")
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
    model = loadModel()
    print("model loaded")

    if(isVideo):
        print(f"loading first frame of video {inputPath.name}")
        seqGen = dataLoader.loadVideo(inputPath)
    else:
        print(f"loading first frame of sportsmot sequence {inputPath.name}")
        seqGen = dataLoader.loadSequence(inputPath)

    data = next(seqGen)
    frame = data["frame"].copy()
    gtBoxes = data["boxes"]
    print(f"gt has {len(gtBoxes)} boxes")
    print("running yolo detection")
    detections = detect(frame, model)
    print(f"yolo detected {len(detections)} players")

    for trackId, x, y, w, h in gtBoxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    outPath = config.outputsDir / "detection_test.jpg"
    cv2.imwrite(str(outPath), frame)
    print(f"saved annotated frame to {outPath}")
    if(isVideo):
        print("green = yolo detections, no gt available for video input")
    else:
        print("red = ground truth, green = yolo, open the file to compare")