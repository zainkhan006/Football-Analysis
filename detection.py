# feature 1

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
    print("loading yolov8n model")
    model = loadModel()
    print("model loaded")
    print("loading first frame of test sequence")
    seqGen = dataLoader.loadSequence(config.testSequencePath)
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
    print("red = ground truth, green = yolo, open the file to compare")