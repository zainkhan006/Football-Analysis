#paths, thresholds, constants etc iss mai
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

projectRoot = Path(r"C:\Users\Zain Ul Ibad\Desktop\projects\cv_project")
datasetRoot = projectRoot / "sportsmot_publish" / "dataset"
modelsDir = projectRoot / "models"
outputsDir = projectRoot / "outputs"
testSequence = "v_i2_L4qquVg0_c006"
testSplit = "val"
testSequencePath = datasetRoot / testSplit / testSequence
yoloPlayerModel = modelsDir / "yolov8n.pt"
yoloPitchModel = modelsDir / "football-pitch-detection.pt"
yoloBallModel = modelsDir / "football-ball-detection.pt"

# pipeline mode (True = read SportsMOT GT boxes, False = run YOLO detection)
useGt = True

stride = 1
yoloConfidence = 0.25
imgsz = 1280

frameRate = 25
pitchWidthM = 105
pitchHeightM = 68
sprintSpeedThresholdKmh = 25
sprintMinFrames = 10
speedSmoothingWindow = 5
heatmapGaussianSigma = 1.5
heatmapPitchResolution = 1


if __name__ == "__main__":
    print("checking config paths")
    checks = [
        ("project root", projectRoot),
        ("dataset root", datasetRoot),
        ("test sequence", testSequencePath),
        ("models dir", modelsDir),
        ("outputs dir", outputsDir),
    ]
    for name, p in checks:
        if p.exists():
            print(f"  {name} found at {p}")

        else:
            print(f"  {name} missing, expected at {p}")