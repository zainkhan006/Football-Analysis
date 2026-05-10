"""
Download the required model weights for the pipeline.

Run this ONCE after cloning:
    python setup_models.py

Downloads:
    1. yolov8n.pt           (~6MB)  - COCO pretrained, used as fallback
    2. football-ball-detection.pt   - Roboflow football ball detector
    3. football-pitch-detection.pt  - Roboflow pitch keypoint detector

Models are saved to the project root and models/ directory.
"""

from pathlib import Path
import subprocess
import sys


MODELS_DIR = Path("models")


def ensure_ultralytics():
    """Make sure ultralytics is installed."""
    try:
        import ultralytics
        print(f"[setup] ultralytics {ultralytics.__version__} found")
    except ImportError:
        print("[setup] Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])


def download_yolov8n():
    """Download the base YOLOv8n model (COCO pretrained)."""
    target = Path("yolov8n.pt")
    if target.exists():
        print(f"[setup] {target} already exists ({target.stat().st_size / 1e6:.1f} MB)")
        return
    print("[setup] Downloading yolov8n.pt ...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # auto-downloads from ultralytics hub
    print(f"[setup] yolov8n.pt downloaded ({target.stat().st_size / 1e6:.1f} MB)")


def download_roboflow_models():
    """
    Download the Roboflow football models.

    These are hosted on Roboflow Universe. We use the roboflow pip package
    to pull them. If roboflow isn't installed, we give manual instructions.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    ball_model = MODELS_DIR / "football-ball-detection.pt"
    pitch_model = MODELS_DIR / "football-pitch-detection.pt"

    if ball_model.exists() and pitch_model.exists():
        print(f"[setup] Both Roboflow models already exist:")
        print(f"        {ball_model} ({ball_model.stat().st_size / 1e6:.1f} MB)")
        print(f"        {pitch_model} ({pitch_model.stat().st_size / 1e6:.1f} MB)")
        return

    print()
    print("=" * 65)
    print("  ROBOFLOW MODEL DOWNLOAD")
    print("=" * 65)
    print()
    print("  The two Roboflow models (~137 MB each) are too large for")
    print("  GitHub. You need to download them manually:")
    print()
    print("  1. BALL DETECTION MODEL:")
    print("     Go to: https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg")
    print("     -> 'Model' tab -> Download -> YOLOv8 format")
    print("     Save the .pt file as:")
    print(f"       {ball_model.resolve()}")
    print()
    print("  2. PITCH KEYPOINT MODEL:")
    print("     Go to: https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi")
    print("     -> 'Model' tab -> Download -> YOLOv8 format")
    print("     Save the .pt file as:")
    print(f"       {pitch_model.resolve()}")
    print()
    print("  Alternatively, ask Lamaan for the model files directly")
    print("  (shared via Google Drive / OneDrive / USB).")
    print()
    print("=" * 65)

    # Try automated download via roboflow package
    try:
        import roboflow
        print("\n[setup] roboflow package found — attempting auto-download...")
        print("[setup] NOTE: You may need a Roboflow API key.")
        print("[setup] Get one at https://app.roboflow.com/settings/api")
        api_key = input("  Enter Roboflow API key (or press Enter to skip): ").strip()
        if api_key:
            rf = roboflow.Roboflow(api_key=api_key)

            if not ball_model.exists():
                print("[setup] Downloading ball detection model...")
                project = rf.workspace("roboflow-jvuqo").project("football-ball-detection-rejhg")
                version = project.version(1)
                version.download("yolov8", location=str(MODELS_DIR / "ball_tmp"))
                # Find the .pt file in the download
                print(f"[setup] Check {MODELS_DIR / 'ball_tmp'} for the .pt file")

            if not pitch_model.exists():
                print("[setup] Downloading pitch detection model...")
                project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
                version = project.version(1)
                version.download("yolov8", location=str(MODELS_DIR / "pitch_tmp"))
                print(f"[setup] Check {MODELS_DIR / 'pitch_tmp'} for the .pt file")
        else:
            print("[setup] Skipped — use manual download instructions above.")
    except ImportError:
        print("\n[setup] roboflow package not installed.")
        print("  Install with: pip install roboflow")
        print("  Or download models manually using the links above.")


def check_dataset():
    """Remind about dataset setup."""
    dataset_dir = Path("dataset/train")
    if dataset_dir.exists():
        clips = [p.name for p in dataset_dir.iterdir() if p.is_dir()]
        print(f"\n[setup] Dataset found: {len(clips)} clips in dataset/train/")
    else:
        print()
        print("=" * 65)
        print("  DATASET SETUP")
        print("=" * 65)
        print()
        print("  The SportsMOT dataset is not included in the repo.")
        print("  Download it from: https://github.com/MCG-NJU/SportsMOT")
        print()
        print("  Expected structure:")
        print("    dataset/train/v_gQNyhv8y0QY_c013/img1/*.jpg")
        print("    dataset/train/v_gQNyhv8y0QY_c013/gt/gt.txt")
        print("    dataset/train/v_gQNyhv8y0QY_c013/seqinfo.ini")
        print("    ... (repeat for each clip)")
        print("=" * 65)


def main():
    print("[setup] Football Analysis — Model Setup")
    print()

    ensure_ultralytics()
    download_yolov8n()
    download_roboflow_models()
    check_dataset()

    print()
    print("[setup] Done! Next steps:")
    print("  1. Make sure both .pt files are in models/")
    print("  2. Make sure dataset is in dataset/train/")
    print("  3. Run: python compute_homographies.py  (generates homographies)")
    print("  4. Run: python run_touch_detection.py   (interactive touch detection)")


if __name__ == "__main__":
    main()
