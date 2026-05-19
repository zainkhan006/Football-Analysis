#integration

"""
Integration pipeline for the football analysis project.

This is a thin orchestrator. It does not contain feature logic itself.
It shows a numbered menu, asks which clip to run on, then hands off to
the relevant feature module.

Feature to clip mapping:
  features 1, 2, 3, 4, 8, 9, 11  -> work on both SportsMOT and staticCam
  features 5, 7                  -> SportsMOT sequences only
  features 6, 10                 -> staticCam videos only

Run:
    python pipeline.py
"""

import sys
import subprocess
from pathlib import Path
import config


# the three staticCam clips and the homography file each one pairs with
staticCamClips = [
    ("tactical_clip_A.mp4", "tactical_clip_A.npz"),
    ("tactical_clipB.mp4", "tactical_clipB.npz"),
    ("tactical_clipC.mp4", "tactical_clipC.npz"),
]

featureNames = {
    1: "Player Detection",
    2: "Multi-Object Tracking",
    3: "Pitch Homography",
    4: "Optical Flow Camera Motion",
    5: "Player Movement Analysis",
    6: "Zone Heatmap",
    7: "Touch Detection",
    8: "Team Assignment",
    9: "Jersey Colour / GK & Referee",
    10: "Formation Detection",
    11: "Ball Detection & Tracking",
}

# features locked to one clip type
sportsmotOnly = {5, 7}
staticCamOnly = {6, 10}


def printMenu():
    """shows the feature list to the user"""
    print()
    print("football analysis pipeline")
    print("choose a feature to run")
    for num in sorted(featureNames.keys()):
        tag = ""
        if(num in sportsmotOnly):
            tag = "  (SportsMOT only)"
        elif(num in staticCamOnly):
            tag = "  (staticCam only)"
        print(f"  {num:2d}  {featureNames[num]}{tag}")
    print("   0  exit")
    print()


def askSportsmotSequence():
    """prompts the user for a SportsMOT sequence folder path"""
    raw = input("enter the SportsMOT sequence folder path: ").strip().strip('"')
    seqPath = Path(raw)
    if(not seqPath.is_dir()):
        print(f"folder not found at {seqPath}")
        return None
    return seqPath


def askStaticCamClip():
    """shows the staticCam clip list and returns the chosen video and homography paths"""
    print("available staticCam clips")
    for i, (videoName, homoName) in enumerate(staticCamClips):
        print(f"  {i + 1}  {videoName}")
    choice = input("enter the clip number: ").strip()
    if(not choice.isdigit()):
        print("that is not a number")
        return None, None
    idx = int(choice) - 1
    if(idx < 0 or idx >= len(staticCamClips)):
        print("that number is not in the list")
        return None, None
    videoName, homoName = staticCamClips[idx]
    videoPath = config.projectRoot / "staticCam" / videoName
    homoPath = config.projectRoot / "homographies" / homoName
    if(not videoPath.is_file()):
        print(f"video not found at {videoPath}")
        return None, None
    if(not homoPath.is_file()):
        print(f"homography not found at {homoPath}")
        return None, None
    return videoPath, homoPath


def askClipType():
    """asks the user whether to use a SportsMOT sequence or a staticCam clip"""
    print("select clip type")
    print("  1  SportsMOT sequence")
    print("  2  staticCam video")
    choice = input("enter your choice: ").strip()
    if(choice == "1"):
        seqPath = askSportsmotSequence()
        if(seqPath is None):
            return None
        return ("sportsmot", seqPath, None)
    elif(choice == "2"):
        videoPath, homoPath = askStaticCamClip()
        if(videoPath is None):
            return None
        return ("staticcam", videoPath, homoPath)
    else:
        print("that is not a valid choice")
        return None


def runModule(scriptName, args):
    """runs a feature module as a subprocess so each one stays self-contained"""
    cmd = [sys.executable, scriptName] + [str(a) for a in args]
    print(f"running {scriptName}")
    subprocess.run(cmd)


def runFeature1():
    """player detection - works on both clip types"""
    clip = askClipType()
    if(clip is None):
        return
    print("feature 1 runs through its own module")
    runModule("detection.py", [])


def runFeature2():
    """tracking - works on both clip types"""
    clip = askClipType()
    if(clip is None):
        return
    runModule("tracking.py", [])


def runFeature3():
    """homography viewer - works on both clip types"""
    clip = askClipType()
    if(clip is None):
        return
    kind, path, homoPath = clip
    if(kind == "sportsmot"):
        runModule("view_homography.py", ["--seq", path.name])
    else:
        runModule("view_homography_video.py", [path, homoPath])


def runFeature4():
    """optical flow camera motion - works on both clip types"""
    clip = askClipType()
    if(clip is None):
        return
    runModule("opticalFlow.py", [])


def runFeature5():
    """player movement analysis - SportsMOT only"""
    seqPath = askSportsmotSequence()
    if(seqPath is None):
        return
    runModule("movement.py", [])


def runFeature6():
    """zone heatmap - staticCam only"""
    videoPath, homoPath = askStaticCamClip()
    if(videoPath is None):
        return
    runModule("heatmap.py", [videoPath, homoPath])


def runFeature7():
    """touch detection - SportsMOT only"""
    seqPath = askSportsmotSequence()
    if(seqPath is None):
        return
    runModule("touch_pipeline.py", ["--seq", seqPath.name])


def runFeature8():
    """team assignment - works on both clip types"""
    clip = askClipType()
    if(clip is None):
        return
    runModule("feature_8.py", [])


def runFeature9():
    """jersey colour and gk / referee - depends on feature 8, runs it first then runs feature 9"""
    clip = askClipType()
    if(clip is None):
        return
    print("feature 9 depends on feature 8, running feature 8 first")
    runModule("feature_8.py", [])
    print("feature 8 done, now running feature 9")
    runModule("feature_9.py", [])


def runFeature10():
    """formation detection - staticCam only"""
    videoPath, homoPath = askStaticCamClip()
    if(videoPath is None):
        return
    runModule("feature_10.py", [videoPath, homoPath])


def runFeature11():
    """ball detection and tracking - works on both clip types"""
    runModule("play_ball_detection.py", [])


featureRunners = {
    1: runFeature1,
    2: runFeature2,
    3: runFeature3,
    4: runFeature4,
    5: runFeature5,
    6: runFeature6,
    7: runFeature7,
    8: runFeature8,
    9: runFeature9,
    10: runFeature10,
    11: runFeature11,
}


if(__name__ == "__main__"):
    while(True):
        printMenu()
        selection = input("enter a feature number: ").strip()
        if(selection == "0"):
            print("exiting pipeline")
            break
        if(not selection.isdigit()):
            print("that is not a number, try again")
            continue
        num = int(selection)
        if(num not in featureRunners):
            print("that feature number does not exist, try again")
            continue
        featureRunners[num]()
        print()
        print("feature run complete, returning to menu")