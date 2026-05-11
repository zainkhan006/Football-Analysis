================================================================================
 README — Dataset Download Instructions
 Computer Vision Milestone 1
 Zain Ul Ebad (29217), Lamaan Ali Bakhsh (29233), Mohammad Sameem (29209)
================================================================================

DATASET: SportsMOT
Paper:   "SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports
          Scenes" — Cui et al., ICCV 2023
Format:  MOT Challenge 17


--------------------------------------------------------------------------------
 WHAT YOU NEED
--------------------------------------------------------------------------------

Only the football sequences are used in this project. The full SportsMOT
download includes basketball and volleyball as well, so expect roughly 30 GB
total. If disk space is tight, the football-only sequences are approximately
10 GB.


--------------------------------------------------------------------------------
 STEP 1 — Register on CodaLab
--------------------------------------------------------------------------------

SportsMOT is hosted on the CodaLab benchmark platform. You need a free account
to access the download links.

1. Go to: https://codalab.lisn.upsaclay.fr/competitions/12480
2. Click "Participate" and register with any email address.
3. Once registered, click "Get Data" under the Participate tab.
   The download links will appear on that page.


--------------------------------------------------------------------------------
 STEP 2 — Download the dataset
--------------------------------------------------------------------------------

The dataset is split into three ZIP archives: train, val, and test.
Download all three from the CodaLab page.

Alternatively, the dataset is also mirrored on Hugging Face:
https://huggingface.co/datasets/MCG-NJU/SportsMOT

And on the official GitHub repository:
https://github.com/MCG-NJU/SportsMOT

The GitHub README contains direct download links to OneDrive and
Baidu Netdisk (password: 4dnw) if CodaLab registration is unavailable.


--------------------------------------------------------------------------------
 STEP 3 — Extract and organise
--------------------------------------------------------------------------------

Extract all three archives into a single folder called sportsmot_publish/.
The final directory structure should look like this:

    cv_project/
    └── sportsmot_publish/
        ├── dataset/
        │   ├── train/
        │   │   └── v_<sequence_name>/
        │   │       ├── img1/          ← JPEG frames (000001.jpg, 000002.jpg, ...)
        │   │       ├── gt/
        │   │       │   └── gt.txt     ← Ground truth annotations
        │   │       └── seqinfo.ini    ← Sequence metadata
        │   ├── val/
        │   │   └── (same structure as train)
        │   └── test/
        │       └── v_<sequence_name>/
        │           ├── img1/          ← Frames only, no gt/ folder
        │           └── seqinfo.ini
        ├── splits_txt/
        │   ├── football.txt           ← Names of all football sequences
        │   ├── basketball.txt
        │   ├── volleyball.txt
        │   ├── train.txt
        │   ├── val.txt
        │   └── test.txt
        └── scripts/
            ├── mot_to_coco.py
            └── sportsmot_to_trackeval.py

The test split has no gt/ folder. This is expected — annotations for the
test set are withheld for the online leaderboard evaluation.


--------------------------------------------------------------------------------
 STEP 4 — Verify the download
--------------------------------------------------------------------------------

Run the EDA script (eda.py) from the milestone 1/ directory:

    python eda.py

The script will read football.txt, walk the dataset directory, and print a
summary. If the dataset is correctly placed, you should see:

    Football sequences : 80  (train=15, val=15, test=50)
    GT annotations     : 262,343

If the sequence count is lower than 80, check that all three archives were
extracted into the same sportsmot_publish/dataset/ folder and that the
directory names match the entries in splits_txt/football.txt exactly.


--------------------------------------------------------------------------------
 ANNOTATION FORMAT
--------------------------------------------------------------------------------

Each line in gt/gt.txt has nine comma-separated fields:

    frame_id, track_id, x, y, w, h, conf, class, visibility

    frame_id    — 1-indexed frame number
    track_id    — unique player ID, persistent across the sequence
    x, y        — top-left pixel coordinate of the bounding box
    w, h        — width and height of the bounding box in pixels
    conf        — annotation confidence flag (always 1 in SportsMOT)
    class       — object class (always 1 for player)
    visibility  — occlusion flag (always 1 in the football subset)


--------------------------------------------------------------------------------
 LICENSE
--------------------------------------------------------------------------------

SportsMOT is released under the Creative Commons Attribution-NonCommercial
4.0 International License (CC BY-NC 4.0).

It may not be used for commercial purposes. Any use must credit the original
authors:

    Cui, Y., Zeng, C., Zhao, X., Yang, Y., Wu, G., & Wang, L. (2023).
    SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes.
    ICCV 2023. https://arxiv.org/abs/2304.05170


--------------------------------------------------------------------------------
 QUESTIONS
--------------------------------------------------------------------------------

If the CodaLab links are broken or the download is unavailable, the dataset
can also be requested directly from the authors via the GitHub issues page:
https://github.com/MCG-NJU/SportsMOT/issues

================================================================================
