# %% [markdown]
# # SportsMOT EDA — Football Sequences
# **Milestone 1: Exploratory Data Analysis**
#
# Dataset : SportsMOT (ICCV 2023) — Football split only
# Format  : MOT Challenge 17
# License : Creative Commons Attribution-NonCommercial 4.0 International

# %% [markdown]
# ## Cell 0 — Imports & Configuration

# %%
import os
import configparser
import random
import warnings
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\Zain Ul Ibad\Desktop\projects\cv_project")
DATASET_ROOT = PROJECT_ROOT / "sportsmot_publish" / "dataset"
SPLITS_TXT   = PROJECT_ROOT / "sportsmot_publish" / "splits_txt"
FIGURES_DIR  = PROJECT_ROOT / "milestone 1" / "eda_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "savefig.bbox":    "tight",
    "savefig.dpi":     180,
    "savefig.pad_inches": 0.15,
})
SPLIT_COLOURS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

SAMPLE_PER_SEQ = 5   # frames sampled per sequence for pixel analyses

print(" Configuration complete.")
print(f"  Figures → {FIGURES_DIR}")


# %% [markdown]
# ## Cell 1 — Load Dataset Index

# %%
def read_txt_list(path: Path) -> set:
    """Return a set of stripped sequence names from a splits .txt file."""
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}

def parse_seqinfo(seqinfo_path: Path) -> dict:
    """Parse a MOT-format seqinfo.ini and return a flat dict of fields."""
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path)
    sec = cfg["Sequence"]
    return {
        "frame_rate": int(sec.get("frameRate",  25)),
        "seq_length": int(sec.get("seqLength",   0)),
        "im_width":   int(sec.get("imWidth",      0)),
        "im_height":  int(sec.get("imHeight",     0)),
        "im_ext":     sec.get("imExt", ".jpg"),
    }

football_seqs = read_txt_list(SPLITS_TXT / "football.txt")

records = []
for split in ["train", "val", "test"]:
    split_dir = DATASET_ROOT / split
    for seq_name in sorted(os.listdir(split_dir)):
        if seq_name not in football_seqs:
            continue
        seq_path    = split_dir / seq_name
        seqinfo_ini = seq_path / "seqinfo.ini"
        gt_path     = seq_path / "gt" / "gt.txt"
        img_dir     = seq_path / "img1"
        if not seqinfo_ini.exists():
            continue
        meta = parse_seqinfo(seqinfo_ini)
        records.append({
            "seq_name": seq_name,
            "split":    split,
            "seq_path": seq_path,
            "img_dir":  img_dir,
            "gt_path":  gt_path if gt_path.exists() else None,
            **meta,
        })

seq_df = pd.DataFrame(records)

GT_COLS = ["frame_id", "track_id", "x", "y", "w", "h", "conf", "class", "visibility"]

gt_records = []
for _, row in seq_df[seq_df["gt_path"].notna()].iterrows():
    try:
        df = pd.read_csv(row["gt_path"], header=None, names=GT_COLS)
        df["seq_name"] = row["seq_name"]
        df["split"]    = row["split"]
        gt_records.append(df)
    except Exception as e:
        print(f"  ⚠ Could not read {row['gt_path']}: {e}")

gt_df = pd.concat(gt_records, ignore_index=True) if gt_records else pd.DataFrame(columns=GT_COLS)
gt_df["aspect_ratio"] = gt_df["w"] / gt_df["h"].replace(0, np.nan)
gt_df["area"]         = gt_df["w"] * gt_df["h"]
gt_df["cx"]           = gt_df["x"] + gt_df["w"] / 2
gt_df["cy"]           = gt_df["y"] + gt_df["h"] / 2

split_counts = seq_df.groupby("split")["seq_name"].count().reindex(["train", "val", "test"])
split_frames = seq_df.groupby("split")["seq_length"].sum().reindex(["train", "val", "test"])

print(f" Dataset index built.")
print(f"  Football sequences : {len(seq_df)}  "
      f"(train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']})")
print(f"  GT annotations     : {len(gt_df):,}")


# %% [markdown]
# ---
# ## Cell 2 — Analysis 1: Dataset Summary

# %%
total_seqs   = len(seq_df)
total_frames = seq_df["seq_length"].sum()
total_boxes  = len(gt_df)
avg_len      = seq_df["seq_length"].mean()
avg_fps      = seq_df["frame_rate"].mean()
duration_hrs = (total_frames / avg_fps) / 3600

# FIX: use distinct keys so neither row overwrites the other
summary_rows = {
    "Total sequences (football)":           f"{total_seqs}",
    "  → Seq. split (Train / Val / Test)":  f"{split_counts['train']} / {split_counts['val']} / {split_counts['test']}",
    "Total frames (all splits)":            f"{total_frames:,}",
    "  → Frame split (Train / Val / Test)": f"{split_frames['train']:,} / {split_frames['val']:,} / {split_frames['test']:,}",
    "GT annotations (train + val)":         f"{total_boxes:,}",
    "Average sequence length (frames)":     f"{avg_len:.0f}",
    "Frame rate":                           f"{avg_fps:.0f} FPS",
    "Approx. total duration":               f"{duration_hrs:.2f} hours",
    "Resolution":                           f"{seq_df['im_width'].iloc[0]} × {seq_df['im_height'].iloc[0]} px",
    "Data source":                          "Olympic Games & YouTube (25 FPS, 720P)",
    "License":                              "CC BY-NC 4.0",
}

print("=" * 58)
print("Dataset Summary")
print("=" * 58)
for k, v in summary_rows.items():
    print(f"  {k:<46} {v}")
print("=" * 58)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.axis("off")
table_data = [[k, v] for k, v in summary_rows.items()]
tbl = ax.table(
    cellText=table_data,
    colLabels=["Attribute", "Value"],
    loc="center",
    cellLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 1.65)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#eaf0fb")
    # Make indented sub-rows slightly lighter
    if r > 0 and table_data[r - 1][0].startswith("  →"):
        cell.set_facecolor("#f5f5f5")
ax.set_title("SportsMOT Dataset Summary",
             fontsize=14, fontweight="bold", pad=14)
fig.savefig(FIGURES_DIR / "fig1_dataset_summary.png")
plt.show()
print(" Saved fig1_dataset_summary.png")


# %% [markdown]
# ---
# ## Cell 3 — Analysis 2: Class Distribution

# %%
# FIX: remove the confusing twin y-axis approach.
# Left plot: side-by-side bars for sequence count AND frame count per split,
# using a secondary axis only to scale frames — but clearly labelled.
# Right plot: use integer bins since track counts are integers.

tracks_per_seq = (
    gt_df.groupby("seq_name")["track_id"]
    .nunique()
    .reset_index(name="n_tracks")
    .merge(seq_df[["seq_name", "split"]], on="seq_name")
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── (a) Sequences per split (simple, no twin axis confusion) ─────────────────
split_order = ["train", "val", "test"]
seq_vals    = [split_counts[s] for s in split_order]
frame_vals  = [split_frames[s] for s in split_order]
colours     = [SPLIT_COLOURS[s] for s in split_order]
x = np.arange(len(split_order))
w = 0.35

bars_seq   = axes[0].bar(x - w/2, seq_vals, width=w, color=colours,
                          alpha=0.95, edgecolor="white", linewidth=0.8,
                          label="Sequences")
ax_right   = axes[0].twinx()
bars_frame = ax_right.bar(x + w/2, frame_vals, width=w, color=colours,
                           alpha=0.50, edgecolor="white", linewidth=0.8,
                           label="Frames")

# Annotate sequence bars (above bar)
for bar, val in zip(bars_seq, seq_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 str(int(val)),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

# Annotate frame bars
for bar, val in zip(bars_frame, frame_vals):
    ax_right.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 100,
                  f"{int(val):,}",
                  ha="center", va="bottom", fontsize=9, color="#555555")

axes[0].set_xticks(x)
axes[0].set_xticklabels([s.capitalize() for s in split_order])
axes[0].set_ylabel("Number of Sequences", fontsize=11)
axes[0].set_ylim(0, max(seq_vals) * 1.3)
ax_right.set_ylabel("Total Frames", fontsize=11, color="#888888")
ax_right.tick_params(axis="y", labelcolor="#888888")
ax_right.set_ylim(0, max(frame_vals) * 1.3)
axes[0].set_title("(a) Sequences & Frames per Split\n(Football only)", pad=10)

# Combined legend
seq_patch   = mpatches.Patch(color="#666666", alpha=0.95, label="Sequences (left axis)")
frame_patch = mpatches.Patch(color="#666666", alpha=0.45, label="Frames (right axis)")
axes[0].legend(handles=[seq_patch, frame_patch], loc="upper left", fontsize=9)

# ── (b) Unique tracks per sequence — integer bins ────────────────────────────
min_t = int(tracks_per_seq["n_tracks"].min())
max_t = int(tracks_per_seq["n_tracks"].max())
bins  = np.arange(min_t - 0.5, max_t + 1.5, 1)   # one bin per integer value

for split in ["train", "val"]:
    grp = tracks_per_seq[tracks_per_seq["split"] == split]
    axes[1].hist(grp["n_tracks"], bins=bins, alpha=0.75,
                 color=SPLIT_COLOURS[split], label=split.capitalize(),
                 edgecolor="white", linewidth=0.7)

mean_tracks = tracks_per_seq["n_tracks"].mean()
axes[1].axvline(mean_tracks, color="black", linestyle="--",
                linewidth=1.6, label=f"Mean = {mean_tracks:.1f}")
axes[1].set_xlabel("Unique Tracks (Players) per Sequence")
axes[1].set_ylabel("Number of Sequences")
axes[1].set_xticks(range(min_t, max_t + 1))
axes[1].set_title("(b) Distribution of Unique Tracks per Sequence\n(Train + Val, GT-annotated)", pad=10)
axes[1].legend()

fig.suptitle("Class Distribution", fontsize=15,
             fontweight="bold", y=0.967)
fig.tight_layout(pad=2.0)
fig.subplots_adjust(wspace=0.35)
fig.savefig(FIGURES_DIR / "fig2_class_distribution.png")
plt.show()
print(" Saved fig2_class_distribution.png")
print(f"  Unique tracks/seq → mean: {mean_tracks:.1f}, "
      f"min: {tracks_per_seq['n_tracks'].min()}, "
      f"max: {tracks_per_seq['n_tracks'].max()}")


# %% [markdown]
# ---
# ## Cell 4 — Analysis 3: Image Statistics

# %%
print(f"Sampling {SAMPLE_PER_SEQ} frames per sequence for pixel analysis…")

pixel_stats = []

for _, row in seq_df.iterrows():
    all_frames = sorted(row["img_dir"].glob(f"*{row['im_ext']}"))
    if not all_frames:
        continue
    sampled = random.sample(all_frames, min(SAMPLE_PER_SEQ, len(all_frames)))
    for fpath in sampled:
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        flat = img_rgb.flatten()
        pixel_stats.append({
            "seq_name": row["seq_name"],
            "split":    row["split"],
            "mean":     flat.mean(),
            "std":      flat.std(),
            "min":      int(flat.min()),
            "max":      int(flat.max()),
            "median":   float(np.median(flat)),
        })

stats_df = pd.DataFrame(pixel_stats)

print(f"\n  Frames sampled         : {len(stats_df)}")
print(f"  Global pixel mean      : {stats_df['mean'].mean():.2f}")
print(f"  Global pixel std       : {stats_df['std'].mean():.2f}")
print(f"  Global pixel min       : {stats_df['min'].min()}")
print(f"  Global pixel max       : {stats_df['max'].max()}")
print(f"  Median (avg per-frame) : {stats_df['median'].mean():.2f}")

# FIX: scatter is useless when all sequences share one resolution.
# Replace with: (a) resolution summary as annotated text panel,
#               (b) full pixel intensity stats as box plots per split.

n_resolutions = seq_df.groupby(["im_width", "im_height"]).size()
uniform = len(n_resolutions) == 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── (a) Resolution panel ──────────────────────────────────────────────────────
axes[0].axis("off")
if uniform:
    w_val, h_val = n_resolutions.index[0]
    res_text = (
        f"All {len(seq_df)} football sequences share\n"
        f"a uniform resolution of\n\n"
        f"{w_val} × {h_val} px  (720p)\n\n"
        f"No dimension variability exists\n"
        f"within the football subset.\n\n"
        f"Frame rate: {int(avg_fps)} FPS\n"
        f"Avg. seq. length: {avg_len:.0f} frames\n"
        f"Min: {seq_df['seq_length'].min()}  |  "
        f"Max: {seq_df['seq_length'].max()} frames"
    )
    axes[0].text(0.5, 0.5, res_text,
                 ha="center", va="center", fontsize=12,
                 transform=axes[0].transAxes,
                 bbox=dict(boxstyle="round,pad=0.7",
                           facecolor="#eaf0fb", edgecolor="#4C72B0",
                           linewidth=1.5))
else:
    # If multiple resolutions exist, show a scatter with jitter
    jitter = 2
    for _, row_ in seq_df.iterrows():
        axes[0].scatter(
            row_["im_width"]  + np.random.uniform(-jitter, jitter),
            row_["im_height"] + np.random.uniform(-jitter, jitter),
            color=SPLIT_COLOURS[row_["split"]], s=40, alpha=0.75
        )
    for split in ["train", "val", "test"]:
        axes[0].scatter([], [], color=SPLIT_COLOURS[split],
                        label=split.capitalize(), s=50)
    axes[0].legend()
    axes[0].set_xlabel("Image Width (px)")
    axes[0].set_ylabel("Image Height (px)")

axes[0].set_title("(a) Image Dimensions\n(Resolution Profile)", pad=10)

# ── (b) Per-frame mean pixel intensity — box plots per split ─────────────────
split_order_stat = [s for s in ["train", "val", "test"]
                    if s in stats_df["split"].values]
data_by_split = [stats_df[stats_df["split"] == s]["mean"].values
                 for s in split_order_stat]
bp = axes[1].boxplot(data_by_split,
                     labels=[s.capitalize() for s in split_order_stat],
                     patch_artist=True,
                     medianprops=dict(color="black", linewidth=2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))
for patch, split in zip(bp["boxes"], split_order_stat):
    patch.set_facecolor(SPLIT_COLOURS[split])
    patch.set_alpha(0.75)

# Overlay the raw points as a strip
for i, (split, d) in enumerate(zip(split_order_stat, data_by_split), start=1):
    x_jitter = np.random.uniform(-0.12, 0.12, size=len(d))
    axes[1].scatter(i + x_jitter, d,
                    color=SPLIT_COLOURS[split], alpha=0.35, s=12, zorder=3)

axes[1].set_xlabel("Split")
axes[1].set_ylabel("Mean Pixel Intensity (0–255)")
axes[1].set_title("(b) Frame-Level Mean Pixel Intensity\nby Split (box + strip)", pad=10)

global_mean = stats_df["mean"].mean()
axes[1].axhline(global_mean, color="red", linestyle="--", linewidth=1.2,
                label=f"Global mean = {global_mean:.1f}")
axes[1].legend()

fig.suptitle("Image Statistics", fontsize=15,
             fontweight="bold", y=0.967)
fig.tight_layout(pad=2.0)
fig.savefig(FIGURES_DIR / "fig3_image_statistics.png")
plt.show()
print(" Saved fig3_image_statistics.png")


# %% [markdown]
# ---
# ## Cell 5 — Analysis 4: Sample Visualisation (5 × 5 Grid)

# %%
annotated_seqs = seq_df[seq_df["gt_path"].notna()].reset_index(drop=True)
frame_pool = []
for _, row in annotated_seqs.iterrows():
    for fpath in sorted(row["img_dir"].glob(f"*{row['im_ext']}")):
        frame_pool.append((row, fpath))

sampled_frames = random.sample(frame_pool, min(25, len(frame_pool)))

gt_lookup = {}
for seq_name, grp in gt_df.groupby("seq_name"):
    gt_lookup[seq_name] = {}
    for fid, fgrp in grp.groupby("frame_id"):
        gt_lookup[seq_name][fid] = fgrp[["x", "y", "w", "h"]].values.tolist()

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    "Sample Visualisation in a 5x5 grid",
    fontsize=14, fontweight="bold", y=0.995
)

for idx, (row, fpath) in enumerate(sampled_frames):
    ax = fig.add_subplot(5, 5, idx + 1)
    img_bgr = cv2.imread(str(fpath))
    if img_bgr is None:
        ax.axis("off")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)

    frame_id = int(fpath.stem)
    boxes = gt_lookup.get(row["seq_name"], {}).get(frame_id, [])
    for (x, y, w, h) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=1.2, edgecolor="#00ff44", facecolor="none"
        )
        ax.add_patch(rect)

    # Short label: last 14 chars of seq name + frame id + split
    short_name = row["seq_name"][-14:]
    ax.set_title("")
    ax.axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.1, h_pad=0.15, w_pad=0.15)
fig.savefig(FIGURES_DIR / "fig4_sample_grid.png")
plt.show()

avg_boxes = sum(
    len(gt_lookup.get(r["seq_name"], {}).get(int(f.stem), []))
    for r, f in sampled_frames
) / len(sampled_frames)
print(f" Saved fig4_sample_grid.png")
print(f"  Avg GT boxes per sampled frame: {avg_boxes:.1f}")


# %% [markdown]
# ---
# ## Cell 6 — Analysis 5: Colour & Texture Analysis

# %%
random.seed(42)
COLOUR_SAMPLE = 3

print(f"Loading {COLOUR_SAMPLE} frames per sequence for colour analysis…")

r_hist = np.zeros(256, dtype=np.int64)
g_hist = np.zeros(256, dtype=np.int64)
b_hist = np.zeros(256, dtype=np.int64)
per_frame_ch = []

for _, row in seq_df.iterrows():
    all_frames = sorted(row["img_dir"].glob(f"*{row['im_ext']}"))
    if not all_frames:
        continue
    sampled = random.sample(all_frames, min(COLOUR_SAMPLE, len(all_frames)))
    for fpath in sampled:
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        r_hist += np.bincount(img_rgb[:, :, 0].flatten(), minlength=256)
        g_hist += np.bincount(img_rgb[:, :, 1].flatten(), minlength=256)
        b_hist += np.bincount(img_rgb[:, :, 2].flatten(), minlength=256)
        per_frame_ch.append({
            "split":  row["split"],
            "R_mean": img_rgb[:, :, 0].mean(),
            "G_mean": img_rgb[:, :, 1].mean(),
            "B_mean": img_rgb[:, :, 2].mean(),
        })

ch_df = pd.DataFrame(per_frame_ch)
bins  = np.arange(256)

# Normalise histograms
r_norm = r_hist / r_hist.sum()
g_norm = g_hist / g_hist.sum()
b_norm = b_hist / b_hist.sum()

r_wmean = (bins * r_norm).sum()
g_wmean = (bins * g_norm).sum()
b_wmean = (bins * b_norm).sum()

# Check for saturated pixel spike at 255 (artefact indicator)
sat_pct_r = r_hist[255] / r_hist.sum() * 100
sat_pct_g = g_hist[255] / g_hist.sum() * 100
sat_pct_b = b_hist[255] / b_hist.sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# ── (a) Channel histograms — clip x at 250 to avoid spike distortion ──────────
# FIX: Blue channel has a saturated-pixel spike at 255.
# We clip the display at 250 and annotate it.
CLIP_AT = 250
axes[0].plot(bins[:CLIP_AT], r_norm[:CLIP_AT],
             color="#c0392b", alpha=0.85, linewidth=1.8, label="Red")
axes[0].plot(bins[:CLIP_AT], g_norm[:CLIP_AT],
             color="#27ae60", alpha=0.85, linewidth=1.8, label="Green")
axes[0].plot(bins[:CLIP_AT], b_norm[:CLIP_AT],
             color="#2980b9", alpha=0.85, linewidth=1.8, label="Blue")

axes[0].set_xlabel("Pixel Intensity (0–255)")
axes[0].set_ylabel("Normalised Frequency", labelpad= 10)
axes[0].set_title("(a) Channel-wise Pixel Intensity Histograms\n(display clipped at 250)", pad=10)
axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.78))

# Weighted mean annotation
axes[0].annotate(
    f"Weighted means:\nR={r_wmean:.1f}, G={g_wmean:.1f}, B={b_wmean:.1f}",
    xy=(0.97, 0.97), xycoords="axes fraction",
    ha="right", va="top", fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
              edgecolor="#cccc00", alpha=0.9)
)

# ── (b) Per-channel mean by split ─────────────────────────────────────────────
split_ch = ch_df.groupby("split")[["R_mean", "G_mean", "B_mean"]].mean()
x_pos      = np.arange(3)
ch_labels  = ["Red", "Green", "Blue"]
bar_width  = 0.22
split_plot_order = [s for s in ["train", "val", "test"] if s in split_ch.index]

for i, split in enumerate(split_plot_order):
    vals = [split_ch.loc[split, "R_mean"],
            split_ch.loc[split, "G_mean"],
            split_ch.loc[split, "B_mean"]]
    bars = axes[1].bar(x_pos + i * bar_width, vals,
                       width=bar_width,
                       label=split.capitalize(),
                       color=SPLIT_COLOURS[split],
                       alpha=0.85, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.8,
                     f"{val:.1f}",
                     ha="center", va="bottom", fontsize=8)

centre_offset = (len(split_plot_order) - 1) * bar_width / 2
axes[1].set_xticks(x_pos + centre_offset)
axes[1].set_xticklabels(ch_labels, fontsize=11)
axes[1].set_ylabel("Mean Pixel Intensity (0–255)")
axes[1].set_title("(b) Per-Channel Mean Intensity\nby Split", pad=10)
axes[1].legend()

fig.suptitle("Colour & Texture Analysis", fontsize=15,
             fontweight="bold", y=0.967)
fig.tight_layout(pad=2.0)
fig.subplots_adjust(bottom=0.10)
if sat_pct_b > 0.05:
    fig.text(
        0.5, 0.02,
        f"Note: Blue channel contains {sat_pct_b:.2f}% saturated pixels at intensity = 255. "
        f"Display clipped at 250 to preserve histogram readability.",
        ha="center", va="bottom", fontsize=9.5, color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecea",
                  edgecolor="#c0392b", alpha=0.85)
    )
fig.savefig(FIGURES_DIR / "fig5_colour_texture.png")
plt.show()
print(f" Saved fig5_colour_texture.png")
print(f"Weighted means → R: {r_wmean:.1f},  G: {g_wmean:.1f},  B: {b_wmean:.1f}")
print(f"  Saturated px @ 255 → R: {sat_pct_r:.3f}%,  G: {sat_pct_g:.3f}%,  B: {sat_pct_b:.3f}%")


# %% [markdown]
# ---
# ## Cell 7 — Analysis 6: Correlation / Heatmaps

# %%
# visibility is constant (all 1s) in SportsMOT football GT — excluded from corr matrix
corr_cols = ["w", "h", "aspect_ratio", "area", "cx", "cy"]

vis_unique = gt_df["visibility"].nunique()
if vis_unique == 1:
    print(f"  Note: 'visibility' is constant ({gt_df['visibility'].iloc[0]}) "
          f"across all annotations — excluded from correlation matrix.")

mean_w, mean_h = gt_df["w"].mean(), gt_df["h"].mean()
ar_clipped     = gt_df["aspect_ratio"].dropna().clip(0, 2)

# ── Figure 6a: BBox geometry ──────────────────────────────────────────────────
fig_a, axes_a = plt.subplots(1, 2, figsize=(15, 5.5))

hb = axes_a[0].hexbin(gt_df["w"], gt_df["h"],
                      gridsize=55, cmap="YlOrRd",
                      mincnt=1, linewidths=0.1)
plt.colorbar(hb, ax=axes_a[0]).set_label("Count", fontsize=10)
axes_a[0].axvline(mean_w, color="#2980b9", linestyle="--", linewidth=1.5,
                  label=f"Mean W = {mean_w:.0f} px")
axes_a[0].axhline(mean_h, color="#27ae60", linestyle="--", linewidth=1.5,
                  label=f"Mean H = {mean_h:.0f} px")
axes_a[0].set_xlabel("Bounding Box Width (px)")
axes_a[0].set_ylabel("Bounding Box Height (px)")
axes_a[0].set_title("(a) BBox Width vs Height\n(density hex-bin, all GT annotations)", pad=10)
axes_a[0].legend(fontsize=9, loc="upper right")

axes_a[1].hist(ar_clipped, bins=60, color="#8e44ad",
               edgecolor="white", linewidth=0.4, alpha=0.85)
axes_a[1].axvline(1.0, color="black", linestyle="--", linewidth=1.4,
                  label="AR = 1 (square)")
axes_a[1].axvline(ar_clipped.mean(), color="#e74c3c", linestyle=":",
                  linewidth=1.8, label=f"Mean AR = {ar_clipped.mean():.2f}")
axes_a[1].set_xlabel("Aspect Ratio  w / h")
axes_a[1].set_ylabel("Number of Annotations")
axes_a[1].set_title("(b) Bounding Box Aspect Ratio Distribution\n(clipped at 2 for readability)", pad=10)
axes_a[1].legend(fontsize=9)

fig_a.suptitle("Bounding Box Geometry", fontsize=15,
               fontweight="bold", y=0.967)
fig_a.tight_layout(pad=2.0)
fig_a.subplots_adjust(wspace=0.45)
fig_a.savefig(FIGURES_DIR / "fig6a_bbox_geometry.png")
plt.show()

# ── Figure 6b: Correlation matrix ────────────────────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(7, 6.5))
fig_b.subplots_adjust(left=0.18, right=0.88)

corr_matrix = gt_df[corr_cols].dropna().corr()
sns.heatmap(
    corr_matrix,
    ax=ax_b,
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.8,
    annot_kws={"size": 11},
    cbar_kws={"shrink": 0.82, "label": "Pearson r"},
)
ax_b.set_title("Correlation Matrix of BBox Geometric Features", pad=14, fontsize=13)
ax_b.tick_params(axis="x", rotation=35, labelsize=11)
ax_b.tick_params(axis="y", rotation=0,  labelsize=11)

fig_b.suptitle("Correlation Heatmap", fontsize=15,
               fontweight="bold", y=0.967)
fig_b.tight_layout(pad=2.0)
fig_b.subplots_adjust(bottom=0.18)
fig_b.savefig(FIGURES_DIR / "fig6b_correlation_heatmap.png")
plt.show()

print(f" Saved fig6a_bbox_geometry.png and fig6b_correlation_heatmap.png")
print(f"  BBox → mean W: {mean_w:.1f} px, mean H: {mean_h:.1f} px, "
      f"mean AR: {ar_clipped.mean():.2f}")


# %% [markdown]
# ---
# ## Cell 8 — Analysis 7: Data Quality

# %%
quality_issues = []

print("Running data quality audit…")

for _, row in seq_df.iterrows():
    issues = []
    seq_path = row["seq_path"]
    split    = row["split"]

    if not (seq_path / "seqinfo.ini").exists():
        issues.append("missing_seqinfo")

    if not row["img_dir"].exists():
        issues.append("missing_img1_dir")
    else:
        actual_frames = list(row["img_dir"].glob(f"*{row['im_ext']}"))
        n_actual   = len(actual_frames)
        n_expected = row["seq_length"]
        if n_actual != n_expected:
            issues.append(f"frame_count_mismatch(expected={n_expected},found={n_actual})")

        names = [f.name for f in actual_frames]
        if len(names) != len(set(names)):
            issues.append("duplicate_frame_filenames")

        check = random.sample(actual_frames, min(2, len(actual_frames)))
        for fpath in check:
            if cv2.imread(str(fpath)) is None:
                issues.append(f"corrupt_image({fpath.name})")

    if split in ("train", "val"):
        gt_p = seq_path / "gt" / "gt.txt"
        if not gt_p.exists():
            issues.append("missing_gt_txt")
        elif gt_p.stat().st_size == 0:
            issues.append("empty_gt_txt")
        else:
            try:
                tmp = pd.read_csv(gt_p, header=None, names=GT_COLS)
                bad = ((tmp["w"] <= 0) | (tmp["h"] <= 0)).sum()
                if bad > 0:
                    issues.append(f"invalid_bbox_dims({bad})")
            except Exception:
                issues.append("gt_parse_error")

    quality_issues.append({
        "seq_name": row["seq_name"],
        "split":    split,
        "n_issues": len(issues),
        "issues":   "; ".join(issues) if issues else "none",
    })

quality_df = pd.DataFrame(quality_issues)
flagged    = quality_df[quality_df["n_issues"] > 0]

print(f"\n  Sequences audited    : {len(quality_df)}")
print(f"  Sequences clean      : {(quality_df['n_issues'] == 0).sum()}")
print(f"  Sequences with issues: {len(flagged)}")
if len(flagged):
    for _, r in flagged.iterrows():
        print(f"    {r['seq_name']}  [{r['split']}]  → {r['issues']}")

# ── Build issue type counts ────────────────────────────────────────────────────
all_issue_types = []
for row_ in quality_df["issues"]:
    if row_ != "none":
        for issue in row_.split(";"):
            all_issue_types.append(issue.strip().split("(")[0])
issue_counts = pd.Series(all_issue_types).value_counts() if all_issue_types else pd.Series(dtype=int)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# FIX: when no issues exist, show a structured audit checklist instead of
# a nearly-blank panel with just a text message.
checks = {
    "seqinfo.ini present":          (quality_df["n_issues"] == 0).sum(),
    "img1/ directory present":      len(quality_df),
    "Frame count matches seqinfo":  len(quality_df),
    "No duplicate filenames":       len(quality_df),
    "No corrupt images (sampled)":  len(quality_df),
    "gt.txt present (train/val)":   int(seq_df["gt_path"].notna().sum()),
    "No invalid bbox dims":         int(seq_df["gt_path"].notna().sum()),
}

if len(issue_counts) == 0:
    # All-clear checklist
    axes[0].axis("off")
    y_positions = np.linspace(0.88, 0.10, len(checks))
    axes[0].text(0.5, 0.97, "Audit Checklist(All Checks Passed)",
                 ha="center", va="top", fontsize=11, fontweight="bold",
                 transform=axes[0].transAxes, color="#1a5e20")

    for (check_name, n_ok), y in zip(checks.items(), y_positions):
        axes[0].text(0.07, y, f"  {check_name}",
                     ha="left", va="center", fontsize=10,
                     transform=axes[0].transAxes, color="#1a5e20")
        axes[0].text(0.93, y, f"{n_ok}/{len(quality_df)}",
                     ha="right", va="center", fontsize=10,
                     transform=axes[0].transAxes, color="#555555")

    axes[0].set_facecolor("#f0faf0")
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#27ae60")
        spine.set_linewidth(1.5)
    axes[0].set_title("(a) Detected Issue Types", pad=10)
else:
    axes[0].barh(issue_counts.index, issue_counts.values,
                 color="#e74c3c", edgecolor="white", linewidth=0.5)
    for i, v in enumerate(issue_counts.values):
        axes[0].text(v + 0.08, i, str(v), va="center", fontsize=10)
    axes[0].set_xlabel("Number of Occurrences")
    axes[0].set_title("(a) Detected Issue Types", pad=10)
    axes[0].invert_yaxis()

# (b) Issue status by split
for i, split in enumerate(["train", "val", "test"]):
    sub       = quality_df[quality_df["split"] == split]
    ok_count  = (~sub["n_issues"].astype(bool)).sum()
    bad_count = sub["n_issues"].astype(bool).sum()
    axes[1].bar(i - 0.2, ok_count,  width=0.35,
                color="#27ae60", alpha=0.85,
                label="No issues" if i == 0 else "")
    axes[1].bar(i + 0.2, bad_count, width=0.35,
                color="#e74c3c", alpha=0.85,
                label="Has issue"  if i == 0 else "")
    axes[1].text(i - 0.2, ok_count  + 0.2, str(ok_count),
                 ha="center", fontsize=11, fontweight="bold")
    axes[1].text(i + 0.2, bad_count + 0.2, str(bad_count),
                 ha="center", fontsize=11, fontweight="bold")

axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["Train", "Val", "Test"])
axes[1].set_ylabel("Number of Sequences")
axes[1].set_title("(b) Issue Status by Split", pad=10)
axes[1].legend()

fig.suptitle("Data Quality", fontsize=15,
             fontweight="bold", y=0.967)
fig.tight_layout(pad=2.0)
fig.savefig(FIGURES_DIR / "fig7_data_quality.png")
plt.show()
print(f"\n Saved fig7_data_quality.png")


# %% [markdown]
# ---
# ## Cell 9 — Final Summary

# %%
print("\n" + "=" * 58)
print("  EDA COMPLETE — Figures saved to:")
print(f"  {FIGURES_DIR}")
print("=" * 58)
for fig_file in sorted(FIGURES_DIR.glob("*.png")):
    size_kb = fig_file.stat().st_size / 1024
    print(f"  {fig_file.name:<42}  ({size_kb:.0f} KB)")
print("=" * 58)
print("\nAll 7 required analyses completed.")